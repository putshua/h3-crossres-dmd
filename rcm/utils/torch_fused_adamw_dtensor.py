"""CUDA fused AdamW with checkpointed FP32 state on FSDP2 local shards.

The CUDA path uses PyTorch's fused kernel, not Transformer Engine. CPU uses
PyTorch's scalar reference path so the same state format can be tested offline.
"""
from collections import defaultdict

import torch
from torch.optim.adamw import adamw
from torch.distributed.tensor import DTensor


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class FusedAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0., master_weights=True, capturable=True,
                 adam_w_mode=True, bias_correction=True, amsgrad=False):
        if not adam_w_mode or not bias_correction or amsgrad:
            raise ValueError('PyTorch fused backend requires AdamW, bias correction, and amsgrad=False')
        if lr < 0 or eps < 0 or weight_decay < 0 or any(not 0 <= b < 1 for b in betas):
            raise ValueError('Invalid AdamW hyperparameters')
        self.master_weights = master_weights
        self.capturable = capturable
        # Standard GradScaler unscales/checks gradients before calling step.
        # Do not opt into the fused optimizer's custom GradScaler protocol.
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def _initialize_param(self, p):
        state = self.state[p]
        if not state:
            # Preserve DTensor shape/placements for DCP resharding. Only local
            # tensors are passed to the fused kernel below.
            state['step'] = torch.zeros((), device=p.device, dtype=torch.float32)
            state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
            state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)
            if self.master_weights:
                state['master_param'] = p.detach().to(torch.float32).clone()
        return state

    def initialize_state(self):
        """Allocate DCP templates without a fake optimizer step or parameter change."""
        for group in self.param_groups:
            for p in group['params']:
                self._initialize_param(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            # Bound the temporary FP32 gradient conversion to 16M local elements
            # (~64 MiB), instead of materializing a full FP32 gradient shard.
            batch, elements = [], 0
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('FusedAdamW does not support sparse gradients')
                state = self._initialize_param(p)
                n = local(p).numel()
                if not n:
                    # Step is replicated by DCP, even when this rank owns no
                    # rows. Keep its bias-correction counter equal on all ranks.
                    state['step'].add_(1)
                    continue
                if batch and elements + n > 16 * 1024**2:
                    self._update_batch(batch, group)
                    batch, elements = [], 0
                batch.append((p, state))
                elements += n
            if batch:
                self._update_batch(batch, group)
        return loss

    def _update_batch(self, batch, group):
        # Group by device for CPU tests and models spanning multiple devices.
        devices = defaultdict(list)
        for p, state in batch:
            devices[p.device].append((p, state))
        for device, items in devices.items():
            params = [local(s['master_param']) if self.master_weights else local(p).float()
                      for p, s in items]
            grads = [local(p.grad).float() for p, _ in items]
            adamw(params, grads,
                  [local(s['exp_avg']) for _, s in items],
                  [local(s['exp_avg_sq']) for _, s in items], [],
                  [s['step'] for _, s in items],
                  fused=device.type == 'cuda', foreach=False,
                  capturable=self.capturable and device.type == 'cuda',
                  amsgrad=False, beta1=group['betas'][0], beta2=group['betas'][1],
                  lr=group['lr'], weight_decay=group['weight_decay'],
                  eps=group['eps'], maximize=False)
            for (p, _), updated in zip(items, params):
                local(p).copy_(updated)

    @torch.no_grad()
    def sync_model_weights(self):
        if self.master_weights:
            for p, state in self.state.items():
                if 'master_param' in state:
                    local(p).copy_(local(state['master_param']))

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict normally casts floating state to p.dtype.
        # Restore from the original incoming tensors after loading group metadata
        # so no master/moment precision is lost through a BF16 round trip.
        saved_groups = state_dict['param_groups']
        incoming = state_dict['state']
        if self.master_weights:
            for state in incoming.values():
                if state and 'master_param' not in state:
                    raise ValueError('Checkpoint has no FP32 master_param. Start a new run or explicitly migrate the old optimizer; direct resume is not supported.')
        super().load_state_dict(state_dict)
        for saved, current in zip(saved_groups, self.param_groups):
            for ident, p in zip(saved['params'], current['params']):
                if ident not in incoming:
                    continue
                restored = {}
                for key, value in incoming[ident].items():
                    if key == 'step':
                        restored[key] = torch.as_tensor(value, dtype=torch.float32, device=p.device).clone()
                    elif key in ('exp_avg', 'exp_avg_sq', 'master_param'):
                        value = value.to(device=p.device, dtype=torch.float32)
                        if isinstance(p, DTensor):
                            if isinstance(value, DTensor):
                                value = value.redistribute(p.device_mesh, p.placements)
                            else:
                                if value.shape != local(p).shape:
                                    raise ValueError('Optimizer state must be resharded through DCP before loading')
                                value = DTensor.from_local(value, p.device_mesh, p.placements,
                                    shape=p.shape, stride=p.stride(), run_check=False)
                        elif isinstance(value, DTensor):
                            value = value.full_tensor()
                        restored[key] = value.clone()
                    else:
                        restored[key] = value
                self.state[p] = restored
                if self.master_weights and restored:
                    with torch.no_grad():
                        local(p).copy_(local(restored['master_param']))

"""Native H3 distillation model for Bidirectional's ImaginaireTrainer_Distill.

Only H3 network construction, RF objectives and AV batches live here. The
training loop, closure backward, optimizer steps, scheduling and DCP are the
imported Bidirectional implementations.
"""
import os
import time
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist

from imaginaire.model import ImaginaireModel
from imaginaire.lazy_config import instantiate
from imaginaire.utils import distributed, misc, log
from imaginaire.utils.ema import FastEmaModelUpdater
from rcm.utils.optim_instantiate_dtensor import get_base_scheduler
from rcm.configs.h3 import native_config
from h3dmd.data import H3Batch
from h3dmd.distributed import DistributedContext
from h3dmd.models import H3ModelBundle
from h3dmd.methods import build_method
from h3dmd.precision import configure_h3_precision


class H3DistillModel(ImaginaireModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.h3 = native_config(config.h3)
        configure_h3_precision(self.h3)
        log.info(f"H3 precision: allow_tf32={self.h3.training.allow_tf32}; native module dtypes preserved")
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        self.context = DistributedContext(distributed.get_rank(), distributed.get_world_size(),
                                          device.index, device)
        misc.set_random_seed(self.h3.training.seed)
        # CPU checkpoint loading and CUDA copies can finish at very different
        # times across nodes. Keep early ranks out of NCCL forward collectives
        # while their peers still allocate/copy the remaining model roles.
        # A host barrier avoids launching waiting NCCL kernels during loading.
        initialization_group = (
            dist.new_group(backend="gloo", timeout=timedelta(seconds=3600))
            if self.context.world_size > 1 else None
        )
        for name, role in (("net", "student"), ("net_teacher", "teacher"),
                           ("net_fake_score", "critic")):
            started = time.monotonic()
            log.info(f"H3: loading and sharding {role}...")
            setattr(self, name, instantiate(config.net, config=self.h3, role=role, context=self.context))
            if initialization_group is not None:
                dist.monitored_barrier(group=initialization_group, timeout=timedelta(seconds=3600))
            log.info(f"H3: {role} ready in {time.monotonic() - started:.1f}s")
        if initialization_group is not None:
            dist.destroy_process_group(initialization_group)
        self.method = build_method(self.h3, H3ModelBundle(self.net, self.net_teacher, self.net_fake_score))
        if hasattr(self.method, "score_mapper"):
            # Frozen replica: excluded from both optimizers, included in DCP.
            self.score_mapper = self.method.score_mapper
        self.net_ema_worker = FastEmaModelUpdater()
        if config.ema.enabled:
            self.net_ema = instantiate(config.net, config=self.h3, role="ema", context=self.context)
            self.net_ema_worker.copy_to(self.net, self.net_ema)
            s = config.ema.rate
            if not 0 < s < 1:
                raise ValueError("EMA rate must be between zero and one")
            self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
        misc.set_random_seed(self.h3.training.seed, by_rank=True)

    def train(self, mode=True):
        super().train(mode)
        self.net_teacher.eval()
        if hasattr(self, "score_mapper"):
            self.score_mapper.eval()
        if self.config.ema.enabled:
            self.net_ema.eval()
        return self

    def model_dict(self):
        return {"net": self.net, "fake_score": self.net_fake_score}

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # Match upstream distillation names and exclude the immutable teacher.
        if destination is None:
            from collections import OrderedDict
            destination = OrderedDict()
        for name in ("net", "net_fake_score", "net_ema", "score_mapper"):
            if hasattr(self, name):
                getattr(self, name).state_dict(destination=destination, prefix=prefix + name + ".",
                                               keep_vars=keep_vars)
        return destination

    def load_state_dict(self, state_dict, strict=True, assign=False):
        from torch.nn.modules.module import _IncompatibleKeys
        missing, unexpected = [], []
        names = [n for n in ("net", "net_fake_score", "net_ema", "score_mapper") if hasattr(self, n)]
        for name in names:
            prefix = name + "."
            result = getattr(self, name).load_state_dict(
                {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)},
                strict=strict, assign=assign)
            missing.extend(prefix + k for k in result.missing_keys)
            unexpected.extend(prefix + k for k in result.unexpected_keys)
        unexpected.extend(k for k in state_dict if k.split(".")[0] not in names)
        if strict and unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
        return _IncompatibleKeys(missing, unexpected)

    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        self.optimizer_dict = {
            "net": instantiate(optimizer_config, model=self.net),
            "fake_score": instantiate(self.config.optimizer_fake_score, model=self.net_fake_score),
        }
        if hasattr(self.method, "lr_multiplier"):
            self.scheduler_dict = {
                key: torch.optim.lr_scheduler.LambdaLR(
                    opt, lambda update, role=key: self.method.lr_multiplier(role, update))
                for key, opt in self.optimizer_dict.items()
            }
        else:
            self.scheduler_dict = {k: get_base_scheduler(opt, self, scheduler_config)
                                   for k, opt in self.optimizer_dict.items()}

    def _active_keys(self, iteration):
        student, critic = self.method.update_roles(iteration)
        return [key for key, active in (("net", student), ("fake_score", critic)) if active]

    def get_optimizers(self, iteration):
        return [self.optimizer_dict[k] for k in self._active_keys(iteration)]

    def get_lr_schedulers(self, iteration):
        return [self.scheduler_dict[k] for k in self._active_keys(iteration)]

    def gradient_accumulation_steps(self, iteration):
        if (self.h3.data.dmd_manifest and iteration < self.h3.pcm.warmup_steps):
            return self.h3.staged.pcm_gradient_accumulation_steps
        return self.h3.training.gradient_accumulation_steps

    def training_step_closures(self, data_batch, iteration):
        batch = H3Batch(**data_batch)
        closures = list(self.method.training_step_closures(batch, iteration))
        if not closures:
            raise ValueError(f"No H3 objectives at iteration {iteration}")
        for index, objective in enumerate(closures):
            def closure(objective=objective):
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=self.h3.training.autocast and batch.video_latents.is_cuda,
                                    cache_enabled=False):
                    result = objective.compute()
                prefix = f"{objective.role}/{objective.name}"
                metrics = {f"{prefix}/{k}": v.detach() for k, v in result.metrics.items()}
                metrics[f"{prefix}/loss"] = result.loss.detach()
                metrics["training/gradient_accumulation_steps"] = result.loss.new_tensor(
                    self.gradient_accumulation_steps(iteration))
                metrics["training/effective_batch_size"] = result.loss.new_tensor(
                    batch.video_latents.shape[0] * self.context.world_size
                    * self.gradient_accumulation_steps(iteration))
                if hasattr(self.method, "phase"):
                    phase = self.method.phase(iteration)
                    metrics["training/phase"] = result.loss.new_tensor(
                        {"pcm": 0, "fake_warmup": 1, "dmd": 2}[phase])
                return metrics, result.loss
            yield f"{objective.role}/{objective.name}", closure, index == len(closures) - 1

    def on_before_zero_grad(self, optimizer, scheduler, iteration):
        if optimizer is self.optimizer_dict["net"] and self.config.ema.enabled:
            warmup = self.h3.pcm.warmup_steps if self.h3.pcm.enabled else 0
            effective = iteration
            if self.h3.dmd.update_schedule == "alternating" and iteration >= warmup:
                effective = warmup + (iteration - warmup) // self.h3.dmd.student_update_frequency
            effective += self.config.ema.iteration_shift
            beta = 0.0 if effective < 1 else (1 - 1 / (effective + 1)) ** (self.ema_exp_coefficient + 1)
            self.net_ema_worker.update_average(self.net, self.net_ema, beta=beta)
        role = "student" if optimizer is self.optimizer_dict["net"] else "critic"
        self.method.on_after_optimizer_step(iteration, (role,))

"""Numerical accumulation and lossless optimizer-state restore contracts."""
import copy
import pytest
import torch
from rcm.utils.torch_fused_adamw_dtensor import FusedAdamW


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_master_accumulates_small_updates_and_resumes_exactly(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    p = torch.nn.Parameter(torch.tensor([.125, -.125, .25], device=device, dtype=torch.bfloat16))
    ref = torch.nn.Parameter(p.float().detach().clone())
    plain = torch.nn.Parameter(p.detach().clone())
    opt = FusedAdamW([p], lr=1e-6, betas=(0., .999), weight_decay=.01)
    baseline = torch.optim.AdamW([ref], lr=1e-6, betas=(0., .999), weight_decay=.01,
                                fused=device == 'cuda')
    rounded = torch.optim.AdamW([plain], lr=1e-6, betas=(0., .999), weight_decay=.01)
    initial = p.detach().clone()
    for _ in range(400):
        p.grad = torch.ones_like(p)
        ref.grad = p.grad.float()
        plain.grad = p.grad.clone()
        opt.step(); baseline.step(); rounded.step()
    torch.testing.assert_close(opt.state[p]['master_param'], ref, rtol=0, atol=0)
    assert torch.equal(p, ref.bfloat16())
    assert not torch.equal(p, initial)
    assert torch.equal(plain, initial)
    q = torch.nn.Parameter(p.detach().clone())
    resumed = FusedAdamW([q], lr=1e-6, betas=(0., .999), weight_decay=.01)
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    for key in ('master_param', 'exp_avg', 'exp_avg_sq', 'step'):
        assert resumed.state[q][key].dtype == torch.float32
        torch.testing.assert_close(resumed.state[q][key], opt.state[p][key], rtol=0, atol=0)
    for _ in range(5):
        p.grad = torch.full_like(p, .37); q.grad = p.grad.clone()
        opt.step(); resumed.step()
    assert torch.equal(q, p)
    assert torch.equal(resumed.state[q]['master_param'], opt.state[p]['master_param'])


def test_missing_gradient_and_checkpoint_initialization_do_not_step():
    p = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    opt = FusedAdamW([p])
    opt.initialize_state()
    opt.step()
    assert opt.state[p]['step'].item() == 0
    assert torch.equal(p, torch.ones_like(p))


def test_old_optimizer_resume_requires_explicit_migration():
    p = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    old = torch.optim.AdamW([p]); p.grad = torch.ones_like(p); old.step()
    opt = FusedAdamW([p])
    with pytest.raises(ValueError, match='no FP32 master_param'):
        opt.load_state_dict(old.state_dict())

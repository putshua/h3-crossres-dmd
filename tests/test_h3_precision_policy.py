import pytest
import torch
from h3dmd.config import H3Config
from h3dmd.inference import H3InferencePipeline

@pytest.mark.parametrize('enabled', [False, True])
def test_pipeline_applies_configured_precision_not_ambient_state(enabled):
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        config = H3Config()
        config.training.allow_tf32 = enabled
        torch.backends.cuda.matmul.allow_tf32 = not enabled
        torch.backends.cudnn.allow_tf32 = not enabled
        H3InferencePipeline(None, config)
        assert torch.backends.cuda.matmul.allow_tf32 == enabled
        assert torch.backends.cudnn.allow_tf32 == enabled
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn

@pytest.mark.parametrize('enabled,expected', [(False, 5.), (True, 1.)])
def test_clip_switch_preserves_norm_logging(enabled, expected):
    from types import SimpleNamespace
    from rcm.callbacks.h3 import H3TrainingCallback
    param = torch.nn.Parameter(torch.zeros(2))
    param.grad = torch.tensor([3., 4.])
    opt = torch.optim.SGD([param], lr=1)
    cfg = H3Config(); cfg.training.grad_clip_enabled = enabled
    model = SimpleNamespace(h3=cfg, optimizer_dict={'net': opt})
    callback = H3TrainingCallback.__new__(H3TrainingCallback); callback.grad_norms = {}
    callback.on_before_optimizer_step(model,opt,None,SimpleNamespace(unscale_=lambda o: None))
    torch.testing.assert_close(param.grad.norm(), torch.tensor(expected))
    assert callback.grad_norms['grad/student'].item() == 5
    assert callback.grad_norms['grad_clip_coefficient/student'].item() == pytest.approx(
        .2 if enabled else 1.)

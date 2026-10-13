from unittest.mock import patch

import pytest
import torch
from torch import nn

from h3dmd.config import H3Config, load_config
from h3dmd.dmd import H3DMDMethod, H3Latents
from h3dmd.h3.adapter import H3Velocity
from h3dmd.models import H3ModelBundle
from test_dmd import make_batch, make_method


class ConstantRole(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def freeze(self):
        self.requires_grad_(False)
        return self

    def forward(self, video, audio, prompt, **kwargs):
        return H3Velocity(
            torch.ones_like(video) * self.value, torch.ones_like(audio) * self.value
        )


def constant_method(student=0.2, teacher=0.7):
    config = H3Config()
    config.dmd.rf_loss_weight = 0
    config.pcm.enabled = True
    models = H3ModelBundle(
        ConstantRole(student), ConstantRole(teacher), ConstantRole(-0.2)
    )
    return H3DMDMethod(config, models), models


@pytest.mark.parametrize("phase,cell,skip", [(0, 7, 3), (3, 1, 1)])
@pytest.mark.parametrize("grid", ["q", "sigma"])
def test_pcm_analytic_target_gradient_and_zero_endpoint(phase, cell, skip, grid):
    method, models = constant_method()
    method.config.pcm.grid_interpolation = grid
    method.config.pcm.skipping_interval_steps = skip
    with patch("torch.randint", side_effect=[torch.tensor(phase), torch.tensor(cell)]):
        result = method.pcm_loss(make_batch())
    result.loss.backward()
    endpoint = method.scheduler.level(result.metrics["endpoint_q"])
    upper = method.scheduler.level(method.config.pcm.phase_boundaries_q[phase])
    expected_loss = 0
    expected_grad = 0
    for modality in ("video", "audio"):
        e = getattr(endpoint, modality + "_sigma")
        if grid == "sigma":
            width = getattr(upper, modality + "_sigma") - e
            a = e + width * cell / method.config.pcm.steps_per_phase
            b = e + width * (cell - skip) / method.config.pcm.steps_per_phase
        else:
            a = getattr(method.scheduler.level(result.metrics["start_q"]), modality + "_sigma")
            b = getattr(method.scheduler.level(result.metrics["target_q"]), modality + "_sigma")
        torch.testing.assert_close(result.metrics[f"start_{modality}_sigma"], a)
        torch.testing.assert_close(result.metrics[f"target_{modality}_sigma"], b)
        residual = (a - b) * (0.2 - 0.7)
        expected_loss += residual.square()
        expected_grad += 2 * residual * (a - e)
    torch.testing.assert_close(result.loss, expected_loss, rtol=2e-4, atol=1e-7)
    torch.testing.assert_close(
        models.student.value.grad, expected_grad, rtol=2e-4, atol=1e-7
    )
    assert models.teacher.value.grad is None
    assert models.critic.value.grad is None
    if phase == 3:
        assert result.metrics["target_q"] == result.metrics["endpoint_q"] == 0


def test_pcm_shifts_edges_then_subdivides_each_modal_sigma():
    method, _ = constant_method()
    method.scheduler.video_shift = 6.
    method.scheduler.audio_shift = 3.
    lower, upper = torch.tensor(0.), torch.tensor(.25)
    midpoint = method._pcm_level(upper, lower, torch.tensor(.5))
    torch.testing.assert_close(midpoint.video_sigma, torch.tensor(1/3))
    torch.testing.assert_close(midpoint.audio_sigma, torch.tensor(.25))
    torch.testing.assert_close(midpoint.video_timestep, torch.tensor(2/3))
    torch.testing.assert_close(midpoint.audio_timestep, torch.tensor(.75))
    for phase in range(4):
        upper, lower = (torch.tensor(v) for v in method.config.pcm.phase_boundaries_q[phase:phase+2])
        levels = [method._pcm_level(upper, lower, torch.tensor(i/24)) for i in range(25)]
        for modality in ['video', 'audio']:
            values = torch.stack([getattr(level, modality+'_sigma') for level in levels])
            expected = torch.linspace(float(values[0]), float(values[-1]), 25)
            torch.testing.assert_close(values, expected)
            for index, q in [(0, lower), (-1, upper)]:
                torch.testing.assert_close(values[index], getattr(method.scheduler.level(q), modality+'_sigma'))
    method.config.pcm.grid_interpolation = 'q'
    midpoint = method._pcm_level(torch.tensor(.25), torch.tensor(0.), torch.tensor(.5))
    torch.testing.assert_close(midpoint.video_sigma, torch.tensor(6/13))
    torch.testing.assert_close(midpoint.audio_sigma, torch.tensor(.3))


def test_pcm_rejects_unknown_grid_interpolation():
    config = H3Config()
    config.pcm.grid_interpolation = 'invalid'
    with pytest.raises(ValueError, match='grid_interpolation'):
        config.validate()


def test_pcm_exact_constant_flow_is_consistent():
    method, _ = constant_method(student=0.7)
    assert method.pcm_loss(make_batch()).loss.item() < 1e-12


def test_pcm_rollout_source_never_uses_placeholder_values():
    method, _ = constant_method()
    method.config.pcm.data_mode = "rollout"
    batch = make_batch()
    clean = H3Latents(
        torch.randn_like(batch.video_latents), torch.randn_like(batch.audio_latents)
    )
    with patch.object(method, "rollout", return_value=clean) as rollout:
        torch.manual_seed(12)
        first = method.pcm_loss(batch)
        batch.video_latents.fill_(999)
        batch.audio_latents.fill_(-999)
        torch.manual_seed(12)
        second = method.pcm_loss(batch)
    assert rollout.call_count == 2
    torch.testing.assert_close(first.loss, second.loss)


def test_update_schedule_warmup_and_rollout_cycle():
    method, _ = make_method()
    cfg = method.config
    cfg.dmd.rollout_steps = 4
    cfg.dmd.update_schedule = "alternating"
    cfg.dmd.cycle_rollout_steps = True
    assert [method.update_roles(i) for i in range(6)] == [(True, False)] + [
        (False, True)
    ] * 4 + [(True, False)]
    assert [method._rollout_count(i) for i in [0, 5, 10, 15, 20]] == [1, 2, 3, 4, 1]
    assert [method._rollout_count(i, critic=True) for i in [1, 2, 3, 4, 6]] == [
        1,
        2,
        3,
        4,
        1,
    ]
    cfg.pcm.enabled = True
    cfg.pcm.warmup_steps = 3
    cfg.pcm.warmup_only = True
    assert all(method.update_roles(i) == (True, False) for i in range(4))
    assert method.update_roles(4) == (False, True)
    assert [method._rollout_count(i) for i in [3, 8, 13, 18, 23]] == [1, 2, 3, 4, 1]
    assert [method._rollout_count(i, critic=True) for i in [4, 5, 6, 7, 9]] == [1, 2, 3, 4, 1]
    cfg.pcm.warmup_steps = 1
    cfg.dmd.rollout_steps = 8
    assert [method._rollout_count(i) for i in range(1, 41, 5)] == list(range(1, 9))
    cfg.dmd.rf_loss_weight = 0
    assert "pcm/video" in method.student_loss(make_batch(), 0).metrics
    post = method.student_loss(make_batch(), 3).metrics
    assert "pcm/video" not in post and "dmd/video" in post
    cfg.dmd.dmd_loss_weight = 0
    cfg.pcm.warmup_only = False
    assert method.update_roles(4) == (True, False)


def test_defaults_remain_pure_dmd_and_validate_pcm():
    config = load_config("configs/h3_dmd_4step_512x288_synthetic_audio.yaml")
    assert not config.pcm.enabled
    assert config.dmd.rf_loss_weight == 0 and config.dmd.dmd_loss_weight == 1
    assert config.dmd.rollout_solver == "renoise"
    assert config.dmd.student_update_frequency == 5
    assert config.dmd.cycle_rollout_steps
    config.pcm.phase_boundaries_q = [1, 0.5, 0.7, 0]
    with pytest.raises(ValueError, match="phase_boundaries_q"):
        config.validate()
    config.pcm.phase_boundaries_q = [1, 0]
    config.pcm.skipping_interval_steps = 25
    with pytest.raises(ValueError, match="skipping_interval_steps"):
        config.validate()

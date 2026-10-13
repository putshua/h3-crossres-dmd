import torch
import pytest
from torch import nn
from unittest.mock import patch

from h3dmd.config import H3Config
from h3dmd.data import H3Batch
from h3dmd.dmd import H3DMDMethod, H3Latents
from h3dmd.h3.adapter import H3Velocity
from h3dmd.models import H3ModelBundle


class TinyH3Role(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def freeze(self):
        self.requires_grad_(False)
        self.eval()
        return self

    def forward(self, video, audio, prompt, **kwargs):
        prompt_term = prompt.float().mean() * 0
        return H3Velocity(video * self.scale + prompt_term, audio * self.scale + prompt_term)


def make_batch() -> H3Batch:
    return H3Batch(
        prompt_embeds=torch.randn(1, 2, 8),
        text_token_tags=torch.ones(2, dtype=torch.long),
        video_latents=torch.randn(1, 24, 2, 4, 4),
        audio_latents=torch.randn(1, 2, 32, 3),
        sample_ids=["tiny"],
    )


def make_method() -> tuple[H3DMDMethod, H3ModelBundle]:
    config = H3Config()
    config.dmd.rollout_steps = 2
    config.dmd.rollout_grad_steps = 1
    models = H3ModelBundle(TinyH3Role(0.1), TinyH3Role(0.5), TinyH3Role(-0.25))
    return H3DMDMethod(config, models), models


def test_rf_loss_uses_separate_modal_weights_and_backpropagates() -> None:
    method, models = make_method()
    method.config.dmd.video_loss_weight = 2.0
    method.config.dmd.audio_loss_weight = 0.25
    batch = make_batch()
    torch.manual_seed(7)
    result = method.rf_loss(batch)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert result.metrics.keys() >= {"video", "audio", "q", "video_sigma", "audio_sigma"}
    assert models.student.scale.grad is not None
    assert torch.isfinite(models.student.scale.grad)


def test_dmd_one_step_runs_for_data_and_simulated_rollout() -> None:
    method, models = make_method()
    batch = make_batch()
    data_critic = method.critic_loss(batch, step=0)
    data_critic.loss.backward()
    assert models.critic.scale.grad is not None
    models.critic.scale.grad = None

    rollout_critic = method.critic_loss(batch, step=1)
    rollout_critic.loss.backward()
    assert rollout_critic.metrics["source_rollout"] == 1

    student = method.student_loss(batch)
    student.loss.backward()
    assert torch.isfinite(student.loss)
    assert models.student.scale.grad is not None
    assert all(parameter.grad is None for parameter in models.teacher.parameters())


def test_rollout_uses_shared_configurable_q_grid() -> None:
    method, _ = make_method()
    method.config.schedule.inference_q = [1.0, 0.4, 0.0]
    batch = make_batch()
    initial = H3Latents(torch.randn_like(batch.video_latents), torch.randn_like(batch.audio_latents))
    output = method.rollout(batch, requires_grad=True, initial=initial)
    assert output.video.shape == batch.video_latents.shape
    assert output.audio.shape == batch.audio_latents.shape
    assert output.video.grad_fn is not None


def test_renoise_rollout_prefix_and_last_forward_grad():
    method, models = make_method()
    method.config.dmd.rollout_solver = 'renoise'
    method.config.schedule.inference_q = [1, .4, 0]
    batch = make_batch()
    initial = H3Latents(torch.randn_like(batch.video_latents), torch.randn_like(batch.audio_latents))
    torch.manual_seed(42)
    output = method.rollout(batch, requires_grad=True, initial=initial)
    torch.manual_seed(42)
    first_clean = H3Latents(initial.video * 1.1, initial.audio * 1.1)
    mid = method.scheduler.level(.4)
    noisy, _ = method._noise(first_clean, mid)
    expected_video = noisy.video * (1 + mid.video_sigma * .1)
    expected_audio = noisy.audio * (1 + mid.audio_sigma * .1)
    torch.testing.assert_close(output.video, expected_video)
    torch.testing.assert_close(output.audio, expected_audio)
    output.video.mean().backward()
    torch.testing.assert_close(models.student.scale.grad, (noisy.video * mid.video_sigma).mean())
    one = method.rollout(batch, requires_grad=False, initial=initial, num_steps=1)
    torch.testing.assert_close(one.video, first_clean.video)


def test_fixed_backward_simulation_times_do_not_depend_on_noise_seed():
    method, _ = make_method()
    method.config.dmd.rollout_steps = 4
    method.config.dmd.rollout_solver = "renoise"
    method.config.schedule.inference_q = [1, .75, .5, .25, 0]
    batch = make_batch()
    original_call = method._call
    for count in range(1, 5):
        traces = []
        for seed in (7, 91):
            trace = []
            def record(role, state, data, layout, level):
                trace.append((float(level.q), float(level.video_timestep),
                              float(level.audio_timestep), torch.is_grad_enabled()))
                return original_call(role, state, data, layout, level)
            torch.manual_seed(seed)
            with patch.object(method, "_call", side_effect=record):
                result = method.rollout(batch, requires_grad=True, num_steps=count)
            assert result.video.requires_grad
            assert [r[0] for r in trace] == [1, .75, .5, .25][:count]
            assert [r[3] for r in trace] == [False] * (count - 1) + [True]
            traces.append(trace)
        assert traces[0] == traces[1]


def test_dmd_surrogate_matches_analytic_modal_gradients_and_isolates_roles():
    method, models = make_method()
    batch = make_batch()
    generated = H3Latents(torch.full_like(batch.video_latents, .3, requires_grad=True),
                          torch.full_like(batch.audio_latents, -.2, requires_grad=True))
    level = method.scheduler.level(.4)
    noise = H3Latents(torch.full_like(generated.video, .7), torch.full_like(generated.audio, -.8))
    noisy = H3Latents(method.scheduler.add_noise(generated.video, noise.video, level.video_sigma),
                     method.scheduler.add_noise(generated.audio, noise.audio, level.audio_sigma))
    method.config.dmd.video_loss_weight = 2
    method.config.dmd.audio_loss_weight = .25
    with patch.object(method, "rollout", return_value=generated), \
         patch.object(method, "_sample_level", return_value=level), \
         patch.object(method, "_noise", return_value=(noisy, noise)):
        result = method.dmd_loss(batch)
    result.loss.backward()
    expected_loss = 0
    for modality, weight in (("video", 2), ("audio", .25)):
        x = getattr(generated, modality)
        xt = getattr(noisy, modality).detach()
        sigma = getattr(level, modality + "_sigma")
        teacher = xt + sigma * xt * .5
        fake = xt + sigma * xt * -.25
        direction = (fake - teacher) / (x.detach() - teacher).abs().mean().clamp_min(1e-5)
        torch.testing.assert_close(x.grad, 2 * weight * direction / x.numel())
        expected_loss += weight * direction.square().mean()
    torch.testing.assert_close(result.loss, expected_loss)
    assert all(p.grad is None for p in models.parameters())


def test_pure_dmd_and_rollout_critic_ignore_shape_only_cache_values():
    method, _ = make_method()
    method.config.dmd.rf_loss_weight = 0
    method.config.dmd.data_mode = "rollout"
    batch = make_batch()
    for compute in (method.dmd_loss, method.critic_loss):
        batch.video_latents.zero_()
        batch.audio_latents.zero_()
        torch.manual_seed(18)
        before = compute(batch).loss
        batch.video_latents.fill_(999)
        batch.audio_latents.fill_(-999)
        torch.manual_seed(18)
        after = compute(batch).loss
        torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_score_video_shift_is_independent_and_audio_unchanged():
    config = H3Config()
    config.schedule.video_shift = 5
    config.schedule.train_video_shift = 6
    config.schedule.audio_shift = 3
    _, models = make_method()
    method = H3DMDMethod(config, models)
    with patch('torch.rand', return_value=torch.tensor(.5)):
        level = method._sample_level(torch.device('cpu'))
    assert float(level.video_sigma) == pytest.approx(6 / 7)
    assert float(level.audio_sigma) == pytest.approx(.75)
    grid = method.scheduler.schedule(q=[1, .75, .5, .25, 0])
    torch.testing.assert_close(grid.video_sigmas, torch.tensor([1, 15/16, 5/6, 5/8, 0]))
    torch.testing.assert_close(grid.audio_sigmas, torch.tensor([1, .9, .75, .5, 0]))


def test_old_schedule_keeps_native_score_shift():
    method, _ = make_method()
    assert method.config.schedule.train_video_shift is None
    assert method.training_scheduler.video_shift == 12
    assert method.training_scheduler.audio_shift == method.scheduler.audio_shift == 3


def test_joint_score_shift_two_does_not_change_audio_or_video_rollout():
    config = H3Config()
    config.schedule.video_shift = 6
    config.schedule.audio_shift = 3
    config.schedule.train_video_shift = config.schedule.train_audio_shift = 2
    config.schedule.train_q_min, config.schedule.train_q_max = 0, 1
    _, models = make_method()
    method = H3DMDMethod(config, models)
    with patch('torch.rand', return_value=torch.tensor(.5)):
        level = method._sample_level(torch.device('cpu'))
    torch.testing.assert_close(level.video_sigma, torch.tensor(2/3))
    torch.testing.assert_close(level.audio_sigma, torch.tensor(2/3))
    rollout_level = method.scheduler.level(.5)
    torch.testing.assert_close(rollout_level.video_sigma, torch.tensor(6/7))
    torch.testing.assert_close(rollout_level.audio_sigma, torch.tensor(.75))


@pytest.mark.parametrize('error', [0.0, 0.125])
def test_fake_fm_uses_generated_clean_fresh_noise_and_only_fake_gradients(error):
    method, models = make_method()
    method.config.dmd.data_mode = 'rollout'
    method.config.dmd.audio_loss_weight = .2
    method.config.dmd.critic_loss_weight = 1.
    batch = make_batch()
    clean = H3Latents(torch.full_like(batch.video_latents, .3, requires_grad=True),
                      torch.full_like(batch.audio_latents, -.2, requires_grad=True))
    level = method.training_scheduler.level(.4)
    torch.manual_seed(317)
    ev = torch.randn_like(clean.video, dtype=torch.float32)
    ea = torch.randn_like(clean.audio, dtype=torch.float32)

    class OracleFake(nn.Module):
        def __init__(self):
            super().__init__()
            self.error = nn.Parameter(torch.tensor(error))

        def forward(self, video, audio, prompt, **kwargs):
            for modality, x, noise in [('video', video, ev), ('audio', audio, ea)]:
                sigma = getattr(level, modality + '_sigma')
                target = getattr(clean, modality).detach()
                torch.testing.assert_close(x, (1-sigma)*target + sigma*noise)
                torch.testing.assert_close(kwargs[modality + '_timestep'], 1-sigma)
                assert not x.requires_grad
            return H3Velocity(clean.video.detach()-ev+self.error,
                              clean.audio.detach()-ea+self.error)

    models.critic = OracleFake()
    torch.manual_seed(317)
    with patch.object(method, 'rollout', return_value=clean) as rollout, \
         patch.object(method, '_sample_level', return_value=level):
        result = method.critic_loss(batch, step=1)
    assert rollout.call_args.kwargs['requires_grad'] is False
    torch.testing.assert_close(result.loss, torch.tensor(1.2 * error**2), atol=1e-7, rtol=1e-5)
    result.loss.backward()
    torch.testing.assert_close(models.critic.error.grad, torch.tensor(2*1.2*error), atol=1e-7, rtol=1e-5)
    assert clean.video.grad is None and clean.audio.grad is None
    assert all(p.grad is None for p in models.student.parameters())
    assert all(p.grad is None for p in models.teacher.parameters())


@pytest.mark.parametrize('score_shift', [2, 3])
def test_rcm_fullrange_score_sampler_preserves_rollout_grid(score_shift):
    config = H3Config()
    config.schedule.video_shift = 6
    config.schedule.train_video_shift = score_shift
    config.schedule.audio_shift = 3
    config.schedule.train_q_min = 0
    config.schedule.train_q_max = 1
    config.schedule.inference_q = [1, .75, .5, .25, 0]
    _, models = make_method()
    method = H3DMDMethod(config, models)
    # Extreme quantiles detect accidental old [.02,.98] base-q truncation.
    for u in [.001, .5, .999]:
        with patch('torch.rand', return_value=torch.tensor(u)):
            level = method._sample_level(torch.device('cpu'))
        assert float(level.q) == pytest.approx(u)
        assert float(level.video_sigma) == pytest.approx(score_shift*u/(1+(score_shift-1)*u))
        assert float(level.audio_sigma) == pytest.approx(3*u/(1+2*u))
    grid = method.scheduler.schedule(q=config.schedule.inference_q)
    torch.testing.assert_close(grid.video_sigmas, torch.tensor([1, 18/19, 6/7, 2/3, 0]))
    torch.testing.assert_close(grid.audio_sigmas, torch.tensor([1, .9, .75, .5, 0]))

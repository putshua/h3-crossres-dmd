"""Resolution transition, score domains, gradient truncation and preview parity."""
from unittest.mock import patch

import pytest
import torch

from h3dmd.inference import H3InferencePipeline
from h3dmd.methods.multires_dmd import H3MultiResolutionDMDMethod
from h3dmd.upsampling import new_mapper
from test_dmd import make_batch, make_method
from test_h3_crossres_dmd import settings


def method(kind="neural"):
    from h3dmd.upsampling import load_frozen_mapper
    c = settings()
    c.method = "h3_multires_dmd"
    c.crossres.rollout_high_resolution_from_step = 4
    c.schedule.video_shift, c.schedule.audio_shift = 12, 3
    c.schedule.train_video_shift, c.schedule.train_audio_shift = 6, 3
    c.dmd.cycle_rollout_steps = True
    c.dmd.update_schedule = "alternating"
    c.dmd.student_update_frequency = 6
    _, models = make_method()
    if kind == "bicubic":
        c.crossres.mapper_kind = "bicubic"
        mapper = load_frozen_mapper(c)
    else:
        mapper = new_mapper(c)
    return H3MultiResolutionDMDMethod(c, models, score_mapper=mapper), models


@pytest.mark.parametrize("exit_step", [1, 2, 3, 4])
@pytest.mark.parametrize("role", ["student", "critic"])
@pytest.mark.parametrize("kind", ["neural", "bicubic"])
def test_all_exit_score_domains_single_clean_mapping_and_gradients(exit_step, role, kind):
    m, models = method(kind)
    calls, maps = [], []
    original_call, original_map = m._call, m.score_mapper.forward

    def call(name, state, batch, layout, level):
        calls.append((name, tuple(state.video.shape[-2:]), torch.is_grad_enabled(),
                      layout.num_target_audio_rows, float(level.video_sigma), float(level.audio_sigma)))
        assert batch.video_latents.shape == state.video.shape
        return original_call(name, state, batch, layout, level)

    def mapping(clean):
        maps.append((tuple(clean.shape[-2:]), torch.is_grad_enabled()))
        return original_map(clean)

    with patch.object(m, "_call", side_effect=call), \
            patch.object(m.score_mapper, "forward", side_effect=mapping), \
            patch.object(m, "_rollout_count", return_value=exit_step):
        result = (m.dmd_loss(make_batch()) if role == "student"
                  else m.critic_loss(make_batch()))
    result.loss.backward()
    student = [c for c in calls if c[0] == "student"]
    scores = [c for c in calls if c[0] != "student"]
    assert [c[1] for c in student] == [(4, 4)] * min(exit_step, 3) + ([(6, 6)] if exit_step == 4 else [])
    assert all(c[1] == (6, 6) for c in scores)
    assert len(maps) == 1 and maps[0][0] == (4, 4)
    assert all(c[3] == student[0][3] for c in calls)
    assert [c[2] for c in student] == [False] * (exit_step - 1) + [role == "student"]
    assert maps[0][1] == (role == "student" and exit_step < 4)
    assert (models.student.scale.grad is not None) == (role == "student")
    assert (models.critic.scale.grad is not None) == (role == "critic")
    assert all(p.grad is None for p in m.score_mapper.parameters())
    assert all(p.grad is None for p in models.teacher.parameters())


@pytest.mark.parametrize("exit_step", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", ["neural", "bicubic"])
def test_preview_has_same_student_resolution_prefix_and_high_resolution_output(exit_step, kind):
    m, models = method(kind)
    pipeline = H3InferencePipeline(models.student, m.config, score_mapper=m.score_mapper)
    calls = []
    original = models.student.forward

    def forward(video, audio, prompt, **kwargs):
        calls.append((tuple(video.shape[-2:]), audio.shape, kwargs["layout"].num_target_audio_rows))
        return original(video, audio, prompt, **kwargs)

    with patch.object(models.student, "forward", side_effect=forward):
        result = pipeline.generate_latents(torch.randn(1, 2, 8), torch.ones(2, dtype=torch.long),
            height=64, width=64, num_frames=124, num_inference_steps=exit_step + 1,
            generator=torch.Generator().manual_seed(1))
    assert [c[0] for c in calls] == [(4, 4)] * min(exit_step, 3) + ([(6, 6)] if exit_step == 4 else [])
    assert result.video_latents.shape[-2:] == (6, 6)
    assert result.num_forwards == exit_step
    assert all(c[1:] == calls[0][1:] for c in calls)


def test_transition_is_in_clean_space_before_high_resolution_noise():
    m, _ = method()
    original_map, original_noise = m.score_mapper.forward, m._noise
    events = []

    def mapping(clean):
        mapped = original_map(clean)
        events.append(("map", mapped.detach().clone()))
        return mapped

    def noise(clean, level):
        events.append(("noise", clean.video.detach().clone()))
        return original_noise(clean, level)

    with patch.object(m.score_mapper, "forward", side_effect=mapping), \
            patch.object(m, "_noise", side_effect=noise):
        m.rollout(make_batch(), requires_grad=False, num_steps=4)
    assert [e[0] for e in events] == ["noise", "noise", "map", "noise"]
    torch.testing.assert_close(events[-2][1], events[-1][1])


def test_fixed_bicubic_matches_interpolation_and_exact_input_gradient():
    from h3dmd.upsampling import BicubicCleanMap, frames, video
    from torch.nn import functional as F
    x = torch.randn(1, 24, 3, 4, 4, requires_grad=True)
    mapper = BicubicCleanMap((6, 6))
    mapped = mapper(x)
    reference = video(F.interpolate(frames(x), size=(6, 6), mode="bicubic", align_corners=False), 1, 3)
    torch.testing.assert_close(mapped, reference)
    direction = torch.randn_like(mapped)
    actual = torch.autograd.grad(mapped, x, direction)[0]
    expected = torch.autograd.grad(reference, x, direction)[0]
    torch.testing.assert_close(actual, expected)
    assert not list(mapper.parameters())

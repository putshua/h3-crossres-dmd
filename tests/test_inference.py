import torch
import pytest
from torch import nn

from h3dmd.config import H3Config
from h3dmd.h3.adapter import H3Velocity
from h3dmd.h3.layout import unpack_audio
from h3dmd.inference import (
    H3InferencePipeline,
    align_num_frames,
    video_latent_num_frames,
)


class ZeroH3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, video, audio, prompt, **kwargs):
        self.calls += 1
        return H3Velocity(
            torch.zeros_like(video) + self.anchor, torch.zeros_like(audio) + self.anchor
        )


def test_geometry_contract() -> None:
    assert align_num_frames(120) == 124
    assert align_num_frames(360) == 362
    assert video_latent_num_frames(124) == 37
    assert video_latent_num_frames(362) == 107


def test_dense_count_and_sglang_independent_noise_streams() -> None:
    config = H3Config()
    model = ZeroH3()
    pipeline = H3InferencePipeline(model, config)
    seed = 123
    result = pipeline.generate_latents(
        torch.randn(1, 1, 8),
        torch.ones(1, dtype=torch.long),
        height=32,
        width=32,
        num_frames=124,
        generator=torch.Generator().manual_seed(seed),
    )
    assert result.num_forwards == 49
    assert model.calls == 49
    expected_video = torch.randn(
        (1, 24, 37, 2, 2), generator=torch.Generator().manual_seed(seed)
    )
    expected_audio_rows = torch.randn(
        (1, 2 * 207, 32), generator=torch.Generator().manual_seed(seed)
    )
    expected_audio = unpack_audio(expected_audio_rows, 2, 32)
    # The FP32 Euler blend performs two rounded multiplies per step even when
    # velocity is zero, so the 49-step state is only numerically identical.
    assert torch.allclose(result.video_latents, expected_video, atol=2e-6, rtol=2e-6)
    assert torch.allclose(result.audio_latents, expected_audio, atol=2e-6, rtol=2e-6)


def test_few_step_explicit_q_controls_forward_count() -> None:
    model = ZeroH3()
    pipeline = H3InferencePipeline(model, H3Config())
    result = pipeline.generate_latents(
        torch.randn(1, 1, 8),
        torch.ones(1, dtype=torch.long),
        height=32,
        width=32,
        num_frames=124,
        q_schedule=[1.0, 0.6, 0.2, 0.0],
    )
    assert result.num_forwards == 3
    assert model.calls == 3


@pytest.mark.parametrize('base_steps',[6,8])
def test_explicit_step_count_uses_training_prefix_and_shift(base_steps):
    config=H3Config()
    config.schedule.inference_q=[1-i/base_steps for i in range(base_steps+1)]
    config.dmd.rollout_steps=base_steps
    config.schedule.video_shift=6
    config.schedule.audio_shift=3
    config.dmd.rollout_solver='renoise'
    class RecordingH3(ZeroH3):
        def __init__(self):
            super().__init__();self.times=[]
        def forward(self,video,audio,prompt,**kwargs):
            self.times.append((float(kwargs['video_timestep']),float(kwargs['audio_timestep'])))
            return super().forward(video,audio,prompt,**kwargs)
    model=RecordingH3()
    pipe=H3InferencePipeline(model,config)
    result=pipe.generate_latents(torch.zeros(1,1,8),torch.ones(1,dtype=torch.long),
                                height=32,width=32,num_frames=124,num_inference_steps=5)
    assert result.num_forwards==4
    for (video_t,audio_t),q in zip(model.times,config.schedule.inference_q[:4]):
        assert video_t==pytest.approx(1-6*q/(1+5*q),abs=1e-6)
        assert audio_t==pytest.approx(1-3*q/(1+2*q),abs=1e-6)
    with pytest.raises(ValueError,match='explicit q_schedule'):
        pipe.generate_latents(torch.zeros(1,1,8),torch.ones(1,dtype=torch.long),
                              height=32,width=32,num_frames=124,num_inference_steps=base_steps+2)


def test_renoise_inference_uses_seeded_fresh_noise():
    config = H3Config()
    config.dmd.rollout_solver = "renoise"
    pipeline = H3InferencePipeline(ZeroH3(), config)
    kwargs = dict(height=32, width=32, num_frames=124, q_schedule=[1.0, 0.5, 0.0])
    prompt, tags = torch.zeros(1, 1, 8), torch.ones(1, dtype=torch.long)
    output = pipeline.generate_latents(
        prompt, tags, generator=torch.Generator().manual_seed(23), **kwargs
    )
    rng = torch.Generator().manual_seed(23)
    initial = torch.randn(output.video_latents.shape, generator=rng)
    fresh = torch.randn(output.video_latents.shape, generator=rng)
    sigma = pipeline.scheduler.level(0.5).video_sigma
    torch.testing.assert_close(
        output.video_latents, (1 - sigma) * initial + sigma * fresh
    )
    again = pipeline.generate_latents(
        prompt, tags, generator=torch.Generator().manual_seed(23), **kwargs
    )
    torch.testing.assert_close(output.audio_latents, again.audio_latents)


def test_pcm_only_inference_uses_phase_edges_and_euler():
    config = H3Config()
    config.pcm.enabled = True
    config.pcm.phase_boundaries_q = [1.0, 0.4, 0.0]
    config.dmd.dmd_loss_weight = 0
    config.dmd.rollout_solver = "renoise"  # PCM-only selects Euler automatically.
    pipeline = H3InferencePipeline(ZeroH3(), config)
    output = pipeline.generate_latents(
        torch.zeros(1, 1, 8),
        torch.ones(1, dtype=torch.long),
        height=32,
        width=32,
        num_frames=124,
        generator=torch.Generator().manual_seed(23),
    )
    assert output.num_forwards == 2
    expected = torch.randn(
        output.video_latents.shape, generator=torch.Generator().manual_seed(23)
    )
    torch.testing.assert_close(output.video_latents, expected)


def test_clean_boundary_resizes_x0_before_fresh_noise_and_preserves_audio():
    from torch.nn import functional as F
    config = H3Config()
    config.dmd.rollout_solver = 'renoise'
    config.schedule.video_shift = 12

    class RecordingH3(ZeroH3):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, video, audio, prompt, **kwargs):
            self.inputs.append((video.clone(), audio.clone(), kwargs['layout']))
            return H3Velocity(torch.ones_like(video), torch.zeros_like(audio))

    class ResizeClean(nn.Module):
        def forward(self, value):
            self.received = value.clone()
            b, c, t, h, w = value.shape
            flat = value.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
            return F.interpolate(flat, size=(4, 4), mode='bicubic', align_corners=False).reshape(
                b, t, c, 4, 4).permute(0, 2, 1, 3, 4).contiguous()

    model = RecordingH3()
    pipe = H3InferencePipeline(model, config)
    resize = ResizeClean()
    kwargs = dict(height=32, width=32, num_frames=124, q_schedule=[1., .25, 0.],
                  solver='renoise', generator=torch.Generator().manual_seed(1))
    prompt, tags = torch.zeros(1, 1, 8), torch.ones(1, dtype=torch.long)
    result = pipe.generate_latents(prompt, tags, clean_video_transforms={1:resize}, **kwargs)
    first, second = model.inputs
    torch.testing.assert_close(resize.received, first[0]+1.)
    rng = torch.Generator().manual_seed(1)
    torch.randn(first[0].shape, generator=rng)
    fresh = torch.randn(second[0].shape, generator=rng)
    clean = resize(resize.received)
    sigma = pipe.scheduler.level(.25).video_sigma
    torch.testing.assert_close(second[0], (1-sigma)*clean+sigma*fresh)
    assert second[0].shape[-2:] == (4,4)
    assert result.num_forwards == 2
    baseline = H3InferencePipeline(ZeroH3(), config).generate_latents(
        prompt, tags, **dict(kwargs, generator=torch.Generator().manual_seed(1)))
    torch.testing.assert_close(result.audio_latents, baseline.audio_latents)
    with pytest.raises(ValueError, match='interior target grid'):
        pipe.generate_latents(prompt,tags,clean_video_transforms={2:resize},**kwargs)
    with pytest.raises(ValueError, match='require renoise'):
        pipe.generate_latents(prompt,tags,clean_video_transforms={1:resize},
                              **dict(kwargs,solver='euler'))

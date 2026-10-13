from types import SimpleNamespace

import pytest
import torch
from torch import nn

from h3dmd.h3.adapter import H3ModelAdapter
from h3dmd.h3.layout import build_packed_layout


class EchoPackedTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.config = SimpleNamespace(in_channels=24, audio_in_channels=32, patch_size=(1, 2, 2))
        self.seen = None

    def forward(self, hidden_states, audio_hidden_states, **kwargs):
        self.seen = kwargs
        return hidden_states * self.scale, audio_hidden_states * self.scale


def test_adapter_preserves_video_audio_native_shapes() -> None:
    transformer = EchoPackedTransformer()
    model = H3ModelAdapter(transformer)
    layout = build_packed_layout(
        text_token_tags=torch.ones(3, dtype=torch.long),
        num_latent_frames=2,
        latent_height=4,
        latent_width=6,
        num_audio_latents=5,
    )
    video = torch.randn(1, 24, 2, 4, 6)
    audio = torch.randn(1, 2, 32, 5)
    text = torch.randn(1, 3, 5120)
    output = model(
        video,
        audio,
        text,
        layout=layout,
        video_timestep=0.1,
        audio_timestep=0.4,
    )
    assert torch.equal(output.video, 2 * video)
    assert torch.equal(output.audio, 2 * audio)
    assert transformer.seen["timestep"].dtype == torch.float32
    assert transformer.seen["timestep_indices"].shape == (layout.sequence_length,)


def test_adapter_rejects_padding_when_transformer_has_no_mask_input() -> None:
    model = H3ModelAdapter(EchoPackedTransformer())
    layout = build_packed_layout(
        text_token_tags=torch.ones(3, dtype=torch.long),
        num_latent_frames=2,
        latent_height=4,
        latent_width=6,
        num_audio_latents=5,
        alignment=64,
    )
    with pytest.raises(NotImplementedError, match="padded attention"):
        model(
            torch.randn(1, 24, 2, 4, 6),
            torch.randn(1, 2, 32, 5),
            torch.randn(1, 3, 5120),
            layout=layout,
            video_timestep=0.1,
            audio_timestep=0.4,
        )


def test_outer_autocast_preserves_native_fp32_projection_computation() -> None:
    class ProjectedTransformer(EchoPackedTransformer):
        def __init__(self):
            super().__init__()
            self.proj_in = nn.Linear(96, 96)
            self.audio_proj_in = nn.Linear(32, 32)

        def forward(self, hidden_states, audio_hidden_states, **kwargs):
            return self.proj_in(hidden_states.float()), self.audio_proj_in(audio_hidden_states.float())

    model = H3ModelAdapter(ProjectedTransformer())
    layout = build_packed_layout(text_token_tags=torch.ones(3, dtype=torch.long),
        num_latent_frames=2, latent_height=4, latent_width=6, num_audio_latents=5)
    video = torch.randn(1, 24, 2, 4, 6)
    audio = torch.randn(1, 2, 32, 5)
    text = torch.randn(1, 3, 5120)
    kwargs = dict(layout=layout, video_timestep=0.1, audio_timestep=0.4)
    reference = model(video, audio, text, **kwargs)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with torch.no_grad():
            model(video, audio, text, **kwargs)
        actual = model(video, audio, text, **kwargs)
    assert actual.video.dtype == actual.audio.dtype == torch.float32
    torch.testing.assert_close(actual.video, reference.video, rtol=0, atol=0)
    torch.testing.assert_close(actual.audio, reference.audio, rtol=0, atol=0)
    (actual.video.square().mean() + actual.audio.square().mean()).backward()
    for parameter in (model.transformer.proj_in.weight, model.transformer.audio_proj_in.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

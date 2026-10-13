"""Numerical contracts needed before enabling cross-resolution H3 DMD."""
import pytest
import torch
from torch.nn import functional as F

from h3dmd.upsampling import NeuralCleanMap, frames, video


@pytest.mark.parametrize("kind", ["dysample", "carafe", "liif", "rcan"])
def test_h3_frozen_mapper_preserves_bicubic_input_jacobian_at_initialization(kind):
    torch.set_num_threads(2)
    mapper = NeuralCleanMap((6, 8), kind, channels=24, width=16).eval().requires_grad_(False)
    x = torch.randn(1, 24, 2, 4, 4, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    expected = video(F.interpolate(frames(reference_x), size=(6, 8), mode="bicubic",
                                   align_corners=False), 1, 2)
    actual = mapper(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    direction = torch.randn_like(actual)
    (actual * direction).mean().backward()
    (expected * direction).mean().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    assert x.grad.abs().sum() > 0
    assert all(p.grad is None for p in mapper.parameters())


@pytest.mark.parametrize("kind", ["dysample", "carafe", "liif", "rcan"])
def test_h3_learned_mapper_does_not_mix_temporal_positions(kind):
    torch.set_num_threads(2)
    mapper = NeuralCleanMap((6, 8), kind, channels=24, width=16)
    # Exercise an active residual, rather than only the bicubic skip.
    with torch.no_grad():
        output_layer = mapper.decoder if hasattr(mapper.decoder, "weight") else mapper.decoder[-1]
        output_layer.weight.normal_(std=.01)
    x = torch.randn(1, 24, 2, 4, 4)
    baseline = mapper(x)
    altered = x.clone()
    altered[:, :, 1] += 10
    result = mapper(altered)
    torch.testing.assert_close(baseline[:, :, 0], result[:, :, 0], rtol=0, atol=0)
    assert not torch.equal(baseline[:, :, 1], result[:, :, 1])

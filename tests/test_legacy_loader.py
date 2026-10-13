import pytest
import torch

from h3dmd.h3.legacy_loader import (
    legacy_config_to_diffusers, legacy_key_targets, legacy_tensor_values, split_grouped_qkv,
)


@pytest.mark.parametrize("name", ["blocks.0.mlp.fc1.weight", "token_refiner.blocks.1.mlp.fc1.weight"])
def test_converted_swiglu_matches_native_forward_and_input_gradient(name):
    from diffusers.models.activations import SwiGLU

    torch.manual_seed(7)
    weight = torch.randn(16, 6)
    x = torch.randn(3, 6, requires_grad=True)
    gate, up = torch.nn.functional.linear(x, weight).chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate) * up
    module = SwiGLU(6, 8, bias=False)
    with torch.no_grad():
        module.proj.weight.copy_(legacy_tensor_values(name, weight, 1, 1)[0])
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad_expected = torch.autograd.grad(expected.square().sum(), x, retain_graph=True)[0]
    grad_actual = torch.autograd.grad(actual.square().sum(), x)[0]
    torch.testing.assert_close(grad_actual, grad_expected)
    with torch.no_grad():
        module.proj.weight.copy_(weight)
        assert not torch.allclose(module(x), expected)


def test_legacy_names_map_to_exact_diffusers_modules() -> None:
    assert legacy_key_targets("blocks.0.attn.q_norm.weight") == (
        "transformer_blocks.0.attn.norm_q.weight",
    )
    assert legacy_key_targets("token_refiner.blocks.1.attn.k_norm.weight") == (
        "token_refiner.refiner_blocks.1.attn.norm_k.weight",
    )
    assert legacy_key_targets("time_embedder.proj_in.weight") == ("time_embedder.linear_1.weight",)
    assert legacy_key_targets("final_layer.video_out.bias") == ("proj_out.bias",)
    assert legacy_key_targets("token_refiner.final_norm.weight") == ("token_refiner.final_norm.weight",)
    assert legacy_key_targets("blocks.3.attn.qkv_proj.weight") == (
        "transformer_blocks.3.attn.to_q.weight",
        "transformer_blocks.3.attn.to_k.weight",
        "transformer_blocks.3.attn.to_v.weight",
    )


def test_grouped_head_qkv_split_is_not_flat_thirds() -> None:
    weight = torch.arange(12).reshape(12, 1)
    q, k, v = split_grouped_qkv(weight, num_heads=2, head_dim=2)
    assert q.flatten().tolist() == [0, 1, 6, 7]
    assert k.flatten().tolist() == [2, 3, 8, 9]
    assert v.flatten().tolist() == [4, 5, 10, 11]


def test_legacy_config_translation() -> None:
    converted = legacy_config_to_diffusers(
        {
            "token_refiner_num_layers": 2,
            "ffn_hidden_size": 14336,
            "latents_dim": 24,
            "audio_latents_dim": 32,
            "patch_size": [1, 2, 2],
        }
    )
    assert converted == {
        "num_refiner_layers": 2,
        "ffn_dim": 14336,
        "in_channels": 24,
        "audio_in_channels": 32,
        "patch_size": (1, 2, 2),
    }

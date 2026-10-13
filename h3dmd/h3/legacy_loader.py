from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch


def resolve_transformer_dir(pretrained_path: str | Path, subfolder: str | None = "transformer") -> Path:
    root = Path(pretrained_path).expanduser()
    candidate = root / subfolder if subfolder and (root / subfolder).is_dir() else root
    if not (candidate / "config.json").is_file():
        raise FileNotFoundError(f"no H3 transformer config under {candidate}")
    return candidate


def legacy_config_to_diffusers(config: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "num_attention_heads": "num_attention_heads",
        "attention_head_dim": "attention_head_dim",
        "hidden_size": "hidden_size",
        "num_layers": "num_layers",
        "token_refiner_num_layers": "num_refiner_layers",
        "ffn_hidden_size": "ffn_dim",
        "latents_dim": "in_channels",
        "audio_latents_dim": "audio_in_channels",
        "patch_size": "patch_size",
        "text_dim": "text_dim",
        "timestep_input_dim": "freq_dim",
        "time_embed_hidden_size": "time_embed_hidden_dim",
        "time_embed_dim": "time_embed_dim",
        "rope_inv_freq_len": "rope_freq_dim",
        "norm_eps": "norm_eps",
        "qk_norm_eps": "qk_norm_eps",
        "final_norm_eps": "final_norm_eps",
    }
    result = {target: config[source] for source, target in mapping.items() if source in config}
    if "patch_size" in result:
        result["patch_size"] = tuple(result["patch_size"])
    return result


def _rename_legacy_key(name: str) -> str:
    # Handle namespaces before component-level renames. A global ``blocks.``
    # replacement would also corrupt the ``refiner_blocks`` substring.
    if name.startswith("token_refiner.blocks."):
        name = "token_refiner.refiner_blocks." + name[len("token_refiner.blocks.") :]
    elif name.startswith("blocks."):
        name = "transformer_blocks." + name[len("blocks.") :]
    replacements = (
        ("video_patch_proj.", "proj_in."),
        ("audio_patch_proj.", "audio_proj_in."),
        ("condition_proj.", "context_embedder."),
        ("time_embedder.proj_in.", "time_embedder.linear_1."),
        ("time_embedder.proj_out.", "time_embedder.linear_2."),
        (".attn.q_norm.", ".attn.norm_q."),
        (".attn.k_norm.", ".attn.norm_k."),
        (".attn.out_proj.", ".attn.to_out.0."),
        (".mlp.fc1.", ".ff.net.0.proj."),
        (".mlp.fc2.", ".ff.net.2."),
        ("final_layer.adaln_proj.linear.", "norm_out.linear."),
        ("final_layer.norm.", "norm_out.norm."),
        ("final_layer.video_out.", "proj_out."),
        ("final_layer.audio_out.", "audio_proj_out."),
    )
    value = name
    for source, target in replacements:
        value = value.replace(source, target)
    return value


def legacy_key_targets(name: str) -> tuple[str, ...]:
    if name == "rope.inv_freq":
        return ()
    renamed = _rename_legacy_key(name)
    if renamed.endswith(".attn.qkv_proj.weight"):
        prefix = renamed[: -len("qkv_proj.weight")]
        return (prefix + "to_q.weight", prefix + "to_k.weight", prefix + "to_v.weight")
    return (renamed,)


def split_grouped_qkv(weight: torch.Tensor, num_heads: int, head_dim: int) -> tuple[torch.Tensor, ...]:
    expected = num_heads * 3 * head_dim
    if weight.shape[0] != expected:
        raise ValueError(f"legacy QKV has {weight.shape[0]} rows, expected {expected}")
    grouped = weight.reshape(num_heads, 3, head_dim, *weight.shape[1:])
    return tuple(grouped[:, index].reshape(num_heads * head_dim, *weight.shape[1:]).contiguous() for index in range(3))


def legacy_tensor_values(
    name: str, value: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, ...]:
    """Convert checkpoint storage conventions, not just parameter names."""
    if name.endswith(".attn.qkv_proj.weight"):
        return split_grouped_qkv(value, num_heads, head_dim)
    if name.endswith(".mlp.fc1.weight"):
        # Native H3: silu(gate) * up, with checkpoint rows [gate, up].
        # Diffusers SwiGLU: up * silu(gate), with rows [up, gate].
        # This applies to both the main blocks and the text token refiner.
        if value.shape[0] % 2:
            raise ValueError(f"legacy SwiGLU projection {name} needs an even row count")
        gate, up = value.chunk(2, dim=0)
        return (torch.cat((up, gate), dim=0),)
    return (value,)


def legacy_checkpoint_keys(transformer_dir: str | Path) -> set[str]:
    directory = Path(transformer_dir)
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        return set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"])
    from safetensors import safe_open

    keys: set[str] = set()
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


def _iter_tensors(directory: Path) -> Iterator[tuple[str, torch.Tensor]]:
    from safetensors import safe_open

    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = sorted(path.name for path in directory.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors weights under {directory}")
    for shard_name in shards:
        with safe_open(directory / shard_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


def load_legacy_fl2va_transformer_strict(
    pretrained_path: str | Path,
    *,
    subfolder: str | None = "transformer",
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Stream the original FL2VA key layout into Diffusers' H3 class.

    This is a strict conversion: every source key must be consumed, every
    Diffusers parameter must be populated, and H3's original mixed dtypes are
    retained. It does not materialize an intermediate 61.7 GB state dict.
    """

    try:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from diffusers import MiniMaxH3Transformer3DModel
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "Legacy FL2VA loading needs a Diffusers build containing "
            "MiniMaxH3Transformer3DModel plus accelerate."
        ) from exc

    directory = resolve_transformer_dir(pretrained_path, subfolder)
    raw_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model_config = legacy_config_to_diffusers(raw_config)
    num_heads = int(model_config.get("num_attention_heads", 56))
    head_dim = int(model_config.get("attention_head_dim", 128))
    with init_empty_weights(include_buffers=True):
        model = MiniMaxH3Transformer3DModel(**model_config)
    expected = set(dict(model.named_parameters()))
    loaded: set[str] = set()
    source_seen: set[str] = set()
    rope_value: torch.Tensor | None = None

    for source_name, value in _iter_tensors(directory):
        if source_name in source_seen:
            raise RuntimeError(f"duplicate source tensor {source_name}")
        source_seen.add(source_name)
        if source_name == "rope.inv_freq":
            rope_value = value.float()
            continue
        targets = legacy_key_targets(source_name)
        values = legacy_tensor_values(source_name, value, num_heads, head_dim)
        for target, tensor in zip(targets, values):
            if target not in expected:
                raise RuntimeError(f"legacy tensor {source_name} maps to unexpected Diffusers key {target}")
            if target in loaded:
                raise RuntimeError(f"multiple legacy tensors map to {target}")
            # Accelerate otherwise casts the checkpoint value to the meta
            # parameter's construction dtype (FP32). H3's source safetensors
            # intentionally mix BF16 blocks with selected FP32 projections,
            # so the source dtype must be explicit here.
            set_module_tensor_to_device(model, target, device, value=tensor, dtype=tensor.dtype)
            loaded.add(target)

    missing = sorted(expected - loaded)
    if missing:
        raise RuntimeError(f"strict legacy FL2VA load left {len(missing)} parameters missing: {missing[:20]}")
    if rope_value is None:
        raise RuntimeError("strict legacy FL2VA load requires rope.inv_freq")
    set_module_tensor_to_device(
        model,
        "rope.inv_freq",
        device,
        value=rope_value,
        dtype=torch.float32,
    )
    model.register_to_config(_name_or_path=str(pretrained_path))
    return model


__all__ = [
    "legacy_checkpoint_keys",
    "legacy_config_to_diffusers",
    "legacy_key_targets",
    "load_legacy_fl2va_transformer_strict",
    "resolve_transformer_dir",
    "split_grouped_qkv",
]

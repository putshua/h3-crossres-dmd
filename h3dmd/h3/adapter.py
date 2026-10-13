from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from h3dmd.registry import MODELS

from .layout import H3PackedLayout, pack_audio, patchify_video, unpack_audio, unpatchify_video
from .legacy_loader import load_legacy_fl2va_transformer_strict


FP32_MODULE_PATTERNS = (
    "proj_in",
    "audio_proj_in",
    "time_embedder",
    "proj_out",
    "audio_proj_out",
    "rope",
)


@dataclass(frozen=True)
class H3Velocity:
    video: torch.Tensor
    audio: torch.Tensor


@MODELS.register("minimax_h3_diffusers")
class H3ModelAdapter(nn.Module):
    """Training-friendly adapter around Diffusers' packed H3 transformer."""

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str | Path,
        *,
        subfolder: str = "transformer",
        loader: str = "diffusers",
        dtype: torch.dtype = torch.bfloat16,
        strict: bool = True,
        device: str | torch.device | None = None,
    ) -> "H3ModelAdapter":
        if dtype is not torch.bfloat16:
            raise ValueError("H3 must load BF16 while retaining its selected FP32 modules")
        if loader == "legacy_fl2va":
            transformer = load_legacy_fl2va_transformer_strict(
                pretrained_path, subfolder=subfolder, device=device or "cpu"
            )
        elif loader == "diffusers":
            try:
                from diffusers import MiniMaxH3Transformer3DModel
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    "Install a Diffusers build with MiniMaxH3Transformer3DModel "
                    "(currently main/0.40.0.dev0 or newer)."
                ) from exc
            kwargs: dict[str, Any] = {
                "subfolder": subfolder,
                "dtype": dtype,
                "low_cpu_mem_usage": True,
            }
            if strict:
                kwargs["output_loading_info"] = True
            loaded = MiniMaxH3Transformer3DModel.from_pretrained(str(pretrained_path), **kwargs)
            if strict:
                transformer, info = loaded
                failures = {
                    key: info.get(key, [])
                    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                    if info.get(key)
                }
                if failures:
                    raise RuntimeError(f"Diffusers H3 strict load failed: {failures}")
            else:
                transformer = loaded
            if device is not None:
                transformer.to(device)
        else:
            raise ValueError(f"unknown H3 loader {loader!r}")
        adapter = cls(transformer)
        adapter.assert_mixed_precision()
        return adapter

    @property
    def config(self) -> Any:
        return self.transformer.config

    @property
    def video_channels(self) -> int:
        return int(getattr(self.config, "in_channels", 24))

    @property
    def audio_latent_channels(self) -> int:
        return int(getattr(self.config, "audio_in_channels", 32))

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return tuple(getattr(self.config, "patch_size", (1, 2, 2)))

    def assert_mixed_precision(self) -> None:
        wrong: list[str] = []
        for name, parameter in self.transformer.named_parameters():
            keep_fp32 = any(pattern in name for pattern in FP32_MODULE_PATTERNS)
            expected = torch.float32 if keep_fp32 else torch.bfloat16
            if parameter.dtype != expected:
                wrong.append(f"{name}: {parameter.dtype} != {expected}")
            if len(wrong) == 20:
                break
        for name, buffer in self.transformer.named_buffers():
            if "rope" in name and torch.is_floating_point(buffer) and buffer.dtype != torch.float32:
                wrong.append(f"{name}: {buffer.dtype} != {torch.float32}")
        if wrong:
            raise RuntimeError("H3 checkpoint mixed-precision contract violated:\n" + "\n".join(wrong))

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()
        elif hasattr(self.transformer, "gradient_checkpointing"):
            self.transformer.gradient_checkpointing = True
        else:
            raise NotImplementedError("wrapped transformer has no gradient-checkpointing API")

    def configure_lora(
        self,
        *,
        path: str | None = None,
        rank: int = 0,
        alpha: int = 0,
        target_modules: list[str] | None = None,
        trainable: bool = True,
    ) -> None:
        if path or rank > 0:
            self.transformer.requires_grad_(False)
        if path:
            if not hasattr(self.transformer, "load_lora_adapter"):
                raise NotImplementedError("this Diffusers H3 build has no LoRA adapter loader")
            self.transformer.load_lora_adapter(path)
        elif rank > 0:
            try:
                from peft import LoraConfig
            except ImportError as exc:
                raise ImportError("creating an H3 LoRA requires peft") from exc
            config = LoraConfig(
                r=rank,
                lora_alpha=alpha or rank,
                target_modules=target_modules or ["to_q", "to_k", "to_v", "to_out.0"],
                init_lora_weights=True,
            )
            if not hasattr(self.transformer, "add_adapter"):
                raise NotImplementedError("this Diffusers H3 build has no PEFT add_adapter API")
            self.transformer.add_adapter(config)
        else:
            return
        lora_parameters = 0
        for name, parameter in self.transformer.named_parameters():
            is_lora = "lora_" in name
            parameter.requires_grad_(trainable and is_lora)
            lora_parameters += parameter.numel() if is_lora else 0
        if lora_parameters == 0:
            raise RuntimeError("LoRA was requested but no lora_* parameters were installed")

    def freeze(self) -> "H3ModelAdapter":
        self.requires_grad_(False)
        self.eval()
        return self

    def forward(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        layout: H3PackedLayout,
        video_timestep: float | torch.Tensor,
        audio_timestep: float | torch.Tensor,
        condition_video_latents: list[torch.Tensor] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> H3Velocity:
        if video_latents.ndim != 5 or audio_latents.ndim != 4 or prompt_embeds.ndim != 3:
            raise ValueError("expected video [B,C,T,H,W], audio [B,2,C,T], text [B,L,D]")
        if not (video_latents.shape[0] == audio_latents.shape[0] == prompt_embeds.shape[0]):
            raise ValueError("video/audio/text batch sizes must match")
        video_rows = patchify_video(video_latents, layout.patch_size)
        if condition_video_latents:
            condition_rows = [patchify_video(value, layout.patch_size) for value in condition_video_latents]
            video_rows = torch.cat([*condition_rows, video_rows], dim=1)
        elif layout.num_condition_video_rows:
            raise ValueError("layout contains video conditions but no condition latents were supplied")
        audio_rows = pack_audio(audio_latents)
        if video_rows.shape[1] != layout.video_indices.numel():
            raise ValueError("packed video rows do not match layout.video_indices")
        if audio_rows.shape[1] != layout.audio_indices.numel():
            raise ValueError("packed audio rows do not match layout.audio_indices")
        if prompt_embeds.shape[1] != layout.text_indices.numel():
            raise ValueError("prompt rows do not match layout.text_indices")

        unique_t, timestep_indices = layout.build_row_timesteps(video_timestep, audio_timestep)
        kwargs = layout.transformer_kwargs()
        if layout.attention_mask is not None:
            signature = inspect.signature(self.transformer.forward)
            if "attention_mask" not in signature.parameters:
                raise NotImplementedError(
                    "this Diffusers H3 transformer does not expose padded attention; use a padless layout"
                )
            kwargs["attention_mask"] = layout.attention_mask
        # Native H3 explicitly casts each activation to its module's stored
        # dtype: FP32 input/time/output projections, BF16 attention/MLP blocks.
        # An outer trainer/inference autocast overrides those explicit casts
        # and silently runs the FP32 projections in BF16. Let H3 own its
        # precision, also during gradient-checkpoint recomputation.
        with torch.autocast(video_rows.device.type, enabled=False):
            result = self.transformer(
                hidden_states=video_rows,
                audio_hidden_states=audio_rows,
                encoder_hidden_states=prompt_embeds,
                timestep=unique_t,
                timestep_indices=timestep_indices,
                attention_kwargs=attention_kwargs,
                return_dict=False,
                **kwargs,
            )
        video_output, audio_output = result
        video_output = video_output[:, layout.target_video_slice]
        audio_output = audio_output[:, layout.target_audio_slice]
        return H3Velocity(
            video=unpatchify_video(video_output, layout, self.video_channels),
            audio=unpack_audio(audio_output, layout.audio_channels, self.audio_latent_channels),
        )


__all__ = ["FP32_MODULE_PATTERNS", "H3ModelAdapter", "H3Velocity"]

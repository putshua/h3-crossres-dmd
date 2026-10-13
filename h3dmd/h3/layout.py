from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import torch

VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2
PADDING_TAG = -1
ROPE_FRAME_RESCALE = 5.0 / 3.0
ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
ROPE_SPATIAL_SCALE = 32


@dataclass(frozen=True)
class H3PackedLayout:
    """Structural tensors for H3's one packed self-attention document.

    Row order is ``[text | condition video | target audio | target video | pad]``.
    Phase 1 uses no condition or padding rows, while the fields remain explicit
    so FL2VA/Ref2VA can extend the builder without changing the adapter contract.
    """

    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    attention_mask: torch.Tensor | None
    cu_seqlens: torch.Tensor
    num_text_rows: int
    num_condition_video_rows: int
    num_condition_audio_rows: int
    num_target_video_rows: int
    num_target_audio_rows: int
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int
    patch_size: tuple[int, int, int]
    audio_channels: int
    sequence_length: int
    used_sequence_length: int

    def to(self, device: torch.device | str) -> "H3PackedLayout":
        updates: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, torch.Tensor):
                updates[item.name] = value.to(device)
        return replace(self, **updates)

    def transformer_kwargs(self) -> dict[str, torch.Tensor]:
        kwargs = {
            "position_ids": self.position_ids,
            "token_tags": self.token_tags,
            "video_indices": self.video_indices,
            "audio_indices": self.audio_indices,
            "text_indices": self.text_indices,
        }
        return kwargs

    def sglang_kwargs(self) -> dict[str, torch.Tensor | int]:
        """Expose the equivalent SGLang structural naming for oracle checks."""

        return {
            "img_position_ids": self.position_ids,
            "token_tags": self.token_tags,
            "img_pos": self.video_indices,
            "audio_pos": self.audio_indices,
            "text_pos": self.text_indices,
            "cu_seqlens": self.cu_seqlens,
            "seq_len": self.sequence_length,
        }

    def build_row_timesteps(
        self,
        video_timestep: float | torch.Tensor,
        audio_timestep: float | torch.Tensor,
        condition_video_timestep: float = 0.999,
        condition_audio_timestep: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return H3's sorted unique timesteps and per-packed-row indices."""

        device = self.position_ids.device
        video_t = float(torch.as_tensor(video_timestep).detach().cpu())
        audio_t = float(torch.as_tensor(audio_timestep).detach().cpu())
        row_t = torch.full((self.sequence_length,), video_t, dtype=torch.float32, device=device)
        if self.num_condition_video_rows:
            row_t[self.video_indices[: self.num_condition_video_rows]] = max(
                video_t, condition_video_timestep
            )
        target_audio = self.audio_indices[self.num_condition_audio_rows :]
        row_t[target_audio] = audio_t
        if self.num_condition_audio_rows:
            row_t[self.audio_indices[: self.num_condition_audio_rows]] = condition_audio_timestep
        # Padding is its own attention document and never reaches an output
        # head. Giving it video_t keeps AdaLN indices valid.
        return torch.unique(row_t, sorted=True, return_inverse=True)

    @property
    def target_video_slice(self) -> slice:
        return slice(self.num_condition_video_rows, self.video_indices.numel())

    @property
    def target_audio_slice(self) -> slice:
        return slice(self.num_condition_audio_rows, self.audio_indices.numel())


def _spatial_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # np.linspace(endpoint=False) is intentional: it matches the released
    # implementation's fp64 arithmetic, unlike torch.linspace.
    values = np.linspace(left, left + ratio, dim // patch, endpoint=False)
    return torch.from_numpy(values * ROPE_SPATIAL_SCALE).to(torch.float64)


def _temporal_grid(count: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [
            ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[index % len(ROPE_FRAMES_PER_LATENT)]
            for index in range(count)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _last_anchor_time(count: int, origin: float) -> float:
    spans = np.ones(count, dtype=np.float64) * ROPE_FRAME_RESCALE
    for offset in range(len(ROPE_FRAMES_PER_LATENT)):
        spans[offset :: len(ROPE_FRAMES_PER_LATENT)] *= ROPE_FRAMES_PER_LATENT[offset]
    return origin + float(spans.sum()) - ROPE_FRAME_RESCALE


def build_packed_layout(
    *,
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    audio_channels: int = 2,
    keyframe_anchors: tuple[str, ...] = (),
    condition_audio_rows: int = 0,
    alignment: int | None = None,
) -> H3PackedLayout:
    """Build the Diffusers/SGLang-compatible H3 FL2VA/T2VA row layout."""

    # Layout construction intentionally uses NumPy fp64 grids and CPU index
    # tensors for exact reference arithmetic. Callers move the completed,
    # internally consistent structure with ``layout.to(device)``.
    text_token_tags = text_token_tags.detach().cpu()
    if text_token_tags.ndim != 1 or text_token_tags.numel() == 0:
        raise ValueError("text_token_tags must be a non-empty 1-D tensor")
    if not torch.all((text_token_tags == TEXT_TAG) | (text_token_tags == VIDEO_TAG)):
        raise ValueError("text presentation rows may only be tagged text(1) or video(0)")
    if min(num_latent_frames, latent_height, latent_width, num_audio_latents) <= 0:
        raise ValueError("latent dimensions must be positive")
    patch_t, patch_h, patch_w = patch_size
    if num_latent_frames % patch_t or latent_height % patch_h or latent_width % patch_w:
        raise ValueError("video latent geometry must be divisible by patch_size")
    if audio_channels != 2:
        raise ValueError("the released H3 checkpoint packs exactly two audio channels")
    if condition_audio_rows:
        raise NotImplementedError("audio conditioning rows are reserved for Ref2VA phase 2")
    if alignment is not None and alignment < 1:
        raise ValueError("alignment must be positive")

    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    text_rows = int(text_token_tags.numel())
    condition_video_rows = len(keyframe_anchors) * rows_per_frame
    target_audio_rows = num_audio_latents * audio_channels
    target_video_rows = (num_latent_frames // patch_t) * rows_per_frame
    used = text_rows + condition_video_rows + target_audio_rows + target_video_rows
    sequence_length = used if alignment is None else math.ceil(used / alignment) * alignment

    condition_start = text_rows
    audio_start = condition_start + condition_video_rows
    video_start = audio_start + target_audio_rows
    text_indices = torch.arange(text_rows, dtype=torch.long)
    audio_indices = torch.arange(audio_start, video_start, dtype=torch.long)
    video_indices = torch.cat(
        [
            torch.arange(condition_start, audio_start, dtype=torch.long),
            torch.arange(video_start, used, dtype=torch.long),
        ]
    )

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:text_rows, 0] = torch.arange(text_rows, dtype=torch.float64)
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_grid(latent_width, patch_w, sqrt_area)
    hh, ww = torch.meshgrid(height_grid, width_grid, indexing="ij")
    frame_grid = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(text_rows)
        elif anchor == "last":
            anchor_time = _last_anchor_time(num_latent_frames, float(text_rows))
        else:
            raise ValueError(f"keyframe anchor must be 'first' or 'last', got {anchor!r}")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    audio_time = float(text_rows) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(audio_channels)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )
    video_positions = torch.empty(num_latent_frames // patch_t, rows_per_frame, 3, dtype=torch.float64)
    video_positions[:, :, 0] = _temporal_grid(num_latent_frames // patch_t, float(text_rows))[:, None]
    video_positions[:, :, 1:] = frame_grid[None]
    position_ids[video_start:used] = video_positions.reshape(-1, 3)

    token_tags = torch.full((sequence_length,), PADDING_TAG, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[video_indices] = VIDEO_TAG
    token_tags[audio_indices] = AUDIO_TAG
    attention_mask = None
    if used != sequence_length:
        attention_mask = torch.zeros(sequence_length, dtype=torch.bool)
        attention_mask[:used] = True

    return H3PackedLayout(
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        attention_mask=attention_mask,
        cu_seqlens=torch.tensor(
            [0, used, sequence_length] if used != sequence_length else [0, used],
            dtype=torch.int32,
        ),
        num_text_rows=text_rows,
        num_condition_video_rows=condition_video_rows,
        num_condition_audio_rows=condition_audio_rows,
        num_target_video_rows=target_video_rows,
        num_target_audio_rows=target_audio_rows,
        num_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
        patch_size=patch_size,
        audio_channels=audio_channels,
        sequence_length=sequence_length,
        used_sequence_length=used,
    )


def patchify_video(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError("video latents must have shape [B,C,T,H,W]")
    pt, ph, pw = patch_size
    batch, channels, frames, height, width = latents.shape
    if frames % pt or height % ph or width % pw:
        raise ValueError("video latent geometry is not divisible by patch_size")
    rows = latents.reshape(batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw)
    rows = rows.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return rows.reshape(batch, -1, channels * pt * ph * pw).contiguous()


def unpatchify_video(rows: torch.Tensor, layout: H3PackedLayout, channels: int) -> torch.Tensor:
    if rows.ndim != 3 or rows.shape[1] != layout.num_target_video_rows:
        raise ValueError("video rows do not match packed layout")
    pt, ph, pw = layout.patch_size
    batch = rows.shape[0]
    frames = layout.num_latent_frames // pt
    height = layout.latent_height // ph
    width = layout.latent_width // pw
    values = rows.reshape(batch, frames, height, width, channels, pt, ph, pw)
    values = values.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return values.reshape(batch, channels, layout.num_latent_frames, layout.latent_height, layout.latent_width)


def pack_audio(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError("audio latents must have shape [B,2,C,T]")
    return latents.permute(0, 1, 3, 2).reshape(latents.shape[0], -1, latents.shape[2]).contiguous()


def unpack_audio(rows: torch.Tensor, audio_channels: int, latent_channels: int) -> torch.Tensor:
    if rows.ndim != 3 or rows.shape[1] % audio_channels:
        raise ValueError("audio rows do not match channel-major layout")
    frames = rows.shape[1] // audio_channels
    return rows.reshape(rows.shape[0], audio_channels, frames, latent_channels).permute(0, 1, 3, 2).contiguous()


__all__ = [
    "AUDIO_TAG",
    "H3PackedLayout",
    "PADDING_TAG",
    "TEXT_TAG",
    "VIDEO_TAG",
    "build_packed_layout",
    "pack_audio",
    "patchify_video",
    "unpack_audio",
    "unpatchify_video",
]

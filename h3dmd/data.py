from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from h3dmd.registry import DATASETS


@dataclass
class H3Batch:
    prompt_embeds: torch.Tensor
    text_token_tags: torch.Tensor
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    sample_ids: list[str]

    def to(self, device: torch.device | str) -> "H3Batch":
        return H3Batch(
            prompt_embeds=self.prompt_embeds.to(device),
            text_token_tags=self.text_token_tags.to(device),
            video_latents=self.video_latents.to(device),
            audio_latents=self.audio_latents.to(device),
            sample_ids=self.sample_ids,
        )


def _load_tensor_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path)
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"cache file must contain a mapping: {path}")
    return value


@DATASETS.register("h3_cached")
class CachedH3Dataset(Dataset[dict[str, Any]]):
    """Dataset over immutable offline Qwen/VAE cache records."""

    def __init__(
        self, manifest: str | Path, cache_root: str | Path = "", *,
        height: int | None = None, width: int | None = None, num_frames: int | None = None,
    ) -> None:
        self.expected_height = height
        self.expected_width = width
        self.expected_frames = num_frames
        self.manifest = Path(manifest).expanduser().resolve()
        if not self.manifest.is_file():
            raise FileNotFoundError(self.manifest)
        self.root = Path(cache_root).expanduser().resolve() if cache_root else self.manifest.parent
        self.entries: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            if "cache_path" not in entry:
                raise KeyError(f"manifest line {line_number} has no cache_path")
            self.entries.append(entry)
        if not self.entries:
            raise ValueError(f"empty H3 cache manifest: {self.manifest}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        path = Path(entry["cache_path"])
        if not path.is_absolute():
            path = self.root / path
        if entry.get("text_cache_key"):
            from safetensors import safe_open
            if not entry.get("shape_only"):
                raise ValueError("packed text caches are only valid for shape-only DMD")
            key = entry["text_cache_key"]
            frames = int(entry["num_frames"])
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                values = {name: handle.get_tensor(f"{key}.{name}")
                          for name in ("prompt_embeds", "text_token_tags")}
            values.update(
                video_latents=torch.zeros(1).expand(
                    24, (frames - 5) // 17 * 5 + 2,
                    (self.expected_height or entry["height"]) // 16,
                    (self.expected_width or entry["width"]) // 16),
                audio_latents=torch.zeros(1).expand(2, 32, round(frames / 24 * 40)))
        else:
            values = _load_tensor_file(path)
        required = {"prompt_embeds", "text_token_tags", "video_latents", "audio_latents"}
        missing = required - values.keys()
        if missing:
            raise KeyError(f"{path} lacks cached tensors {sorted(missing)}")
        prompt = values["prompt_embeds"]
        if prompt.ndim == 3 and prompt.shape[0] == 1:
            prompt = prompt[0]
        video = values["video_latents"]
        if video.ndim == 5 and video.shape[0] == 1:
            video = video[0]
        audio = values["audio_latents"]
        if audio.ndim == 4 and audio.shape[0] == 1:
            audio = audio[0]
        if prompt.ndim != 2 or video.ndim != 4 or audio.ndim != 3:
            raise ValueError(
                f"bad cache shapes in {path}: text={prompt.shape}, video={video.shape}, audio={audio.shape}"
            )
        # Shape-only DMD caches contain zero AV targets.  Their text cache is
        # resolution independent, so a retargeted manifest may request a new
        # spatial layout without rewriting tens of thousands of prompt files.
        # Real AV caches must still match the encoded dimensions exactly.
        if entry.get("shape_only"):
            target_h = self.expected_height or entry.get("height")
            target_w = self.expected_width or entry.get("width")
            if target_h is not None and target_w is not None:
                target_h_latent, target_w_latent = int(target_h) // 16, int(target_w) // 16
                if video.shape[2:] != (target_h_latent, target_w_latent):
                    video = torch.zeros(
                        (video.shape[0], video.shape[1], target_h_latent, target_w_latent),
                        dtype=video.dtype,
                    )
        if video.shape[0] != 24:
            raise ValueError(f"H3 video cache must have 24 latent channels, got {video.shape}")
        if audio.shape[:2] != (2, 32):
            raise ValueError(f"H3 audio cache must be stereo [2,32,T], got {audio.shape}")
        video_frames = video.shape[1]
        if video_frames < 2 or (video_frames - 2) % 5:
            raise ValueError(f"H3 video latent length must follow 5*n+2, got {video_frames}")
        pixel_frames = 17 * ((video_frames - 2) // 5) + 5
        for name, actual, expected in (
            ("height", video.shape[2] * 16, self.expected_height),
            ("width", video.shape[3] * 16, self.expected_width),
            ("num_frames", pixel_frames, self.expected_frames),
        ):
            if expected is not None and actual != expected:
                raise ValueError(f"cache {name} mismatch in {path}: expected {expected}, got {actual}")
        if "num_frames" in entry and entry["num_frames"] != pixel_frames:
            raise ValueError(f"manifest num_frames disagrees with cached video in {path}")
        expected_audio_frames = round(pixel_frames / 24 * 40)
        if audio.shape[-1] != expected_audio_frames:
            raise ValueError(
                f"audio/video duration mismatch in {path}: expected {expected_audio_frames} "
                f"audio latent frames for {video_frames} video latent frames, got {audio.shape[-1]}"
            )
        tags = values["text_token_tags"].to(torch.long).flatten()
        if tags.numel() != prompt.shape[0]:
            raise ValueError(f"text tag count does not match embeddings in {path}")
        if not torch.all(tags == 1):
            raise ValueError(f"phase-1 T2VA text_token_tags must all equal 1 in {path}")
        return {
            "sample_id": str(entry.get("id", index)),
            "prompt_embeds": prompt,
            "text_token_tags": tags,
            "video_latents": video.float(),
            "audio_latents": audio.float(),
        }


class AlternatingFrameSampler(Sampler[int]):
    """Every global step uses one duration; cycle durations in configured order.

    Shuffle within each bucket and repeat smaller buckets to give every duration
    equal optimizer steps. Each rank receives a different sample when possible.
    """

    def __init__(self, dataset, frame_buckets=None, *, resolution_schedule=None, size_schedule=None,
                 rank=0, world_size=1, seed=42, shuffle=True):
        self.kind = "size" if size_schedule else ("resolution" if resolution_schedule else "frames")
        if size_schedule or resolution_schedule:
            schedule = size_schedule or resolution_schedule
            group_keys = list(dict.fromkeys(schedule))
            self.schedule = list(schedule)
        else:
            group_keys = list(frame_buckets or [])
            self.schedule = group_keys
        if not group_keys:
            raise ValueError("sampler needs frame_buckets, resolution_schedule, or size_schedule")
        self.groups = {key: [] for key in group_keys}
        for index, entry in enumerate(dataset.entries):
            if size_schedule:
                key = f"{entry.get('width')}x{entry.get('height')}x{entry.get('num_frames')}"
                if key not in self.groups:
                    raise ValueError(f"manifest entry {index} needs size in {group_keys}, got {key}")
            elif resolution_schedule:
                width, height = entry.get("width"), entry.get("height")
                key = f"{width}x{height}"
                if key not in self.groups:
                    raise ValueError(
                        f"manifest entry {index} needs resolution in {group_keys}, got {key}"
                    )
            else:
                key = entry.get("num_frames")
                if key not in self.groups:
                    raise ValueError(f"manifest entry {index} needs num_frames in {group_keys}")
            self.groups[key].append(index)
        if any(not group for group in self.groups.values()):
            raise ValueError("every configured frame bucket must contain samples")
        self.rank, self.world_size = rank, world_size
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0
        self.cycles = max(math.ceil(len(group) / world_size) for group in self.groups.values())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.cycles * len(self.schedule)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        groups = []
        for key in self.groups:
            group = self.groups[key]
            order = torch.randperm(len(group), generator=generator).tolist() if self.shuffle else range(len(group))
            groups.append([group[i] for i in order])
        group_orders = dict(zip(self.groups, groups, strict=True))
        group_uses = {key: 0 for key in self.groups}
        for cycle in range(self.cycles):
            for key in self.schedule:
                group = group_orders[key]
                offset = group_uses[key]
                yield group[(offset * self.world_size + self.rank) % len(group)]
                group_uses[key] = offset + 1


def collate_h3_batch(samples: list[dict[str, Any]]) -> H3Batch:
    if not samples:
        raise ValueError("cannot collate an empty H3 batch")
    tensor_keys = ("prompt_embeds", "text_token_tags", "video_latents", "audio_latents")
    for key in tensor_keys:
        shapes = {tuple(sample[key].shape) for sample in samples}
        if len(shapes) != 1:
            raise ValueError(
                f"H3 packed batches require one shared layout; bucket by shape/text length before collating {key}: {shapes}"
            )
    if any(not torch.equal(sample["text_token_tags"], samples[0]["text_token_tags"]) for sample in samples[1:]):
        raise ValueError("H3 packed batches require identical text token tags")
    return H3Batch(
        prompt_embeds=torch.stack([sample["prompt_embeds"] for sample in samples]),
        # The transformer's structural text tags are shared by the replicated
        # batch axis; all items in a bucket must have identical tags.
        text_token_tags=samples[0]["text_token_tags"],
        video_latents=torch.stack([sample["video_latents"] for sample in samples]),
        audio_latents=torch.stack([sample["audio_latents"] for sample in samples]),
        sample_ids=[sample["sample_id"] for sample in samples],
    )


__all__ = ["CachedH3Dataset", "H3Batch", "collate_h3_batch"]

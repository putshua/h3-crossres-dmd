from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Literal
import torch
from h3dmd.data import H3Batch


@dataclass
class H3LossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass
class H3Latents:
    video: torch.Tensor
    audio: torch.Tensor

    def detach(self) -> "H3Latents":
        return H3Latents(self.video.detach(), self.audio.detach())


@dataclass(frozen=True)
class DistillationContext:
    batch: H3Batch
    iteration: int


@dataclass(frozen=True)
class TrainingClosure:
    name: str
    role: Literal["student", "critic"]
    compute: Callable[[], H3LossOutput]

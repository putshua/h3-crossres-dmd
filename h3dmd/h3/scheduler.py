from __future__ import annotations

from dataclasses import dataclass

import torch


def shifted_sigma(q: torch.Tensor, shift: float) -> torch.Tensor:
    """Map the shared unshifted noise coordinate q to one modality's sigma."""

    if shift <= 0:
        raise ValueError("shift must be positive")
    if not torch.is_floating_point(q):
        raise TypeError("q must be floating point")
    if bool(((q < 0) | (q > 1)).any()):
        raise ValueError("q must be in [0,1]")
    return shift * q / (1.0 + (shift - 1.0) * q)


def truncated_q_grid(q: list[float], num_forwards: int) -> list[float]:
    """Use training input times unchanged; the last forward predicts clean x0."""
    if not 1 <= num_forwards <= len(q) - 1:
        raise ValueError("requested forwards must be within the configured trajectory; "
                         "pass an explicit q_schedule to use a different grid")
    return list(q[:num_forwards]) + [0.0]


@dataclass(frozen=True)
class H3NoiseLevel:
    q: torch.Tensor
    video_sigma: torch.Tensor
    audio_sigma: torch.Tensor
    video_timestep: torch.Tensor
    audio_timestep: torch.Tensor


@dataclass(frozen=True)
class H3Schedule:
    q: torch.Tensor
    video_sigmas: torch.Tensor
    audio_sigmas: torch.Tensor

    @property
    def video_timesteps(self) -> torch.Tensor:
        return 1.0 - self.video_sigmas[:-1]

    @property
    def audio_timesteps(self) -> torch.Tensor:
        return 1.0 - self.audio_sigmas[:-1]

    @property
    def num_forwards(self) -> int:
        return int(self.video_sigmas.numel() - 1)

    def to(self, device: torch.device | str) -> "H3Schedule":
        return H3Schedule(
            q=self.q.to(device),
            video_sigmas=self.video_sigmas.to(device),
            audio_sigmas=self.audio_sigmas.to(device),
        )


class H3DualNoiseScheduler:
    """H3 rectified-flow math for the coupled video/audio trajectories.

    Both modalities share one base coordinate ``q``. The configured shifts map
    that coordinate to different sigmas, then the transformer consumes
    ``t=1-sigma`` and predicts ``v=x0-eps``.
    """

    def __init__(self, video_shift: float = 12.0, audio_shift: float = 3.0) -> None:
        if video_shift <= 0 or audio_shift <= 0:
            raise ValueError("scheduler shifts must be positive")
        self.video_shift = float(video_shift)
        self.audio_shift = float(audio_shift)

    def level(self, q: float | torch.Tensor, device: torch.device | str | None = None) -> H3NoiseLevel:
        q_tensor = torch.as_tensor(q, dtype=torch.float32, device=device)
        video_sigma = shifted_sigma(q_tensor, self.video_shift)
        audio_sigma = shifted_sigma(q_tensor, self.audio_shift)
        return H3NoiseLevel(
            q=q_tensor,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
            video_timestep=1.0 - video_sigma,
            audio_timestep=1.0 - audio_sigma,
        )

    def schedule(
        self,
        num_inference_steps: int = 50,
        *,
        q: list[float] | torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> H3Schedule:
        if q is None:
            if num_inference_steps < 2:
                raise ValueError("num_inference_steps counts grid points and must be >=2")
            q_tensor = torch.linspace(1.0, 0.0, num_inference_steps, dtype=torch.float32)
        else:
            q_tensor = torch.as_tensor(q, dtype=torch.float32).flatten()
            if q_tensor.numel() < 2:
                raise ValueError("explicit q schedule needs at least two grid points")
            if not torch.all(q_tensor[:-1] > q_tensor[1:]) or float(q_tensor[-1]) != 0.0:
                raise ValueError("explicit q schedule must be strictly decreasing and terminate at 0")
        video = torch.unique_consecutive(shifted_sigma(q_tensor, self.video_shift))
        audio = torch.unique_consecutive(shifted_sigma(q_tensor, self.audio_shift))
        if video.numel() != audio.numel():
            raise ValueError("video/audio shifted schedules collapsed to unequal lengths")
        return H3Schedule(q_tensor.to(device), video.to(device), audio.to(device))

    @staticmethod
    def _broadcast(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        value = value.to(device=reference.device, dtype=reference.dtype)
        while value.ndim < reference.ndim:
            value = value.unsqueeze(-1)
        return value

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if x0.shape != noise.shape:
            raise ValueError("x0 and noise shapes must match")
        sigma = self._broadcast(sigma, x0)
        return (1.0 - sigma) * x0 + sigma * noise

    @staticmethod
    def velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return x0 - noise

    def to_x0(self, xt: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if xt.shape != velocity.shape:
            raise ValueError("xt and velocity shapes must match")
        return xt + self._broadcast(sigma, xt) * velocity

    def euler_step(
        self,
        xt: torch.Tensor,
        velocity: torch.Tensor,
        sigma: float | torch.Tensor,
        sigma_next: float | torch.Tensor,
    ) -> torch.Tensor:
        sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=xt.device)
        sigma_next_t = torch.as_tensor(sigma_next, dtype=torch.float32, device=xt.device)
        if bool((sigma_t <= 0).any()):
            raise ValueError("Euler step requires sigma > 0")
        x0 = self.to_x0(xt.float(), velocity.float(), sigma_t)
        ratio = self._broadcast(sigma_next_t / sigma_t, xt.float())
        result = ratio * xt.float() + (1.0 - ratio) * x0
        return result.to(xt.dtype)


__all__ = ["H3DualNoiseScheduler", "H3NoiseLevel", "H3Schedule", "shifted_sigma"]

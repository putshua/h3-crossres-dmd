from __future__ import annotations

import torch
from typing import Literal
from h3dmd.config import H3Config
from h3dmd.data import H3Batch
from h3dmd.h3.layout import H3PackedLayout, build_packed_layout
from h3dmd.h3.scheduler import H3DualNoiseScheduler, H3NoiseLevel
from h3dmd.models import H3ModelBundle
from h3dmd.methods.types import H3Latents, H3LossOutput


class NativeH3Method:
    """Shared native RF math and model calls; independent of distillation loss."""

    def __init__(self, config: H3Config, models: H3ModelBundle) -> None:
        config.validate()
        self.config = config
        self.models = models
        self.scheduler = H3DualNoiseScheduler(
            video_shift=config.schedule.video_shift,
            audio_shift=config.schedule.audio_shift,
        )
        self.training_scheduler = H3DualNoiseScheduler(
            video_shift=(config.schedule.train_video_shift
                         if config.schedule.train_video_shift is not None
                         else config.schedule.video_shift),
            audio_shift=(config.schedule.train_audio_shift
                         if config.schedule.train_audio_shift is not None
                         else config.schedule.audio_shift),
        )

    def build_layout(self, batch: H3Batch) -> H3PackedLayout:
        video = batch.video_latents
        audio = batch.audio_latents
        return build_packed_layout(
            text_token_tags=batch.text_token_tags,
            num_latent_frames=video.shape[2],
            latent_height=video.shape[3],
            latent_width=video.shape[4],
            num_audio_latents=audio.shape[3],
            patch_size=self.config.geometry.patch_size,
            audio_channels=self.config.geometry.audio_channels,
            alignment=self.config.geometry.packed_alignment,
        ).to(video.device)

    def _sample_level(self, device: torch.device) -> H3NoiseLevel:
        minimum = self.config.schedule.train_q_min
        maximum = self.config.schedule.train_q_max
        # One q per packed microbatch: Diffusers H3's batch axis is pure
        # replication and all items share the row-timestep table.
        q = (
            torch.rand((), device=device, dtype=torch.float32) * (maximum - minimum)
            + minimum
        )
        return self.training_scheduler.level(q, device=device)

    def _call(
        self,
        role: Literal["student", "teacher", "critic"],
        latents: H3Latents,
        batch: H3Batch,
        layout: H3PackedLayout,
        level: H3NoiseLevel,
    ):
        model = getattr(self.models, role)
        # A rollout switches from no_grad to grad inside an outer autocast.
        # Cached casts created by an early no_grad forward would detach FP32
        # parameters (including LoRA) on the final differentiable forward.
        with torch.autocast(
            "cuda",
            enabled=torch.is_autocast_enabled(),
            dtype=torch.bfloat16,
            cache_enabled=False,
        ):
            return model(
                latents.video,
                latents.audio,
                batch.prompt_embeds,
                layout=layout,
                video_timestep=level.video_timestep,
                audio_timestep=level.audio_timestep,
            )

    def _noise(
        self, clean: H3Latents, level: H3NoiseLevel
    ) -> tuple[H3Latents, H3Latents]:
        video_noise = torch.randn_like(clean.video, dtype=torch.float32)
        audio_noise = torch.randn_like(clean.audio, dtype=torch.float32)
        noisy = H3Latents(
            self.scheduler.add_noise(
                clean.video.float(), video_noise, level.video_sigma
            ),
            self.scheduler.add_noise(
                clean.audio.float(), audio_noise, level.audio_sigma
            ),
        )
        return noisy, H3Latents(video_noise, audio_noise)

    def _weighted_modal_loss(
        self, video_loss: torch.Tensor, audio_loss: torch.Tensor, scale: float
    ) -> H3LossOutput:
        def reduce(value):
            if self.config.dmd.loss_reduction == "sum":
                return value.flatten(1).sum(1).mean()
            return value.mean()

        video = reduce(video_loss)
        audio = reduce(audio_loss)
        total = scale * (
            self.config.dmd.video_loss_weight * video
            + self.config.dmd.audio_loss_weight * audio
        )
        return H3LossOutput(total, {"video": video.detach(), "audio": audio.detach()})

    def _project(
        self, state, velocity, start: H3NoiseLevel, end: H3NoiseLevel
    ) -> H3Latents:
        # x_end = x_start + (sigma_start - sigma_end) * v. Unlike the
        # ratio-based solver this is also defined at sigma_start=end=0.
        return H3Latents(
            state.video.float()
            + (start.video_sigma - end.video_sigma) * velocity.video.float(),
            state.audio.float()
            + (start.audio_sigma - end.audio_sigma) * velocity.audio.float(),
        )

from __future__ import annotations

import torch.nn.functional as F
from typing import Literal
from h3dmd.data import H3Batch
from h3dmd.methods.types import H3Latents, H3LossOutput


class RFLossMixin:
    """Native dataward velocity regression for student or fake-score model."""

    def rf_loss(
        self,
        batch: H3Batch,
        *,
        role: Literal["student", "critic"] = "student",
        clean: H3Latents | None = None,
    ) -> H3LossOutput:
        """Rectified-flow regression with ``v=x0-eps`` for both modalities."""

        clean = clean or H3Latents(batch.video_latents, batch.audio_latents)
        layout = self.build_layout(batch)
        level = self._sample_level(clean.video.device)
        noisy, noise = self._noise(clean, level)
        prediction = self._call(role, noisy, batch, layout, level)
        output = self._weighted_modal_loss(
            F.mse_loss(
                prediction.video.float(),
                clean.video.float() - noise.video,
                reduction="none",
            ),
            F.mse_loss(
                prediction.audio.float(),
                clean.audio.float() - noise.audio,
                reduction="none",
            ),
            self.config.dmd.rf_loss_weight
            if role == "student"
            else self.config.dmd.critic_loss_weight,
        )
        output.metrics.update(
            {
                "q": level.q.detach(),
                "video_sigma": level.video_sigma.detach(),
                "audio_sigma": level.audio_sigma.detach(),
            }
        )
        return output

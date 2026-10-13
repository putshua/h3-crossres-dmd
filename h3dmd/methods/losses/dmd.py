from __future__ import annotations

import torch
from h3dmd.data import H3Batch
from h3dmd.methods.types import H3Latents, H3LossOutput


def _expand_batch_scalar(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        return value
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-1)
    return value


class DMDLossMixin:
    """Distribution matching surrogate and fake-score training target."""

    def prepare_score_inputs(self, clean: H3Latents, batch: H3Batch):
        """Identity in native DMD; cross-resolution algorithms map clean here."""
        return clean, batch

    def _normalize_gradient(
        self,
        gradient: torch.Tensor,
        generated: torch.Tensor,
        teacher_x0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.dmd.normalize_dmd_gradient:
            return torch.nan_to_num(gradient), torch.ones(
                gradient.shape[0], device=gradient.device
            )
        # Native DMD weighting: normalize the fake-vs-real x0 residual by the
        # teacher's current denoising error relative to the generated sample.
        # This retains residual magnitude information and avoids importing
        # Wan's TrigFlow parameterization.
        denominator = (
            (generated.detach().float() - teacher_x0.detach().float())
            .abs()
            .flatten(1)
            .mean(1)
        )
        denominator = denominator.clamp_min(self.config.dmd.normalization_eps)
        normalized = gradient / _expand_batch_scalar(denominator, gradient)
        return torch.nan_to_num(normalized), denominator

    def dmd_loss(self, batch: H3Batch, step: int = 0) -> H3LossOutput:
        """DMD surrogate from critic-vs-teacher x0 estimates on a student rollout."""

        count = self._rollout_count(step)
        generated = self.rollout(batch, requires_grad=True, num_steps=count)
        generated, score_batch = self.prepare_score_inputs(generated, batch)
        layout = self.build_layout(score_batch)
        level = self._sample_level(generated.video.device)
        with torch.no_grad():
            noisy, _ = self._noise(generated.detach(), level)
            critic_v = self._call("critic", noisy, score_batch, layout, level)
            teacher_v = self._call("teacher", noisy, score_batch, layout, level)
            critic_video_x0 = self.scheduler.to_x0(
                noisy.video, critic_v.video, level.video_sigma
            )
            teacher_video_x0 = self.scheduler.to_x0(
                noisy.video, teacher_v.video, level.video_sigma
            )
            critic_audio_x0 = self.scheduler.to_x0(
                noisy.audio, critic_v.audio, level.audio_sigma
            )
            teacher_audio_x0 = self.scheduler.to_x0(
                noisy.audio, teacher_v.audio, level.audio_sigma
            )
            video_gradient, video_normalizer = self._normalize_gradient(
                critic_video_x0 - teacher_video_x0,
                generated.video,
                teacher_video_x0,
            )
            audio_gradient, audio_normalizer = self._normalize_gradient(
                critic_audio_x0 - teacher_audio_x0,
                generated.audio,
                teacher_audio_x0,
            )

        video_target = (generated.video.float() - video_gradient.float()).detach()
        audio_target = (generated.audio.float() - audio_gradient.float()).detach()
        output = self._weighted_modal_loss(
            (generated.video.float() - video_target).square(),
            (generated.audio.float() - audio_target).square(),
            self.config.dmd.dmd_loss_weight,
        )
        output.metrics.update(
            {
                "q": level.q.detach(),
                "video_sigma": level.video_sigma.detach(),
                "audio_sigma": level.audio_sigma.detach(),
                "rollout_steps": torch.tensor(float(count), device=level.q.device),
                "video_grad_abs": video_gradient.abs().mean().detach(),
                "audio_grad_abs": audio_gradient.abs().mean().detach(),
                "video_normalizer": video_normalizer.mean().detach(),
                "audio_normalizer": audio_normalizer.mean().detach(),
            }
        )
        return output

    def critic_loss(self, batch: H3Batch, step: int = 0) -> H3LossOutput:
        mode = self.config.dmd.data_mode
        use_rollout = mode == "rollout" or (mode == "mixed" and step % 2 == 1)
        if use_rollout:
            with torch.no_grad():
                clean = self.rollout(
                    batch,
                    requires_grad=False,
                    num_steps=self._rollout_count(step, critic=True),
                )
        else:
            clean = H3Latents(batch.video_latents, batch.audio_latents)
        result = self.rf_loss(batch, role="critic", clean=clean.detach())
        result.metrics["source_rollout"] = torch.tensor(
            float(use_rollout), device=result.loss.device
        )
        return result

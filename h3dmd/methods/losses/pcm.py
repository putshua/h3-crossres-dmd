from __future__ import annotations

import torch
from h3dmd.data import H3Batch
from h3dmd.methods.types import H3Latents, H3LossOutput
from h3dmd.h3.scheduler import H3NoiseLevel


class PCMLossMixin:
    """Phase-local finite-difference consistency without TrigFlow conversion."""

    def _pcm_level(self, upper_q, lower_q, fraction):
        q = lower_q + (upper_q - lower_q) * fraction
        if self.config.pcm.grid_interpolation == "q":
            return self.scheduler.level(q)
        lower, upper = self.scheduler.level(lower_q), self.scheduler.level(upper_q)
        video = torch.lerp(lower.video_sigma, upper.video_sigma, fraction)
        audio = torch.lerp(lower.audio_sigma, upper.audio_sigma, fraction)
        # q describes position within the configured phase, not an inverse
        # shift of these sigmas. Noise/solver/model calls use the actual sigmas.
        return H3NoiseLevel(q, video, audio, 1. - video, 1. - audio)

    def pcm_loss(self, batch: H3Batch) -> H3LossOutput:
        """Phase-local consistency, frozen teacher Euler target, online student stop-grad.

        Mirrors Bidirectional's PCM closure in H3's shared RF q coordinate.
        ``data`` requires real AV latents; ``rollout`` is an explicit extension
        for prompt-only caches and uses detached student-generated clean states.
        """
        cfg = self.config.pcm
        device = batch.video_latents.device
        layout = self.build_layout(batch)
        with torch.no_grad():
            clean = (
                self.rollout(batch, requires_grad=False)
                if cfg.data_mode == "rollout"
                else H3Latents(batch.video_latents, batch.audio_latents)
            )
            edges = torch.tensor(
                cfg.phase_boundaries_q, device=device, dtype=torch.float32
            )
            phase = torch.randint(len(edges) - 1, (), device=device)
            cell = torch.randint(
                cfg.skipping_interval_steps, cfg.steps_per_phase + 1, (), device=device
            )
            endpoint_q = edges[phase + 1]

            def level(index):
                return self._pcm_level(edges[phase], endpoint_q,
                                       index.float() / cfg.steps_per_phase)

            start = level(cell)
            endpoint = self.scheduler.level(endpoint_q)
            noisy, _ = self._noise(clean.detach(), start)
            target_state = noisy
            for k in range(cfg.skipping_interval_steps):
                current, following = level(cell - k), level(cell - k - 1)
                velocity = self._call("teacher", target_state, batch, layout, current)
                target_state = self._project(target_state, velocity, current, following)
            target_level = level(cell - cfg.skipping_interval_steps)
            # Execute on all ranks, including endpoint samples: identical FSDP
            # collective counts even when ranks sample different phases/cells.
            target_v = self._call("student", target_state, batch, layout, target_level)
            target = self._project(target_state, target_v, target_level, endpoint)
        prediction_v = self._call("student", noisy, batch, layout, start)
        prediction = self._project(noisy, prediction_v, start, endpoint)
        result = self._weighted_modal_loss(
            (prediction.video - target.video).square(),
            (prediction.audio - target.audio).square(),
            cfg.loss_weight,
        )
        result.metrics.update(
            phase_index=phase.float(),
            start_q=start.q,
            target_q=target_level.q,
            endpoint_q=endpoint.q,
            start_video_sigma=start.video_sigma,
            target_video_sigma=target_level.video_sigma,
            endpoint_video_sigma=endpoint.video_sigma,
            start_audio_sigma=start.audio_sigma,
            target_audio_sigma=target_level.audio_sigma,
            endpoint_audio_sigma=endpoint.audio_sigma,
        )
        return result

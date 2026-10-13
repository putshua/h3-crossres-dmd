"""Low-resolution prefix, high-resolution suffix, high-resolution DMD scores."""
from dataclasses import replace

import torch

from h3dmd.registry import METHODS
from .crossres_dmd import H3CrossResolutionDMDMethod


@METHODS.register("h3_multires_dmd")
class H3MultiResolutionDMDMethod(H3CrossResolutionDMDMethod):
    def prepare_score_inputs(self, clean, batch):
        high = (self.config.crossres.score_height // 16,
                self.config.crossres.score_width // 16)
        if clean.video.shape[-2:] == high:
            return clean, replace(batch, video_latents=clean.video)
        return super().prepare_score_inputs(clean, batch)

    def prepare_rollout_transition(self, clean, batch, next_forward):
        if next_forward == self.config.crossres.rollout_high_resolution_from_step:
            return self.prepare_score_inputs(clean, batch)
        return clean, batch

    def _score_metrics(self, result):
        super()._score_metrics(result)
        result.metrics["rollout_high_resolution_from_step"] = torch.tensor(
            float(self.config.crossres.rollout_high_resolution_from_step),
            device=result.loss.device)

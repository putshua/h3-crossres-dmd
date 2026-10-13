"""Low-resolution H3 rollout with frozen neural high-resolution AV scoring."""
from dataclasses import replace

import torch

from h3dmd.registry import METHODS
from h3dmd.upsampling import load_frozen_mapper
from .t2va_distill import H3DistillMethod
from .types import H3Latents


@METHODS.register("h3_crossres_dmd")
class H3CrossResolutionDMDMethod(H3DistillMethod):
    def __init__(self, config, models, score_mapper=None):
        super().__init__(config, models)
        self.score_mapper = score_mapper or load_frozen_mapper(
            config, device=next(models.student.parameters()).device)
        self.score_mapper.eval().requires_grad_(False)

    def prepare_score_inputs(self, clean, batch):
        if clean.video.shape[-2:] != (
                self.config.data.height // 16, self.config.data.width // 16):
            raise ValueError("cross-resolution score input must originate on the student grid")
        self.score_mapper.eval()
        # Frozen parameters, differentiable input: do not wrap this in no_grad
        # during a generator closure. Audio remains on its original timeline.
        with torch.autocast(clean.video.device.type, enabled=False):
            mapped = self.score_mapper(clean.video.float())
        expected = (self.config.crossres.score_height // 16,
                    self.config.crossres.score_width // 16)
        if mapped.shape != (*clean.video.shape[:-2], *expected):
            raise ValueError("upsampler changed channels, temporal positions, or score grid")
        return H3Latents(mapped, clean.audio), replace(batch, video_latents=mapped)

    def rf_loss(self, batch, *, role="student", clean=None):
        if role != "critic" or clean is None:
            raise ValueError("cross-resolution fake FM requires a generated clean target")
        with torch.no_grad():
            clean, score_batch = self.prepare_score_inputs(clean.detach(), batch)
        result = super().rf_loss(score_batch, role=role, clean=clean.detach())
        self._score_metrics(result)
        return result

    def dmd_loss(self, batch, step=0):
        result = super().dmd_loss(batch, step)
        self._score_metrics(result)
        return result

    def _score_metrics(self, result):
        c = self.config
        for key, value in {
                "rollout_height": c.data.height, "rollout_width": c.data.width,
                "score_height": c.crossres.score_height, "score_width": c.crossres.score_width,
                "score_spatial_area_ratio": (c.crossres.score_height * c.crossres.score_width /
                                             (c.data.height * c.data.width)),
                "mapper_trainable": 0}.items():
            result.metrics[key] = torch.tensor(float(value), device=result.loss.device)

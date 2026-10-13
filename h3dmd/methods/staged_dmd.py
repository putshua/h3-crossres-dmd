"""PCM bootstrap, frozen-generator fake bootstrap, then pure DMD.

The trainer and checkpoint machinery remain shared with native H3 DMD.
Scheduler clocks count actual updates of each optimizer, including on resume.
"""
import math

from h3dmd.registry import METHODS
from .t2va_distill import H3DistillMethod


@METHODS.register("h3_pcm_warmup_dmd")
class H3PCMWarmupDMDMethod(H3DistillMethod):
    def phase(self, step):
        if step < self.config.pcm.warmup_steps:
            return "pcm"
        if step < self.dmd_start:
            return "fake_warmup"
        return "dmd"

    @property
    def dmd_start(self):
        return self.config.pcm.warmup_steps + self.config.staged.fake_warmup_steps

    def update_roles(self, step):
        phase = self.phase(step)
        if phase == "pcm":
            return True, False
        if phase == "fake_warmup":
            return False, True
        student = (step - self.dmd_start) % self.config.dmd.student_update_frequency == 0
        return student, not student

    def _rollout_count(self, step, *, critic=False):
        if not self.config.dmd.cycle_rollout_steps:
            return self.config.dmd.rollout_steps
        if self.phase(step) == "fake_warmup":
            effective = step - self.config.pcm.warmup_steps
        else:
            relative = max(0, step - self.dmd_start)
            effective = relative // self.config.dmd.student_update_frequency
            if critic:
                effective = relative - effective - 1
        return effective % self.config.dmd.rollout_steps + 1

    def lr_multiplier(self, role, update):
        cfg = self.config.staged
        if role == "net" and update < self.config.pcm.warmup_steps:
            if update < cfg.pcm_lr_warmup_updates:
                return min(1., (update + 1) / cfg.pcm_lr_warmup_updates)
            progress = (update - cfg.pcm_lr_warmup_updates) / max(
                1, self.config.pcm.warmup_steps - cfg.pcm_lr_warmup_updates - 1)
            return cfg.pcm_lr_min_ratio + (1 - cfg.pcm_lr_min_ratio) * (
                1 + math.cos(math.pi * min(1., progress))) / 2
        if role == "net":
            return min(1., (update - self.config.pcm.warmup_steps + 1) / cfg.dmd_lr_warmup_updates)
        if role == "fake_score":
            return min(1., (update + 1) / cfg.fake_warmup_steps)
        raise ValueError(f"Unknown optimizer role {role}")

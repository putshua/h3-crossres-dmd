from __future__ import annotations

import torch
from h3dmd.registry import METHODS
from .base import NativeH3Method
from .rollout import FewStepRolloutMixin
from .losses.rf import RFLossMixin
from .losses.dmd import DMDLossMixin
from .losses.pcm import PCMLossMixin
from .types import DistillationContext, TrainingClosure, H3LossOutput
from h3dmd.data import H3Batch


@METHODS.register("h3_dmd")  # Compatibility name for earlier configs.
@METHODS.register("h3_distill")
class H3DistillMethod(
    DMDLossMixin, PCMLossMixin, RFLossMixin, FewStepRolloutMixin, NativeH3Method
):
    """Distillation orchestration, analogous to Bidirectional's rCM model hooks.

    New methods may override context construction, student closures, update
    scheduling, or post-optimizer hooks without changing the trainer.
    """

    def update_roles(self, step: int) -> tuple[bool, bool]:
        """Return (student, critic), using global optimizer iterations."""
        dmd_active = self.config.dmd.enabled and self.config.dmd.dmd_loss_weight > 0
        warmup = self.config.pcm.enabled and step < self.config.pcm.warmup_steps
        if warmup or not dmd_active:
            return True, False
        if self.config.dmd.update_schedule == "alternating":
            offset = self.config.pcm.warmup_steps if self.config.pcm.enabled else 0
            student = (step - offset) % self.config.dmd.student_update_frequency == 0
            return student, not student
        return True, step % self.config.dmd.critic_update_frequency == 0

    def _rollout_count(self, step: int, *, critic: bool = False) -> int:
        if not self.config.dmd.cycle_rollout_steps:
            return self.config.dmd.rollout_steps
        warmup = self.config.pcm.warmup_steps if self.config.pcm.enabled else 0
        effective = max(0, step - warmup)
        if self.config.dmd.update_schedule == "alternating" and step >= warmup:
            effective = (step - warmup) // self.config.dmd.student_update_frequency
            if critic:
                effective = step - warmup - effective - 1
        return effective % self.config.dmd.rollout_steps + 1

    def _make_training_ctx(self, batch: H3Batch, iteration: int) -> DistillationContext:
        return DistillationContext(batch, iteration)

    def student_objectives(self, ctx: DistillationContext):
        batch, step = ctx.batch, ctx.iteration
        if self.config.dmd.rf_loss_weight > 0:
            yield TrainingClosure("rf", "student", lambda: self.rf_loss(batch))
        warmup = self.config.pcm.enabled and step < self.config.pcm.warmup_steps
        if self.config.pcm.enabled and (not self.config.pcm.warmup_only or warmup):
            yield TrainingClosure("pcm", "student", lambda: self.pcm_loss(batch))
        if (
            self.config.dmd.enabled
            and self.config.dmd.dmd_loss_weight > 0
            and not warmup
        ):
            yield TrainingClosure("dmd", "student", lambda: self.dmd_loss(batch, step))

    def training_step_closures(self, batch: H3Batch, iteration: int):
        ctx = self._make_training_ctx(batch, iteration)
        student, critic = self.update_roles(iteration)
        if critic:
            yield TrainingClosure(
                "rf", "critic", lambda: self.critic_loss(ctx.batch, ctx.iteration)
            )
        if student:
            yield from self.student_objectives(ctx)

    def on_after_optimizer_step(
        self, iteration: int, updated_roles: tuple[str, ...]
    ) -> None:
        """Extension hook for methods that own an EMA or another target update."""

    def student_loss(self, batch: H3Batch, step: int = 0) -> H3LossOutput:
        """Compatibility/evaluation helper; the trainer backprops closures separately."""
        pieces = [
            (closure.name, closure.compute())
            for closure in self.student_objectives(self._make_training_ctx(batch, step))
        ]
        if not pieces:
            raise ValueError("student has no enabled objective")
        loss = sum(
            (piece.loss for _, piece in pieces),
            torch.zeros((), device=batch.video_latents.device),
        )
        metrics = {
            f"{name}/{key}": value
            for name, piece in pieces
            for key, value in piece.metrics.items()
        }
        return H3LossOutput(loss, metrics)

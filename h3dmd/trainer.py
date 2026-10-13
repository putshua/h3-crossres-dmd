"""Compatibility import; shared trainer lives in h3dmd.trainers."""

from h3dmd.trainers import H3DistillationTrainer

H3Trainer = H3DistillationTrainer

__all__ = ["H3Trainer"]

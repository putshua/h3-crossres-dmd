"""Registered distillation methods. Importing this package registers built-ins."""

from .t2va_distill import H3DistillMethod
from .staged_dmd import H3PCMWarmupDMDMethod
from .crossres_dmd import H3CrossResolutionDMDMethod
from .multires_dmd import H3MultiResolutionDMDMethod
from .staged_crossres_dmd import H3PCMWarmupCrossResolutionDMDMethod
from .types import DistillationContext, TrainingClosure, H3Latents, H3LossOutput
from h3dmd.registry import METHODS


def build_method(config, models):
    return METHODS.build(config.method, config=config, models=models)


__all__ = [
    "H3DistillMethod",
    "build_method",
    "DistillationContext",
    "TrainingClosure",
    "H3Latents",
    "H3LossOutput",
]

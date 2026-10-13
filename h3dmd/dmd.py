"""Compatibility imports; implementation lives in h3dmd.methods."""
from h3dmd.methods import H3DistillMethod, H3Latents, H3LossOutput

H3DMDMethod = H3DistillMethod

__all__ = ["H3DMDMethod", "H3Latents", "H3LossOutput"]

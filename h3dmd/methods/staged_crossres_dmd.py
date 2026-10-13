"""Real-AV native PCM bootstrap followed by neural high-resolution DMD."""
from h3dmd.registry import METHODS
from .staged_dmd import H3PCMWarmupDMDMethod
from .crossres_dmd import H3CrossResolutionDMDMethod


@METHODS.register("h3_pcm_warmup_crossres_dmd")
class H3PCMWarmupCrossResolutionDMDMethod(
        H3PCMWarmupDMDMethod, H3CrossResolutionDMDMethod):
    # Staged roles/LR clocks, native PCM, and high-resolution generated-target
    # fake FM/DMD compose without changing either existing algorithm.
    pass

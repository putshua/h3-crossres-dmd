"""MiniMax-H3-native distribution matching distillation.

Configuration discovery does not import Torch or load model dependencies.
"""

from importlib import import_module
from .config import H3Config, load_config, load_experiment

_LAZY_EXPORTS = {
    "H3ModelAdapter": ("h3dmd.h3.adapter", "H3ModelAdapter"),
    "H3PackedLayout": ("h3dmd.h3.layout", "H3PackedLayout"),
    "build_packed_layout": ("h3dmd.h3.layout", "build_packed_layout"),
    "H3DualNoiseScheduler": ("h3dmd.h3.scheduler", "H3DualNoiseScheduler"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


__all__ = ["H3Config", "load_config", "load_experiment", *_LAZY_EXPORTS]

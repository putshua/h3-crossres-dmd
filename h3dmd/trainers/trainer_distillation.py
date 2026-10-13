"""Compatibility facade; all training runs use Bidirectional's trainer."""
from h3dmd.registry import TRAINERS


@TRAINERS.register("distillation")
class H3DistillationTrainer:
    def __init__(self, config):
        self.config = config

    def train(self):
        from rcm.configs.legacy_h3 import launch_legacy_config
        return launch_legacy_config(self.config)


__all__ = ["H3DistillationTrainer"]

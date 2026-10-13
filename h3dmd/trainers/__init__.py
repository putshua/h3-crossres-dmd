from .trainer_distillation import H3DistillationTrainer
from h3dmd.registry import TRAINERS


def build_trainer(config):
    return TRAINERS.build(config.trainer, config=config)


__all__ = ["H3DistillationTrainer", "build_trainer"]

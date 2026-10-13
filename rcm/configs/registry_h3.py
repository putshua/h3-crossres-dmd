"""Bidirectional's config composition with H3 model/data registrations."""
from typing import Any
import attrs

from imaginaire import config
from rcm.configs.defaults.trainer import register_trainer
from rcm.configs.defaults.optimizer import register_optimizer, register_optimizer_fake_score
from rcm.configs.defaults.scheduler import register_scheduler
from rcm.configs.defaults.checkpoint import register_checkpoint
from rcm.configs.defaults.ckpt_type import register_ckpt_type
from rcm.configs.defaults.h3 import register_h3


@attrs.define(slots=False)
class H3CheckpointConfig(config.CheckpointConfig):
    save_on_exit: bool = True


@attrs.define(slots=False)
class Config(config.Config):
    checkpoint: H3CheckpointConfig = attrs.field(factory=H3CheckpointConfig)
    defaults: list[Any] = attrs.field(factory=lambda: [
        "_self_", {"trainer": "distill"}, {"model": "fsdp_h3_distill"},
        {"data_train": "h3_cached"}, {"optimizer": "fusedadamw"},
        {"optimizer_fake_score": "fusedadamw"}, {"scheduler": "lambdalinear"},
        {"callbacks": "h3"}, {"checkpoint": "local"}, {"ckpt_type": "dcp_distill"},
        {"experiment": "h3_synthetic_audio_dmd"},
    ])

    def validate(self):
        from rcm.configs.h3 import native_config
        native_config(self.model.config.h3).validate()
        if self.checkpoint.save_iter <= 0 or self.checkpoint.save_latest_iter < 0:
            raise ValueError("checkpoint intervals must be positive (latest=0 disables rolling)")
        if self.checkpoint.save_latest_iter and (self.checkpoint.dcp_async_mode_enabled or self.checkpoint.save_to_object_store.enabled):
            raise ValueError("H3 rolling checkpoints require synchronous local/shared-filesystem DCP")
        if self.trainer.distributed_parallelism != "fsdp":
            raise ValueError("H3 uses model-owned FSDP2; trainer.distributed_parallelism must be fsdp")
        super().validate()


def make_config():
    c = Config(model=None, optimizer=None, scheduler=None,
               dataloader_train=None, dataloader_val=None)
    c.job.project, c.job.group, c.job.name = "h3", "dmd", "h3_synthetic_audio_dmd"
    c.job.wandb_mode = "disabled"
    c.trainer.run_validation = False
    c.trainer.distributed_parallelism = "fsdp"
    c.trainer.callbacks = None
    for register in (register_trainer, register_h3, register_optimizer,
                     register_optimizer_fake_score, register_scheduler,
                     register_checkpoint, register_ckpt_type):
        register()
    from rcm.configs.experiments.h3.synthetic_audio import register_experiments
    register_experiments()
    from rcm.configs.experiments.h3.crossres import register_experiments as register_crossres
    register_crossres()
    from rcm.configs.experiments.h3.teacher_fm_diagnostic import register_experiments as register_teacher_diagnostic
    register_teacher_diagnostic()
    return c

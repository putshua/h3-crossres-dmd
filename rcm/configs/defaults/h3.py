"""H3 registrations; importing H3 does not require Wan/JVP CUDA extensions."""
from hydra.core.config_store import ConfigStore
from imaginaire.lazy_config import LazyCall as L
from rcm.configs.h3 import native_defaults


def register_h3():
    cs = ConfigStore.instance()
    cs.store(group="model", name="fsdp_h3_distill", package="model", node={
        "_target_": "rcm.models.h3_distill.H3DistillModel",
        "config": {
            "h3": native_defaults(),
            "net": {"_target_": "rcm.networks.h3.build_h3_network"},
            "optimizer_fake_score": None,
            "ema": {"enabled": False, "rate": 0.1, "iteration_shift": 0},
        },
    })
    cs.store(group="data_train", name="h3_cached", package="dataloader_train", node={
        "_target_": "rcm.datasets.h3.build_h3_dataloader",
        "config": "${model.config.h3}",
    })
    cs.store(group="callbacks", name="h3", package="trainer.callbacks", node={
        "h3": {"_target_": "rcm.callbacks.h3.H3TrainingCallback"},
        "fixed_prompt": {
            "_target_": "rcm.callbacks.h3_fixed_prompt.H3FixedPromptCallback",
            "enabled": True, "every_n": 250, "run_at_start": True,
            "manifest": "test_prompts/h3_high_motion_32/encoded/manifest.jsonl",
            "num_prompts": 32, "num_frames": [243], "seed": 1, "decode": True,
        },
    })

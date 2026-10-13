"""Translate old H3 YAMLs once, then run the upstream trainer."""
import argparse
import os
from dataclasses import asdict
from omegaconf import OmegaConf


def convert_legacy_config(native):
    from rcm.configs.registry_h3 import make_config
    from imaginaire.utils.config_helper import override

    options = ["--"]
    if native.method == "h3_crossres_dmd":
        from rcm.configs.experiments.h3.crossres import EXPERIMENT_NAME
        options.append(f"experiment={EXPERIMENT_NAME}")
    c = override(make_config(), options)
    c.model.config.h3 = OmegaConf.create(asdict(native))
    c.optimizer.lr = native.training.learning_rate
    c.optimizer.weight_decay = native.training.weight_decay
    c.model.config.optimizer_fake_score.lr = native.training.critic_learning_rate
    c.model.config.optimizer_fake_score.weight_decay = native.training.weight_decay
    c.trainer.seed = native.training.seed
    c.trainer.max_iter = native.training.max_steps
    c.trainer.grad_accum_iter = native.training.gradient_accumulation_steps
    c.trainer.logging_iter = native.logging.log_every
    c.checkpoint.save_iter = native.checkpoint.save_every
    c.checkpoint.save_on_exit = native.checkpoint.save_on_exit
    c.checkpoint.load_path = native.checkpoint.resume or ""
    return c


def launch_legacy_config(native):
    from scripts.train import launch
    os.environ["IMAGINAIRE_OUTPUT_PATH_LOCAL"] = native.checkpoint.output_dir
    return launch(convert_legacy_config(native), argparse.Namespace())

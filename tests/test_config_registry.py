import argparse

import pytest

from h3dmd.config import config_from_dict, load_config, load_experiment
from h3dmd.cli.common import add_config_arguments, config_from_args
from h3dmd.configs import experiment_names


def test_named_and_yaml_experiments_are_identical_and_independent():
    first = load_config("configs/h3_dmd_4step_512x288_synthetic_audio.yaml")
    named = load_experiment("h3_synthetic_audio_dmd")
    assert first.to_dict() == named.to_dict()
    assert not named.pcm.enabled and named.dmd.rf_loss_weight == 0
    joint = load_experiment("h3_synthetic_audio_pcm_dmd")
    assert joint.pcm.enabled
    joint.pcm.phase_boundaries_q[1] = 0.9
    assert load_experiment("h3_synthetic_audio_dmd").pcm.phase_boundaries_q[1] == 0.75
    assert len(experiment_names()) >= 2


def test_config_precedence_groups_local_keys_then_cli():
    config = config_from_dict(
        {
            "experiment": "h3_synthetic_audio_dmd",
            "defaults": ["method/pcm_dmd"],
            "pcm": {"loss_weight": 0.2},
            "training": {"max_steps": 100},
        },
        ["training.max_steps=17", "pcm.loss_weight=0.3"],
    )
    assert config.pcm.enabled and config.pcm.loss_weight == 0.3
    assert config.training.max_steps == 17
    assert config.data.width == 512
    with pytest.raises(KeyError, match="unknown experiment"):
        load_experiment("not_registered")
    with pytest.raises(KeyError, match="unknown config keys"):
        load_experiment("h3_synthetic_audio_dmd", ["pcm.typo=1"])


def test_cli_named_experiment_and_yaml_are_mutually_exclusive():
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    args = parser.parse_args(
        ["--experiment", "h3_synthetic_audio_dmd", "--set", "training.max_steps=9"]
    )
    assert config_from_args(args).training.max_steps == 9
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--experiment", "h3_synthetic_audio_dmd", "--config", "other.yaml"]
        )


def test_pcm_real_data_mode_rejects_shape_only_cache():
    config = load_experiment("h3_synthetic_audio_dmd")
    config.validate_cached_data(shape_only=True)
    config.pcm.enabled = True
    config.pcm.data_mode = "data"
    with pytest.raises(ValueError, match="real AV latents"):
        config.validate_cached_data(shape_only=True)
    config.validate_cached_data(shape_only=False)
    config.pcm.data_mode = "rollout"
    config.validate_cached_data(shape_only=True)

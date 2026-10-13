import json

import pytest
import torch
import yaml

from h3dmd.config import H3Config, load_config
from h3dmd.data import CachedH3Dataset, collate_h3_batch


def test_config_rejects_wan_or_cfg_semantics(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"dmd": {"conditional_only": False}}))
    with pytest.raises(ValueError, match="conditional-only"):
        load_config(path)

    config = H3Config()
    config.model.task = "fl2va"
    with pytest.raises(NotImplementedError, match="Phase 1"):
        config.validate()


def test_config_overrides_and_explicit_q_validation(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    config = load_config(
        path,
        ["training.max_steps=3", "dmd.rollout_steps=2", "schedule.inference_q=[1,.5,0]"],
    )
    assert config.training.max_steps == 3
    assert config.schedule.inference_q == [1, 0.5, 0]
    with pytest.raises(KeyError, match="unknown config"):
        load_config(path, ["model.wan_timestep=true"])

    with pytest.raises(ValueError, match="finite"):
        load_config(
            path,
            ["dmd.rollout_steps=2", "schedule.inference_q=[1,.5,.nan]"],
        )


def test_cached_dataset_and_shared_layout_collation(tmp_path) -> None:
    tensors = {
        "prompt_embeds": torch.randn(2, 8),
        "text_token_tags": torch.ones(2, dtype=torch.long),
        "video_latents": torch.randn(24, 2, 4, 4),
        "audio_latents": torch.randn(2, 32, 8),
    }
    torch.save(tensors, tmp_path / "one.pt")
    torch.save(tensors, tmp_path / "two.pt")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "one", "cache_path": "one.pt"}) + "\n"
        + json.dumps({"id": "two", "cache_path": "two.pt"}) + "\n"
    )
    dataset = CachedH3Dataset(manifest)
    batch = collate_h3_batch([dataset[0], dataset[1]])
    assert batch.prompt_embeds.shape == (2, 2, 8)
    assert batch.video_latents.shape == (2, 24, 2, 4, 4)
    assert batch.audio_latents.shape == (2, 2, 32, 8)

    changed = dataset[1]
    changed["text_token_tags"] = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="identical text token tags"):
        collate_h3_batch([dataset[0], changed])

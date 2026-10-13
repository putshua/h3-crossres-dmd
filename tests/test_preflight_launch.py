import json

import pytest
import torch

from h3dmd.cli.preflight import check_prompt_mapping, select_indices, validate_caches
from h3dmd.config import load_config
from h3dmd.data import CachedH3Dataset


def test_stratified_preflight_covers_all_durations_and_ends():
    entries = [{'num_frames': n} for n in (124, 243, 362) for _ in range(100)]
    selected = select_indices(entries, 12)
    assert len(selected) == 12
    assert {0, 99, 100, 199, 200, 299} <= set(selected)
    assert select_indices(entries, 0) == list(range(300))


def test_full_preflight_rejects_nonfinite_text(tmp_path):
    tensors = {'prompt_embeds': torch.ones(2, 5120, dtype=torch.bfloat16),
               'text_token_tags': torch.ones(2, dtype=torch.long),
               'video_latents': torch.zeros(1).expand(24, 2, 18, 32),
               'audio_latents': torch.zeros(1).expand(2, 32, 8)}
    torch.save(tensors, tmp_path / 'valid.pt')
    tensors['prompt_embeds'][0, 0] = float('nan')
    torch.save(tensors, tmp_path / 'bad.pt')
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps({'id': 'valid', 'cache_path': 'valid.pt', 'num_frames': 5}) + '\n')
    data = CachedH3Dataset(manifest, width=512, height=288)
    assert validate_caches(data, samples=0, workers=2)['all_caches_validated']
    manifest.write_text(json.dumps({'id': 'bad', 'cache_path': 'bad.pt', 'num_frames': 5}) + '\n')
    with pytest.raises(ValueError, match='nonfinite'):
        validate_caches(CachedH3Dataset(manifest), samples=0, workers=2)


def test_prompt_mapping_requires_exact_text_and_ids(tmp_path):
    path = tmp_path / 'prompts.jsonl'
    path.write_text(json.dumps({'id': 'a', 'prompt': 'A river, flowing water.'}) + '\n')
    assert check_prompt_mapping([{'id': 'a', 'prompt': 'A river, flowing water.'}], path) == 1
    with pytest.raises(ValueError, match='text differs'):
        check_prompt_mapping([{'id': 'a', 'prompt': 'different'}], path)
    with pytest.raises(ValueError, match='ids differ'):
        check_prompt_mapping([{'id': 'b', 'prompt': 'different'}], path)


def test_full_config_saves_checkpoints_and_uses_new_data():
    config = load_config('configs/h3_dmd_4step_512x288_synthetic_audio.yaml')
    assert config.checkpoint.save_on_exit
    assert config.data.manifest.endswith('h3_promptset_synthetic_audio_v1/manifest.jsonl')
    assert config.model.lora_rank == config.model.critic_lora_rank == 0
    assert config.data.frame_buckets == [124, 243, 362]

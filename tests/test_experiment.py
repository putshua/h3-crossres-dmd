import json
from types import SimpleNamespace

import pytest
import torch

from h3dmd.config import H3Config, load_config
from h3dmd.data import AlternatingFrameSampler, CachedH3Dataset


def test_full_experiment_contract():
    config = load_config('configs/h3_dmd_4step_512x288_full.yaml')
    assert (config.data.width, config.data.height) == (512, 288)
    assert config.data.frame_buckets == [124, 243, 362]
    assert config.model.lora_rank == config.model.critic_lora_rank == 0
    assert config.distributed.fsdp_size == 32
    assert config.training.max_steps == 3000
    assert config.schedule.inference_q == [1, .75, .5, .25, 0]
    assert config.dmd.rf_loss_weight == 0 and config.dmd.data_mode == 'rollout'


def test_duration_cycle_across_ranks_and_unequal_buckets():
    entries = [{'num_frames': n} for n, count in [(124, 9), (243, 4), (362, 6)] for _ in range(count)]
    dataset = SimpleNamespace(entries=entries)
    samplers = [AlternatingFrameSampler(dataset, [124, 243, 362], rank=r, world_size=4) for r in range(4)]
    streams = [list(s) for s in samplers]
    assert all(len(stream) == 9 for stream in streams)
    for step in range(9):
        ids = [stream[step] for stream in streams]
        assert len(set(ids)) == 4
        assert {entries[i]['num_frames'] for i in ids} == {[124, 243, 362][step % 3]}
    assert set(range(len(entries))) == {i for stream in streams for i in stream}
    samplers[0].set_epoch(1)
    assert list(samplers[0]) != streams[0]


def test_exact_cache_size_cycle_keeps_rank_layouts_aligned():
    sizes = [(512, 320, 124), (512, 384, 243), (832, 480, 362)]
    entries = [dict(width=w, height=h, num_frames=f) for w, h, f in sizes for _ in range(8)]
    schedule = [f'{w}x{h}x{f}' for w, h, f in sizes]
    dataset = SimpleNamespace(entries=entries)
    streams = [list(AlternatingFrameSampler(dataset, size_schedule=schedule,
                                            rank=rank, world_size=4, shuffle=False))
               for rank in range(4)]
    for step in range(len(streams[0])):
        observed = {(entries[i]['width'], entries[i]['height'], entries[i]['num_frames'])
                    for stream in streams for i in [stream[step]]}
        assert observed == {sizes[step % len(sizes)]}


def test_invalid_duration_and_sharding_config():
    config = H3Config()
    config.data.height = 288
    config.data.width = 512
    config.validate()
    config.data.frame_buckets = [120]
    with pytest.raises(ValueError, match='17'):
        config.validate()
    config.data.frame_buckets = []
    config.distributed.fsdp_size = 0
    with pytest.raises(ValueError, match='fsdp_size'):
        config.validate()


def test_cache_resolution_and_duration_metadata_are_enforced(tmp_path):
    torch.save({'prompt_embeds': torch.zeros(2, 8), 'text_token_tags': torch.ones(2),
                'video_latents': torch.zeros(24, 2, 18, 32),
                'audio_latents': torch.zeros(2, 32, 8)}, tmp_path / 'cache.pt')
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps({'cache_path': 'cache.pt', 'num_frames': 5}) + '\n')
    assert CachedH3Dataset(manifest, height=288, width=512)[0]['video_latents'].shape[-2:] == (18, 32)
    with pytest.raises(ValueError, match='width mismatch'):
        CachedH3Dataset(manifest, width=1024)[0]
    manifest.write_text(json.dumps({'cache_path': 'cache.pt', 'num_frames': 124}) + '\n')
    with pytest.raises(ValueError, match='disagrees'):
        CachedH3Dataset(manifest)[0]


def test_distributed_checkpoint_restores_fresh_adam_state(tmp_path):
    import torch.distributed as dist
    from h3dmd.checkpoint import H3CheckpointManager
    from h3dmd.distributed import DistributedContext
    from h3dmd.models import H3ModelBundle

    class Role(torch.nn.Linear):
        def freeze(self):
            self.requires_grad_(False)
            return self

    def build():
        models = H3ModelBundle(Role(2, 2), Role(2, 2), Role(2, 2))
        optimizers = {r: torch.optim.AdamW(getattr(models, r).parameters()) for r in ('student', 'critic')}
        return models, optimizers

    dist.init_process_group('gloo', init_method=f'file://{tmp_path}/rendezvous', rank=0, world_size=1)
    try:
        config = H3Config()
        config.checkpoint.output_dir = str(tmp_path / 'run')
        manager = H3CheckpointManager(config.checkpoint, DistributedContext(0, 1, 0, torch.device('cpu')))
        models, optimizers = build()
        for role, opt in optimizers.items():
            getattr(models, role)(torch.ones(1, 2)).sum().backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        path = manager.save(models, optimizers, 7, config)
        restored, fresh_optimizers = build()
        assert manager.load(path, restored, fresh_optimizers) == 7
        for role in optimizers:
            for p, q in zip(getattr(models, role).parameters(), getattr(restored, role).parameters()):
                torch.testing.assert_close(p, q)
                torch.testing.assert_close(optimizers[role].state[p]['exp_avg'], fresh_optimizers[role].state[q]['exp_avg'])
                assert fresh_optimizers[role].state[q]['step'] == 1
    finally:
        dist.destroy_process_group()

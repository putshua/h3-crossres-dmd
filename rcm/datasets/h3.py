"""Cached H3 conditioning/AV layouts exposed as an upstream dataloader."""
from dataclasses import fields

from torch.utils.data import DataLoader, DistributedSampler
from imaginaire.utils import distributed
from h3dmd.data import CachedH3Dataset, AlternatingFrameSampler, collate_h3_batch
from rcm.configs.h3 import native_config


def collate(samples):
    batch = collate_h3_batch(samples)
    return {f.name: getattr(batch, f.name) for f in fields(batch)}


def build_h3_dataloader(config):
    config = native_config(config)
    data = config.data
    if data.dmd_manifest:
        from h3dmd.staged_data import StagedH3Dataset, StagedH3Sampler
        dataset = StagedH3Dataset(config)
        sampler = StagedH3Sampler(dataset, config, rank=distributed.get_rank(),
                                 world_size=distributed.get_world_size())
        return DataLoader(dataset, batch_size=1, sampler=sampler,
            num_workers=data.num_workers, pin_memory=data.pin_memory,
            collate_fn=collate, persistent_workers=data.num_workers > 0)
    dataset = CachedH3Dataset(data.manifest, data.cache_root,
                              height=data.height, width=data.width, num_frames=data.num_frames)
    config.validate_cached_data(shape_only=any(e.get("shape_only") for e in dataset.entries))
    dataset[0]
    sampler = None
    if data.size_schedule:
        sampler = AlternatingFrameSampler(dataset, size_schedule=data.size_schedule,
            rank=distributed.get_rank(), world_size=distributed.get_world_size(),
            seed=config.training.seed, shuffle=data.shuffle)
    elif data.resolution_schedule:
        sampler = AlternatingFrameSampler(dataset, resolution_schedule=data.resolution_schedule,
            rank=distributed.get_rank(), world_size=distributed.get_world_size(),
            seed=config.training.seed, shuffle=data.shuffle)
    elif data.frame_buckets:
        sampler = AlternatingFrameSampler(dataset, data.frame_buckets,
            rank=distributed.get_rank(), world_size=distributed.get_world_size(),
            seed=config.training.seed, shuffle=data.shuffle)
    elif distributed.get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=data.shuffle, seed=config.training.seed,
                                     drop_last=data.drop_last)
    loader = DataLoader(dataset, batch_size=data.batch_size, sampler=sampler,
        shuffle=data.shuffle and sampler is None, num_workers=data.num_workers,
        pin_memory=data.pin_memory, drop_last=data.drop_last, collate_fn=collate,
        persistent_workers=data.num_workers > 0)
    if not len(loader):
        raise ValueError("H3 training dataloader is empty")
    return loader

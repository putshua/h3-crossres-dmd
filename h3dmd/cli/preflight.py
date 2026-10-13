"""Validate H3 manifests/caches and local hardware before distributed loading."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import torch
from h3dmd.cli.common import add_config_arguments, config_from_args
from h3dmd.data import CachedH3Dataset, AlternatingFrameSampler


def select_indices(entries, samples):
    if samples <= 0 or samples >= len(entries):
        return list(range(len(entries)))
    groups = {}
    for i, entry in enumerate(entries):
        groups.setdefault(entry.get('num_frames'), []).append(i)
    selected = []
    for group in groups.values():
        take = min(len(group), max(1, samples // len(groups)))
        selected.extend(group[(len(group) - 1) * j // max(1, take - 1)] for j in range(take))
    present = set(selected)
    for i in range(len(entries)):
        if len(selected) >= samples:
            break
        if i not in present:
            selected.append(i)
    return sorted(selected)


def validate_caches(dataset, samples=96, workers=8):
    indices = select_indices(dataset.entries, samples)
    def check(index):
        item = dataset[index]
        prompt = item['prompt_embeds']
        if prompt.shape[1] != 5120 or prompt.dtype != torch.bfloat16:
            raise ValueError(f'{item["sample_id"]}: expected BF16 H3 text [L,5120], got {prompt.shape}/{prompt.dtype}')
        if not torch.isfinite(prompt).all() or not torch.count_nonzero(prompt):
            raise ValueError(f'{item["sample_id"]}: text embedding is nonfinite or entirely zero')
        return prompt.shape[0]
    lengths = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, length in enumerate(pool.map(check, indices), 1):
            lengths.append(length)
            if done % 1000 == 0:
                print(f'Validated {done}/{len(indices)} caches', file=sys.stderr, flush=True)
    return {'validated_caches': len(indices), 'all_caches_validated': len(indices) == len(dataset),
            'validated_text_tokens_min': min(lengths), 'validated_text_tokens_max': max(lengths)}


def check_prompt_mapping(entries, path):
    prompts = {}
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row['id'] in prompts:
                    raise ValueError(f'duplicate prompt id: {row["id"]}')
                prompts[row['id']] = row['prompt']
    if set(prompts) != {entry['id'] for entry in entries}:
        raise ValueError('prompts.jsonl and manifest ids differ')
    if any(prompts[e['id']] != e.get('prompt') for e in entries):
        raise ValueError('prompt text differs between prompts.jsonl and manifest')
    return len(prompts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--local-gpus', type=int, default=4)
    parser.add_argument('--skip-cuda', action='store_true', help='Input checks only; does not certify GPU readiness')
    parser.add_argument('--samples', type=int, default=96, help='Stratified cache samples; 0 validates every cache')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--prompts', help='Also verify every manifest prompt against this prompts.jsonl')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--ema', action='store_true', help='Include the optional FP32 EMA model in memory estimates')
    parser.add_argument('--student-optimizer-state-bytes', type=int, default=4)
    parser.add_argument('--critic-optimizer-state-bytes', type=int, default=4)
    args = parser.parse_args()
    if args.world_size < 1 or args.local_gpus < 1 or args.samples < 0 or args.workers < 1:
        parser.error('world-size/local-gpus/workers must be positive and samples nonnegative')
    torch.set_num_threads(1)
    config = config_from_args(args)
    if config.distributed.sequence_parallel_size != 1:
        raise ValueError('This experiment uses independent DP samples; keep sequence_parallel_size=1')
    shards = min(config.distributed.fsdp_size or args.world_size, args.world_size)
    if args.world_size % shards:
        raise ValueError('WORLD_SIZE must be divisible by effective FSDP size')
    root = Path(config.model.pretrained_path) / config.model.transformer_subfolder
    weights = list(root.glob('*.safetensors'))
    if not weights:
        raise FileNotFoundError(f'No local transformer weights under {root}')
    weight_gib = sum(p.stat().st_size for p in weights) / 2**30
    full = config.model.lora_rank == 0 and config.model.critic_lora_rank == 0
    # Approximate a mostly-BF16 H3: 3 model copies + 2 gradient copies,
    # plus the configured moments/master bytes per trainable parameter.
    multiplier = 5 + (args.student_optimizer_state_bytes + args.critic_optimizer_state_bytes) / 2
    lower_bound = weight_gib * (multiplier if full else 3) / shards
    if args.ema:
        lower_bound += 2 * weight_gib / shards
    report = {'world_size': args.world_size, 'fsdp_shards': shards, 'full_parameter': full,
              'persistent_memory_lower_bound_gib_per_gpu': lower_bound,
              'hardware_check_scope': 'skipped' if args.skip_cuda else 'local_node_only'}
    if not args.skip_cuda:
        from diffusers import MiniMaxH3Transformer3DModel  # noqa: F401
        from torch.distributed._composable.fsdp import fully_shard  # noqa: F401
        if torch.cuda.device_count() < args.local_gpus:
            raise RuntimeError(f'Need {args.local_gpus} CUDA devices, found {torch.cuda.device_count()}')
        memory = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(args.local_gpus)]
        report['gpu_memory_gib'] = memory
        if lower_bound >= min(memory):
            raise RuntimeError(f'Full training states alone need at least {lower_bound:.1f} GiB/GPU; available {min(memory):.1f}. Use more GPUs.')
    dataset = CachedH3Dataset(config.data.manifest, config.data.cache_root,
                              height=config.data.height, width=config.data.width,
                              num_frames=config.data.num_frames)
    ids = [e.get('id', str(i)) for i, e in enumerate(dataset.entries)]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate manifest ids')
    if config.data.size_schedule:
        sampler = AlternatingFrameSampler(
            dataset, size_schedule=config.data.size_schedule,
            world_size=args.world_size,
        )
        report['steps_per_epoch'] = len(sampler)
        report['size_schedule'] = config.data.size_schedule
        report['size_counts'] = {
            key: sum(
                1 for entry in dataset.entries
                if f"{entry.get('width')}x{entry.get('height')}x{entry.get('num_frames')}" == key
            )
            for key in sorted(set(config.data.size_schedule))
        }
    elif config.data.resolution_schedule:
        sampler = AlternatingFrameSampler(
            dataset, resolution_schedule=config.data.resolution_schedule,
            world_size=args.world_size,
        )
        report['steps_per_epoch'] = len(sampler)
        report['resolution_schedule'] = config.data.resolution_schedule
        report['resolution_counts'] = {
            key: sum(
                1 for entry in dataset.entries
                if f"{entry.get('width')}x{entry.get('height')}" == key
            )
            for key in sorted(set(config.data.resolution_schedule))
        }
    elif config.data.frame_buckets:
        sampler = AlternatingFrameSampler(dataset, config.data.frame_buckets, world_size=args.world_size)
        report['steps_per_epoch'] = len(sampler)
    config.validate_cached_data(shape_only=any(e.get('shape_only') for e in dataset.entries))
    for entry in dataset.entries:
        for name in ('height', 'width'):
            expected = getattr(config.data, name)
            if expected is not None and name in entry and entry[name] != expected:
                raise ValueError(f'manifest {name} mismatch: {entry.get("id")}')
    report.update({'manifest': str(dataset.manifest), 'manifest_sha256': hashlib.sha256(dataset.manifest.read_bytes()).hexdigest(),
                   'records': len(dataset), 'frame_buckets': config.data.frame_buckets,
                   'frame_counts': dict(Counter(e.get('num_frames') for e in dataset.entries))})
    if args.prompts:
        report['matched_prompts'] = check_prompt_mapping(dataset.entries, args.prompts)
    report.update(validate_caches(dataset, args.samples, args.workers))
    report['status'] = 'passed'
    result = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(result + '\n')
    print(result, flush=True)


if __name__ == '__main__':
    main()

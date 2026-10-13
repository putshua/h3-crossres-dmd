"""Evaluate the generator from a training DCP checkpoint on cached fixed prompts."""
import argparse
import datetime
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

from h3dmd.config import load_config
from h3dmd.distributed import DistributedContext
from h3dmd.fixed_prompts import load_fixed_prompts, rank_prompt_slots
from h3dmd.inference import H3InferencePipeline
from h3dmd.precision import configure_h3_precision
from h3dmd.decoders import load_h3_vaes
from h3dmd.cli.infer import _save_output
from rcm.networks.h3 import build_h3_network


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run-dir', dest='run', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--local-transformer-root', default=None,
                   help='Local FL2VA root containing a staged transformer directory')
    p.add_argument('--manifest', default='test_prompts/h3_high_motion_32/encoded/manifest.jsonl')
    p.add_argument('--prompt-ids', default=None,
                   help='Comma-separated IDs to evaluate from the validated fixed-prompt manifest')
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--height', type=int, default=320)
    p.add_argument('--frames', type=int, default=243)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--solver', choices=['euler', 'renoise'], default=None)
    p.add_argument('--phase-grid', action='store_true', help='Use PCM phase edges')
    p.add_argument('--decoder-root', default=None, help='Override node-local VAE path')
    args = p.parse_args()
    torch.set_num_threads(2)
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda', local)
    cfg = load_config(Path(args.run) / 'config.yaml')
    decoder_root = args.decoder_root or cfg.model.pretrained_path
    pcm_only = cfg.pcm.enabled and not (cfg.dmd.enabled and cfg.dmd.dmd_loss_weight > 0)
    solver = args.solver or ('euler' if pcm_only else cfg.dmd.rollout_solver)
    q = cfg.pcm.phase_boundaries_q if args.phase_grid or pcm_only else cfg.schedule.inference_q
    if args.local_transformer_root:
        staged = Path(args.local_transformer_root).resolve()
        if not (staged / cfg.model.transformer_subfolder / 'config.json').is_file():
            raise FileNotFoundError(f'No staged transformer under {staged}')
        cfg.model.pretrained_path = str(staged)
    cfg.distributed.fsdp_size = world
    cfg.model.gradient_checkpointing = False
    configure_h3_precision(cfg)
    rows = load_fixed_prompts(args.manifest)
    if args.prompt_ids:
        selected = args.prompt_ids.split(',')
        if not selected or len(selected) != len(set(selected)):
            raise ValueError('--prompt-ids must contain unique IDs')
        by_id = {row['id']: row for row in rows}
        missing = set(selected) - set(by_id)
        if missing:
            raise ValueError(f'Unknown fixed prompt IDs: {sorted(missing)}')
        rows = [by_id[prompt_id] for prompt_id in selected]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint).resolve()
    print(f'rank={rank} loading generator and DCP {ckpt}', flush=True)
    net = build_h3_network(cfg, 'student', DistributedContext(rank, world, local, device))
    sd = get_model_state_dict(net)
    state = {'net.' + k: v for k, v in sd.items()}
    reader = dcp.FileSystemReader(str(ckpt / 'model'))
    metadata = reader.read_metadata()
    expected = {k for k in metadata.state_dict_metadata if k.startswith('net.')}
    if expected != set(state):
        raise RuntimeError(f'Generator key mismatch: missing={expected-set(state)}, extra={set(state)-expected}')
    for key, value in state.items():
        if tuple(metadata.state_dict_metadata[key].size) != tuple(value.shape):
            raise RuntimeError(f'Checkpoint shape mismatch: {key}')
    dcp.load(state, storage_reader=reader)
    set_model_state_dict(net, {k[4:]: v for k, v in state.items()})
    del state, sd
    net.eval().requires_grad_(False)
    pipe = H3InferencePipeline(net, cfg)
    pending = []
    print(f'rank={rank} generator loaded; sampling', flush=True)
    with torch.inference_mode():
        for index, keep in rank_prompt_slots(len(rows), world, rank):
            row = rows[index]
            start = time.monotonic()
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                result = pipe.generate_latents(row['prompt_embeds'][None], row['text_token_tags'],
                    height=args.height, width=args.width, num_frames=args.frames,
                    solver=solver, q_schedule=q,
                    generator=torch.Generator().manual_seed(args.seed))
            if keep:
                result.video_latents = result.video_latents.cpu()
                result.audio_latents = result.audio_latents.cpu()
                stem = f"{row['id']}_{args.width}x{args.height}_{args.frames}f_seed{args.seed}"
                meta = {k:v for k,v in row.items() if not isinstance(v, torch.Tensor)}
                meta.update(checkpoint=str(ckpt), width=args.width, height=args.height,
                    num_frames=args.frames, seed=args.seed, solver=solver,
                    q=q, video_shift=cfg.schedule.video_shift,
                    audio_shift=cfg.schedule.audio_shift, num_forwards=result.num_forwards,
                    weights='student', fps=cfg.geometry.fps)
                torch.save(dict(video_latents=result.video_latents, audio_latents=result.audio_latents, metadata=meta), out / (stem+'.pt'))
                (out / (stem+'.json')).write_text(json.dumps(meta, indent=2)+'\n')
                pending.append((stem,result))
            print(f'rank={rank} sampled {index+1}/{len(rows)} seconds={time.monotonic()-start:.1f}', flush=True)
        dist.barrier()
        pipe.encoders = load_h3_vaes(decoder_root, device)
        for stem, result in pending:
            result.video_latents = result.video_latents.to(device)
            result.audio_latents = result.audio_latents.to(device)
            _save_output(out / (stem+'.mp4'), pipe.decode(result), cfg.geometry.fps)
            result.videos, result.audio = None, None
            result.video_latents = result.video_latents.cpu()
            result.audio_latents = result.audio_latents.cpu()
            print(f'rank={rank} decoded {stem}', flush=True)
    dist.barrier()
    if rank == 0:
        assert len(list(out.glob('*.mp4'))) == len(rows)
        (out/'COMPLETE.json').write_text(json.dumps(dict(vars(args), checkpoint=str(ckpt), prompts=len(rows)), indent=2)+'\n')
        print(f'COMPLETE: {len(rows)} MP4s', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

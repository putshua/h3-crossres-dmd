"""Evaluate full and truncated training DMD trajectories with one loaded generator."""
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
from h3dmd.h3.scheduler import truncated_q_grid
from h3dmd.distributed import DistributedContext
from h3dmd.fixed_prompts import load_fixed_prompts, rank_prompt_slots
from h3dmd.inference import H3InferencePipeline, align_num_frames
from h3dmd.precision import configure_h3_precision
from h3dmd.decoders import load_h3_vaes
from h3dmd.cli.infer import _save_output
from rcm.networks.h3 import build_h3_network


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', required=True)
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument('--checkpoint')
    weights.add_argument('--teacher', action='store_true', help='Use frozen pretrained H3, without a distilled checkpoint')
    parser.add_argument('--teacher-grid-points', type=int, default=50)
    parser.add_argument('--teacher-video-shift', type=float, default=12)
    parser.add_argument('--teacher-audio-shift', type=float, default=3)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--local-transformer-root')
    parser.add_argument('--modes', choices=['both', 'prefix4', 'native'], default='both')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=320)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--completion-tag', default='', help='Unique marker suffix for a prompt shard')
    args = parser.parse_args()
    torch.set_num_threads(2)
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda', local)
    cfg = load_config(Path(args.run_dir) / 'config.yaml')
    decoder_root = cfg.model.pretrained_path
    if args.local_transformer_root:
        staged = Path(args.local_transformer_root).resolve()
        if not (staged / cfg.model.transformer_subfolder / 'config.json').is_file():
            raise FileNotFoundError(staged)
        cfg.model.pretrained_path = str(staged)
    cfg.distributed.fsdp_size = world
    cfg.model.gradient_checkpointing = False
    configure_h3_precision(cfg)
    count = sum(bool(line.strip()) for line in Path(args.manifest).read_text().splitlines())
    rows = load_fixed_prompts(args.manifest, expected_count=count)
    if args.teacher:
        if args.teacher_grid_points < 2:
            raise ValueError('Teacher grid must contain at least two points')
        cfg.schedule.video_shift = args.teacher_video_shift
        cfg.schedule.audio_shift = args.teacher_audio_shift
        cfg.schedule.inference_q = torch.linspace(1, 0, args.teacher_grid_points, dtype=torch.float32).tolist()
        cfg.schedule.num_inference_steps = args.teacher_grid_points
        cfg.dmd.rollout_steps = args.teacher_grid_points - 1
        cfg.dmd.rollout_solver = 'euler'
        modes = [(f'fullstep{args.teacher_grid_points-1}', 'euler', cfg.schedule.inference_q)]
    else:
        grid = list(cfg.schedule.inference_q)
        if cfg.dmd.rollout_solver != 'renoise':
            raise ValueError('This distilled suite requires training solver=renoise')
        modes = []
        if args.modes in {'both', 'prefix4'}:
            modes.append(('prefix4', cfg.dmd.rollout_solver, truncated_q_grid(grid, 4)))
        if args.modes in {'both', 'native'}:
            modes.append((f'native{len(grid)-1}', cfg.dmd.rollout_solver, grid))
    out = Path(args.output)
    for mode, _, _ in modes:
        (out / mode).mkdir(parents=True, exist_ok=True)
    ckpt = None if args.teacher else Path(args.checkpoint).resolve()
    role = 'teacher' if args.teacher else 'student'
    print(f'rank={rank} loading {role}; checkpoint={ckpt}', flush=True)
    net = build_h3_network(cfg, role, DistributedContext(rank, world, local, device))
    if ckpt is not None:
        sd = get_model_state_dict(net)
        state = {'net.' + k: v for k, v in sd.items()}
        reader = dcp.FileSystemReader(str(ckpt / 'model'))
        metadata = reader.read_metadata()
        expected = {k for k in metadata.state_dict_metadata if k.startswith('net.')}
        if expected != set(state):
            raise RuntimeError('Generator checkpoint key mismatch')
        for key, value in state.items():
            if tuple(metadata.state_dict_metadata[key].size) != tuple(value.shape):
                raise RuntimeError(f'Checkpoint shape mismatch: {key}')
        dcp.load(state, storage_reader=reader)
        set_model_state_dict(net, {k[4:]: v for k, v in state.items()})
        del state, sd
    net.eval().requires_grad_(False)
    pipe = H3InferencePipeline(net, cfg)
    pipe.encoders = load_h3_vaes(decoder_root, device)
    print(f'rank={rank} generator and VAEs loaded; sampling {count} prompts in {modes}', flush=True)
    # Equal-duration rounds keep memory use and FSDP forward counts aligned.
    durations = sorted({int(r['duration_seconds']) for r in rows})
    with torch.inference_mode():
        for duration in durations:
            bucket = [r for r in rows if int(r['duration_seconds']) == duration]
            frames = align_num_frames(duration * cfg.geometry.fps)
            for index, keep in rank_prompt_slots(len(bucket), world, rank):
                row = bucket[index]
                for mode, solver, q in modes:
                    stem = f"{row['id']}_{args.width}x{args.height}_{frames}f_seed{args.seed}"
                    dest = out / mode
                    done = keep and all((dest / (stem + ext)).is_file() for ext in ['.mp4', '.pt', '.json'])
                    if done:
                        old = json.loads((dest / (stem + '.json')).read_text())
                        if old.get('solver') != solver or old.get('q') != q or old.get('checkpoint') != (str(ckpt) if ckpt else None):
                            raise RuntimeError('Existing result has a different schedule/checkpoint; use a new output directory')
                    needed = torch.tensor(int(keep and not done), device=device)
                    dist.all_reduce(needed, op=dist.ReduceOp.MAX)
                    if not needed.item():
                        continue
                    started = time.monotonic()
                    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        result = pipe.generate_latents(
                            row['prompt_embeds'][None], row['text_token_tags'],
                            height=args.height, width=args.width, num_frames=frames,
                            q_schedule=q, solver=solver,
                            generator=torch.Generator().manual_seed(args.seed))
                    if keep and not done:
                        if not torch.isfinite(result.video_latents).all() or not torch.isfinite(result.audio_latents).all():
                            raise RuntimeError(f'Nonfinite inference output: {row["id"]} {mode}')
                        meta = {k: v for k, v in row.items() if not isinstance(v, torch.Tensor)}
                        meta.update(checkpoint=str(ckpt) if ckpt else None,
                                    pretrained_path=decoder_root, mode=mode, solver=solver, q=q,
                                    video_shift=cfg.schedule.video_shift, audio_shift=cfg.schedule.audio_shift,
                                    training_q=list(cfg.schedule.inference_q),
                                    video_sigmas=pipe.scheduler.schedule(q=q).video_sigmas.tolist(),
                                    audio_sigmas=pipe.scheduler.schedule(q=q).audio_sigmas.tolist(),
                                    num_forwards=result.num_forwards, seed=args.seed,
                                    width=args.width, height=args.height, num_frames=frames,
                                    fps=cfg.geometry.fps, weights=role, inference_fsdp_size=world)
                        temporary = dest / (stem + '.pt.tmp')
                        torch.save(dict(video_latents=result.video_latents.cpu(),
                                        audio_latents=result.audio_latents.cpu(), metadata=meta), temporary)
                        temporary.replace(dest / (stem + '.pt'))
                        _save_output(dest / (stem + '.partial.mp4'), pipe.decode(result), cfg.geometry.fps)
                        (dest / (stem + '.partial.mp4')).replace(dest / (stem + '.mp4'))
                        # Keep beginning/middle/end screenshots beside every MP4.
                        from PIL import Image, ImageDraw
                        video = result.videos[0]
                        fps = cfg.geometry.fps
                        indices = [min(int(0.5*fps), video.shape[1]-1),
                                   video.shape[1]//2, max(0, video.shape[1]-int(0.5*fps))]
                        contact = Image.new('RGB', (args.width*3, args.height+24), 'white')
                        draw = ImageDraw.Draw(contact)
                        for column, frame_index in enumerate(indices):
                            frame = (video[:, frame_index].permute(1, 2, 0).clamp(0, 1)*255).round().byte().cpu().numpy()
                            contact.paste(Image.fromarray(frame), (column*args.width, 0))
                            draw.text((column*args.width+4, args.height+4),
                                      f'{row["id"]} / {mode} / {frame_index/fps:.2f}s', fill='black')
                        contact.save(dest / (stem + '.jpg'), quality=92)
                        (dest / (stem + '.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
                        result.videos, result.audio = None, None
                    del result
                    print(f'rank={rank} prompt={row["id"]} duration={duration}s mode={mode} keep={keep} seconds={time.monotonic()-started:.1f}', flush=True)
    dist.barrier()
    if rank == 0:
        for mode, solver, q in modes:
            actual = sum((out / mode / f"{row['id']}_{args.width}x{args.height}_{align_num_frames(int(row['duration_seconds']) * cfg.geometry.fps)}f_seed{args.seed}.mp4").is_file() for row in rows)
            assert actual == count, (mode, actual, count)
        marker = 'COMPLETE' if args.teacher or args.modes=='both' else f'COMPLETE_{args.modes}'
        if args.completion_tag:
            if not args.completion_tag.replace('-', '').replace('_', '').isalnum():
                raise ValueError('completion-tag must be alphanumeric, hyphen or underscore')
            marker += '_' + args.completion_tag
        (out / (marker + '.json')).write_text(json.dumps(dict(vars(args), prompts=count,
                total_mp4=count*len(modes), modes=modes), indent=2) + '\n')
        print(f'COMPLETE: {count * len(modes)} MP4s', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

"""Validate the same resolved Hydra experiment used by scripts.train."""
import argparse
import importlib
import sys
from dataclasses import asdict
import yaml
import torch
from pathlib import Path
from tempfile import TemporaryDirectory

from imaginaire.utils.config_helper import get_config_module, override
from rcm.configs.h3 import native_config
from h3dmd.cli import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--world-size", required=True)
    parser.add_argument("--local-gpus", default="4")
    parser.add_argument("--samples", default="96")
    parser.add_argument("--workers", default="8")
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = override(importlib.import_module(get_config_module(args.config)).make_config(), args.opts)
    native = native_config(config.model.config.h3)
    if native.method in {"h3_pcm_warmup_dmd", "h3_pcm_warmup_crossres_dmd"}:
        boundary = native.pcm.warmup_steps + native.staged.fake_warmup_steps
        if boundary >= config.trainer.max_iter:
            raise ValueError("staged experiment has no pure DMD training budget")
        print(f"H3 staged training: PCM=[0,{native.pcm.warmup_steps}), "
              f"fake-only=[{native.pcm.warmup_steps},{boundary}), "
              f"pure DMD=[{boundary},{config.trainer.max_iter}); "
              "per-optimizer scheduler: PCM linear warmup/cosine; DMD linear warmup/constant", flush=True)
    if native.method in {"h3_crossres_dmd", "h3_pcm_warmup_crossres_dmd", "h3_multires_dmd"}:
        from h3dmd.upsampling import load_frozen_mapper
        load_frozen_mapper(native)
        print(f"H3 cross-resolution: rollout={native.data.width}x{native.data.height}; "
              f"teacher/fake={native.crossres.score_width}x{native.crossres.score_height}; "
              f"mapper_kind={native.crossres.mapper_kind}; frozen mapper={native.crossres.mapper_checkpoint}; "
              f"high_resolution_from_forward={native.crossres.rollout_high_resolution_from_step} "
              "(0=all-low-resolution)", flush=True)
    print(f"H3 checkpoints: latest_every={config.checkpoint.save_latest_iter}, archive_every={config.checkpoint.save_iter}", flush=True)
    for role, opt in (("student", config.optimizer), ("critic", config.model.config.optimizer_fake_score)):
        print(f"H3 {role} optimizer: {opt.optim_type}, backend={opt.get('backend', 'torch')}, "
              f"master_weights={opt.get('master_weights', False)}, lr={opt.lr}", flush=True)

    from h3dmd.h3.scheduler import H3DualNoiseScheduler
    scheduler = H3DualNoiseScheduler(native.schedule.video_shift, native.schedule.audio_shift)
    schedule = scheduler.schedule(native.schedule.num_inference_steps, q=native.schedule.inference_q)
    print(f"H3 rollout/inference schedule: video_shift={native.schedule.video_shift}, audio_shift={native.schedule.audio_shift}, "
          f"q={schedule.q.tolist()}, video_sigmas={schedule.video_sigmas.tolist()}, "
          f"audio_sigmas={schedule.audio_sigmas.tolist()}", flush=True)
    print(f"H3 backward simulation: fixed timestep prefixes, solver={native.dmd.rollout_solver}, "
          f"cycle_rollout_steps={native.dmd.cycle_rollout_steps}, max_forwards={native.dmd.rollout_steps}, "
          f"last_grad_forwards={native.dmd.rollout_grad_steps}", flush=True)
    training_shift = native.schedule.train_video_shift
    if training_shift is None:
        training_shift = native.schedule.video_shift
    training_audio_shift = (native.schedule.train_audio_shift if native.schedule.train_audio_shift is not None
                            else native.schedule.audio_shift)
    training_scheduler = H3DualNoiseScheduler(training_shift, training_audio_shift)
    endpoints = training_scheduler.level(torch.tensor([native.schedule.train_q_min, native.schedule.train_q_max]))
    print(f"H3 score training: video_shift={training_shift}, audio_shift={training_audio_shift}, "
          f"q_range={[native.schedule.train_q_min, native.schedule.train_q_max]}, "
          f"video_sigma_range={endpoints.video_sigma.tolist()}, audio_sigma_range={endpoints.audio_sigma.tolist()}", flush=True)
    fixed = config.trainer.callbacks.get("fixed_prompt")
    if fixed and fixed.enabled:
        from h3dmd.fixed_prompts import load_fixed_prompts
        rows = load_fixed_prompts(fixed.manifest, expected_layer=native.cache.text_encoder_layer)
        print(f"H3 fixed prompts: validated {len(rows)} encoded prompts from {fixed.manifest}, "
              f"every {fixed.every_n} iterations, run_at_start={fixed.run_at_start}, "
              f"frames={list(fixed.num_frames)}, decode={fixed.decode}", flush=True)
    # Reuse cache validation, without a second source of training defaults.
    with TemporaryDirectory(prefix="h3-preflight-") as tmp:
        path = Path(tmp) / "resolved.yaml"
        path.write_text(yaml.safe_dump(asdict(native)))
        argv = ["preflight", "--config", str(path), "--world-size", args.world_size,
                "--local-gpus", args.local_gpus, "--samples", args.samples, "--workers", args.workers]
        for role, opt in (("student", config.optimizer), ("critic", config.model.config.optimizer_fake_score)):
            state_bytes = (12 if opt.get('master_weights', False) else 8) if opt.optim_type == 'fusedadam' else 4
            argv.extend([f"--{role}-optimizer-state-bytes", str(state_bytes)])
        if args.skip_cuda:
            argv.append("--skip-cuda")
        if config.model.config.ema.enabled:
            argv.append("--ema")
        previous, sys.argv = sys.argv, argv
        try:
            preflight.main()
            if native.data.dmd_manifest:
                # Validate prompt-only caches independently; do not pass them
                # through the real-AV PCM data-mode safety check.
                from copy import deepcopy
                dmd = deepcopy(native)
                dmd.method = "h3_crossres_dmd"
                dmd.pcm.enabled = False
                dmd.data.manifest = dmd.data.dmd_manifest
                dmd.data.dmd_manifest = ""
                path.write_text(yaml.safe_dump(asdict(dmd)))
                print("Validating separate DMD prompt source", flush=True)
                preflight.main()
        finally:
            sys.argv = previous


if __name__ == "__main__":
    main()

"""H3 fixed-prompt AV sampling on the upstream EveryN callback lifecycle."""
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from imaginaire.callbacks.every_n import EveryN
from imaginaire.utils import log
from h3dmd.fixed_prompts import load_fixed_prompts, rank_prompt_slots
from h3dmd.inference import H3InferencePipeline
from h3dmd.decoders import load_h3_vaes as load_vaes


@contextmanager
def evaluation_state(net):
    """Sampling and VAE initialization must not change the training RNG or modes."""
    modes = [(module, module.training) for module in net.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    device = next(net.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            net.eval()
            with torch.no_grad():
                yield device
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


class H3FixedPromptCallback(EveryN):
    def __init__(self, *, enabled=True, every_n=500, run_at_start=True,
                 manifest="test_prompts/h3_high_motion_32/encoded/manifest.jsonl",
                 num_prompts=32, num_frames=(243,), seed=1, decode=True):
        if every_n <= 0:
            raise ValueError("fixed-prompt every_n must be positive; use enabled=false to disable")
        super().__init__(every_n=every_n, run_at_start=run_at_start, barrier_after_run=True)
        self.enabled, self.manifest = enabled, str(manifest)
        self.num_prompts, self.num_frames = int(num_prompts), tuple(num_frames)
        self.seed, self.decode = int(seed), decode
        if not 1 <= self.num_prompts <= 32:
            raise ValueError("num_prompts must be in [1,32]")
        if (not self.num_frames or len(set(self.num_frames)) != len(self.num_frames)
                or any(f not in (124, 243, 362) for f in self.num_frames)):
            raise ValueError("fixed prompt num_frames must select from [124,243,362]")

    def on_train_start(self, model, iteration=0):
        if self.enabled:
            self.prompts = load_fixed_prompts(self.manifest,
                expected_layer=model.h3.cache.text_encoder_layer)[:self.num_prompts]
            if self.run_at_start:
                # EveryN's run_at_start actually fires after update 1. H3
                # must exercise sampling and VAE decoding before any update,
                # also when resuming at a nonzero checkpoint iteration.
                self.every_n_impl(self.trainer, model, {}, {}, None, iteration)
                if dist.is_initialized():
                    dist.barrier()

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        # The startup run above replaces EveryN's special iteration-1 run.
        if self.enabled and iteration % self.every_n == 0:
            super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        started = time.monotonic()
        log.info(f"H3 fixed prompts: start step={iteration}, prompts={len(self.prompts)}, "
                 f"frames={self.num_frames}, decode={self.decode}")
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        root = Path(self.config.job.path_local) / "fixed_prompt" / f"iter_{iteration:09d}"
        root.mkdir(parents=True, exist_ok=True)
        pipeline = H3InferencePipeline(model.net, model.h3,
                                       score_mapper=getattr(model, "score_mapper", None))
        height, width = model.h3.data.height, model.h3.data.width
        if height is None or width is None:
            # Mixed-resolution training still needs one deterministic preview
            # layout. Use the first scheduled layout, which is the 512x320
            # branch in the mixed 3:3:1 experiment.
            schedule_layout = model.h3.data.size_schedule or model.h3.data.resolution_schedule
            if schedule_layout:
                width_text, height_text = schedule_layout[0].split("x")[:2]
                width, height = int(width_text), int(height_text)
            else:
                height, width = 288, 512
        # Checkpoints count completed updates: the boundary checkpoint still
        # contains the PCM-trained student, before any DMD generator update.
        pcm_only = model.h3.pcm.enabled and (
            not (model.h3.dmd.enabled and model.h3.dmd.dmd_loss_weight > 0)
            or (model.h3.pcm.warmup_steps > 0
                and iteration <= model.h3.pcm.warmup_steps))
        if hasattr(getattr(model, "method", None), "dmd_start"):
            pcm_only = pcm_only or iteration <= model.method.dmd_start
        solver = "euler" if pcm_only else model.h3.dmd.rollout_solver
        q = model.h3.pcm.phase_boundaries_q if pcm_only else model.h3.schedule.inference_q
        schedule = pipeline.scheduler.schedule(model.h3.schedule.num_inference_steps, q=q)
        log.info(f"H3 fixed prompt schedule: solver={solver}, q={schedule.q.tolist()}, "
                 f"video_sigmas={schedule.video_sigmas.tolist()}, audio_sigmas={schedule.audio_sigmas.tolist()}")
        pending = []
        records = []
        with evaluation_state(model.net) as device:
            # Every rank must run every round, including ranks with padded slots.
            # rank0-only generation deadlocks an FSDP-sharded student.
            for frames in self.num_frames:
                for index, keep in rank_prompt_slots(len(self.prompts), world, rank):
                    row = self.prompts[index]
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
                                        cache_enabled=False):
                        result = pipeline.generate_latents(
                            row["prompt_embeds"][None], row["text_token_tags"],
                            height=height, width=width,
                            solver=solver, q_schedule=q,
                            num_frames=frames, generator=torch.Generator().manual_seed(self.seed))
                    if keep:
                        result.video_latents = result.video_latents.cpu()
                        result.audio_latents = result.audio_latents.cpu()
                        stem = f"{row['id']}_{frames}f_seed{self.seed}"
                        metadata = {k: v for k, v in row.items() if not isinstance(v, torch.Tensor)}
                        metadata.update(iteration=iteration, rank=rank, world_size=world,
                            num_frames=frames, seed=self.seed, num_forwards=result.num_forwards,
                            solver=solver, q=schedule.q.tolist(),
                            video_sigmas=schedule.video_sigmas.tolist(),
                            audio_sigmas=schedule.audio_sigmas.tolist(),
                            video_timesteps=schedule.video_timesteps.tolist(),
                            audio_timesteps=schedule.audio_timesteps.tolist(),
                            height=result.video_latents.shape[-2] * model.h3.geometry.spatial_compression_ratio,
                            width=result.video_latents.shape[-1] * model.h3.geometry.spatial_compression_ratio,
                            rollout_initial_height=height, rollout_initial_width=width,
                            rollout_high_resolution_from_step=model.h3.crossres.rollout_high_resolution_from_step,
                            fps=model.h3.geometry.fps, weights="student")
                        torch.save({"video_latents": result.video_latents,
                                    "audio_latents": result.audio_latents, "metadata": metadata}, root / f"{stem}.pt")
                        (root / f"{stem}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
                        pending.append((stem, result))
                        records.append(metadata)
                    del result
            if dist.is_initialized():
                dist.barrier()  # Finish all sharded forwards before loading independent VAEs.
            log.info(f"H3 fixed prompts: latent sampling finished in {time.monotonic() - started:.1f}s")
            if self.decode and pending:
                from h3dmd.cli.infer import _save_output
                try:
                    log.info("H3 fixed prompts: loading video/audio VAEs...")
                    pipeline.encoders = load_vaes(model.h3.model.pretrained_path, device)
                    log.info("H3 fixed prompts: VAEs ready, decoding MP4s...")
                    for stem, result in pending:
                        _save_output(root / f"{stem}.mp4", pipeline.decode(result), model.h3.geometry.fps)
                        result.videos, result.audio = None, None
                finally:
                    pipeline.encoders = None
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            (root / f"rank_{rank:05d}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        log.info(f"H3 fixed prompts: rank 0 finished in {time.monotonic() - started:.1f}s, "
                 f"output={root}; waiting for other ranks")

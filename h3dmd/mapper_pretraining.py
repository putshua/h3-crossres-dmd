"""Paired H3 clean-latent validation and resumable frozen-mapper pretraining."""
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from h3dmd.data import _load_tensor_file
from h3dmd.upsampling import (file_sha256, mapper_spec, new_mapper,
                              reconstruction_losses, frames, video)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def select_pairs(rows, config):
    c = config.crossres
    by_split = {name: [r for r in rows if r.get("split") == name] for name in ("train", "val")}
    train_groups = {r["group"] for r in by_split["train"]}
    val_groups = {r["group"] for r in by_split["val"]}
    if train_groups & val_groups:
        raise ValueError("Paired mapper train/heldout source-video groups overlap")
    selected = {}
    generator = torch.Generator().manual_seed(config.training.seed + 7319)
    for split, count in (("train", c.train_samples), ("val", c.validation_samples)):
        values = sorted(by_split[split], key=lambda r: r["id"])
        if len(values) < count:
            raise ValueError(f"Mapper needs {count} {split} pairs, got {len(values)}")
        order = torch.randperm(len(values), generator=generator)[:count].tolist()
        selected[split] = [values[i] for i in order]
    ids = [r["id"] for rows in selected.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Paired mapper records contain duplicate sample IDs")
    return selected


def load_pair(row, config):
    spec = mapper_spec(config)
    provenance = row.get("pairing", {})
    from h3dmd.stride_pairs import SAMPLING as STRIDE_SAMPLING
    supported_sampling = {"nearest_pts_24hz_non_stride_v1"}
    if "cache_member" in row:
        supported_sampling.add(STRIDE_SAMPLING)
    if (provenance.get("sampling") not in supported_sampling or
            provenance.get("fps") != 24 or not provenance.get("same_source_frames") or
            provenance.get("normalization") != spec["normalization"] or
            row.get("low_pixels") != spec["low_pixels"] or
            row.get("high_pixels") != spec["high_pixels"]):
        raise ValueError("Mapper requires an aligned, same-crop, standardized H3 latent pair")
    values = []
    archived = None
    if "cache_member" in row:
        from h3dmd.stride_cache import read_member
        archived = read_member(row['cache_path'], row['cache_member'], expected_id=row['id'], expected_pairing=provenance)
    for key, pixels in (("low_cache_path", spec["low_pixels"]),
                        ("high_cache_path", spec["high_pixels"])):
        if archived is None:
            payload = _load_tensor_file(Path(row[key]))
            if payload.get("alignment_provenance", {}).get("sample_id") != row["id"]:
                raise ValueError("Paired latent payload identity mismatch")
            if payload.get("pairing") != provenance:
                raise ValueError("Paired payload sampling metadata changed")
            tensor = payload["video_latents"].float()
        else:
            tensor = archived['low' if key == 'low_cache_path' else 'high'].float()
        if tensor.ndim != 4 or tensor.shape[0] != 24 or tuple(tensor.shape[-2:]) != tuple(n // 16 for n in pixels):
            raise ValueError("Paired H3 latent shape mismatch")
        expected_t = ((row["num_frames"] - 5) // 17) * 5 + 2
        if tensor.shape[1] != expected_t or (row["num_frames"] - 5) % 17:
            raise ValueError("Paired H3 temporal contract mismatch")
        if not torch.isfinite(tensor).all():
            raise ValueError("Paired H3 latent contains non-finite values")
        values.append(tensor)
    return tuple(values)


class LazyPairs:
    """Keep selected metadata in RAM; load only the pair currently being used."""
    def __init__(self, rows, config):
        self.rows, self.config = rows, config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return load_pair(self.rows[index], self.config)


def training_crop(pairs, config, step, rank):
    rng = torch.Generator().manual_seed(config.training.seed + step * 100003 + rank * 9973)
    index = int(torch.randint(len(pairs), (), generator=rng))
    low, high = pairs[index]
    length = min(config.crossres.crop_latent_frames, low.shape[1])
    start = int(torch.randint(low.shape[1] - length + 1, (), generator=rng))
    low, high = low[:, start:start + length], high[:, start:start + length]
    if bool(torch.randint(2, (), generator=rng)):
        low, high = low.flip(-1), high.flip(-1)
    return low[None], high[None]


@torch.no_grad()
def evaluate(model, pairs, device):
    totals = torch.zeros(7, dtype=torch.float64, device=device)
    model.eval()
    for low, high in pairs:
        low, high = low[None].to(device), high[None].to(device)
        # Full clips, not training crops: heldout includes temporal endpoints.
        estimate = model(low)
        reference = video(torch.nn.functional.interpolate(frames(low), size=high.shape[-2:],
                           mode="bicubic", align_corners=False), 1, low.shape[2])
        _, predicted = reconstruction_losses(estimate, high)
        _, baseline = reconstruction_losses(reference, high)
        names = ("mse", "spatial_mse", "temporal_mse")
        totals[:3] += torch.stack([predicted[k] for k in names]).double()
        totals[3:6] += torch.stack([baseline[k] for k in names]).double()
        totals[6] += 1
    if dist.is_initialized():
        dist.all_reduce(totals)
    numbers = (totals[:6] / totals[6]).tolist()
    result = {"learned_" + k: numbers[i] for i, k in enumerate(names)}
    result.update({"bicubic_" + k: numbers[i + 3] for i, k in enumerate(names)})
    result.update({k + "_ratio": numbers[i] / max(numbers[i + 3], 1e-12) for i, k in enumerate(names)})
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError("Non-finite heldout mapper metrics")
    return result


def accepted(report, config):
    return (report["mse_ratio"] <= config.crossres.mse_ratio_limit and
            report["temporal_mse_ratio"] <= config.crossres.temporal_ratio_limit)


def pretrain(config, output, device, rank=0, world=1):
    """A separate optimizer/problem; no H3 transformer, prompt or audio forwards."""
    c = config.crossres
    root = Path(output).resolve()
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    rows = [json.loads(l) for l in Path(c.paired_manifest).read_text().splitlines() if l.strip()]
    selections = select_pairs(rows, config)
    if min(len(selections["train"]), len(selections["val"])) < world:
        raise ValueError("Every mapper rank needs train and heldout records")
    signature = dict(spec=mapper_spec(config), settings=asdict(c), seed=config.training.seed,
                     manifest_sha256=file_sha256(c.paired_manifest), world_size=world)
    for key in ("mapper_checkpoint", "mapper_sha256"):
        signature["settings"].pop(key)
    if (root / "frozen.pt").is_file():
        from dataclasses import replace
        from h3dmd.upsampling import load_frozen_mapper
        artifact = torch.load(root / "frozen.pt", map_location="cpu", weights_only=True)
        if artifact.get("signature") != signature:
            raise ValueError("Completed mapper's training configuration or paired dataset changed")
        load_frozen_mapper(replace(config, crossres=replace(c, mapper_checkpoint=str(root / "frozen.pt"))))
        return
    selected_ids = {s: [r["id"] for r in values] for s, values in selections.items()}
    if rank == 0:
        (root / "sample_manifest.json").write_text(json.dumps(selected_ids, indent=2))
    train = LazyPairs(selections["train"][rank::world], config)
    heldout = LazyPairs(selections["val"][rank::world], config)
    module = new_mapper(config).to(device)
    from rcm.utils.torch_fused_adamw_dtensor import FusedAdamW
    optimizer = (FusedAdamW(module.parameters(), lr=c.learning_rate, betas=(.9, .99),
                           weight_decay=0, master_weights=True) if device.type == "cuda" else
                 torch.optim.AdamW(module.parameters(), lr=c.learning_rate, betas=(.9, .99), weight_decay=0))
    start, best, best_step, best_model, best_report = 0, float("inf"), 0, None, None
    latest = root / "latest.pt"
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise ValueError("Mapper resume configuration, data or world size changed")
        module.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best, best_step = saved["step"], saved["best_score"], saved["best_step"]
        best_model, best_report = saved["best_model"], saved["best_report"]
    wrapped = (DistributedDataParallel(module, device_ids=[device.index] if device.type == "cuda" else None,
                                      broadcast_buffers=False) if world > 1 else module)
    writer = None
    if rank == 0 and config.logging.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(root / "tensorboard"))

    def log(record):
        if rank == 0:
            with (root / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if writer:
                for k, v in record.items():
                    if k != "step" and isinstance(v, (int, float)):
                        writer.add_scalar("mapper/" + k, v, record["step"])
                writer.flush()
            print(json.dumps(record), flush=True)

    if not start:
        log(dict(event="validation", step=0, **evaluate(module, heldout, device)))
    step = start
    try:
        for step in range(start + 1, c.max_pretrain_steps + 1):
            wrapped.train()
            low, high = training_crop(train, config, step, rank)
            loss, parts = reconstruction_losses(wrapped(low.to(device)), high.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite mapper training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            log(dict(event="train", step=step, loss=float(loss.detach()), grad_norm=float(norm),
                     **{k: float(v.detach()) for k, v in parts.items()}))
            if step % c.validation_every == 0 or step in (c.pretrain_steps, c.max_pretrain_steps):
                report = evaluate(module, heldout, device)
                log(dict(event="validation", step=step, **report))
                # Track best accepted model, never discard it for a lower-MSE
                # checkpoint that violates the temporal acceptance criterion.
                if accepted(report, config) and report["mse_ratio"] < best:
                    best, best_step, best_report = report["mse_ratio"], step, report
                    best_model = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                if rank == 0:
                    atomic_save(dict(signature=signature, model=module.state_dict(),
                                     optimizer=optimizer.state_dict(), step=step,
                                     best_score=best, best_step=best_step, best_model=best_model,
                                     best_report=best_report), latest)
                if dist.is_initialized():
                    dist.barrier()
                if step >= c.pretrain_steps and best_model is not None:
                    break
        if best_model is None:
            if rank == 0:
                (root / "blocked.json").write_text(json.dumps(dict(step=step, reason="heldout gate failed")))
            raise RuntimeError("Mapper failed heldout acceptance; DMD must not start")
        if rank == 0:
            atomic_save(dict(spec=mapper_spec(config), model=best_model, signature=signature,
                             accepted=True, report=best_report, best_step=best_step,
                             pretrain_steps_completed=step), root / "frozen.pt")
            (root / "COMPLETE.json").write_text(json.dumps(dict(
                sha256=file_sha256(root / "frozen.pt"), step=step, report=best_report), indent=2))
        if dist.is_initialized():
            dist.barrier()
    finally:
        if writer:
            writer.close()

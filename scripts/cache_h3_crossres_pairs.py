"""Prepare and encode matched 320p/480p cache; never overwrite old AV caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import zipfile

import numpy as np
import torch

from h3dmd.mapper_pretraining import atomic_save
from h3dmd.paired_cache import SAMPLING, NORMALIZATION, group_split, sample_pair, PairedVideoEncoder


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json_lines(records, path):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "input.jsonl").exists():
        raise FileExistsError("Paired input snapshot already exists; use it without --prepare")
    original = {r["id"]: r for r in rows(Path(args.manifest).parent / "input.jsonl")}
    pool = {"train": [], "val": []}
    for row in rows(args.manifest):
        if (row.get("video_sampling") != SAMPLING or
                (row.get("width"), row.get("height")) != (512, 320)):
            raise ValueError("Paired-cache preparation requires the validated non-stride 320p manifest")
        source = dict(original[row["id"]])
        source["group"], source["split"] = group_split(source["source_video"])
        source["num_frames"] = row["num_frames"]
        source["original_low_cache_path"] = row["cache_path"]
        pool[source["split"]].append(source)
    rng = random.Random(args.seed)
    selected = []
    for split, count in (("train", args.train_samples), ("val", args.val_samples)):
        candidates = sorted(pool[split], key=lambda r: r["id"])
        rng.shuffle(candidates)
        if len(candidates) < count:
            raise ValueError(f"Not enough distinct-source {split} records")
        selected.extend(candidates[:count])
    atomic_json_lines(selected, root / "input.jsonl")
    (root / "plan.json").write_text(json.dumps(dict(
        source_manifest=str(Path(args.manifest).resolve()), seed=args.seed,
        train_samples=args.train_samples, val_samples=args.val_samples,
        low_pixels=[320, 512], high_pixels=[480, 768], sampling=SAMPLING), indent=2))
    print(f"Prepared {len(selected)} immutable paired-cache inputs in {root}")


def finalize(root, world):
    inputs = rows(root / "input.jsonl")
    expected = {r["id"] for r in inputs}
    if not all((root / "done" / f"rank{rank}.json").is_file() for rank in range(world)):
        raise RuntimeError("Some paired-cache workers have not completed")
    outputs = [r for rank in range(world) for r in rows(root / "partials" / f"rank{rank}.jsonl")]
    if len(outputs) != len(expected) or {r["id"] for r in outputs} != expected:
        raise RuntimeError("Missing, duplicate or unexpected paired-cache IDs")
    for r in outputs:
        for key in ("low_cache_path", "high_cache_path"):
            if not Path(r[key]).is_file():
                raise FileNotFoundError(r[key])
    atomic_json_lines(sorted(outputs, key=lambda r: r["id"]), root / "pairs.jsonl")
    (root / "COMPLETE.json").write_text(json.dumps(dict(count=len(outputs), sampling=SAMPLING)))
    print(f"Committed {len(outputs)} matched pairs")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest", default="data/h3_av50k_512x320_non_stride_v1/manifest.jsonl")
    p.add_argument("--output", type=Path, default=Path("data/h3_crossres_320_480_v1"))
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--train-samples", type=int, default=1024)
    p.add_argument("--val-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vae-seed", type=int, default=42)
    p.add_argument("--h3-root", default="/mnt/data/butong/mmh3/MiniMax-H3/FL2VA")
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--rank-offset", type=int, default=0)
    a = p.parse_args()
    if a.prepare:
        prepare(a)
        return
    root = a.output.resolve()
    if a.finalize:
        finalize(root, a.world_size)
        return
    local = int(os.environ.get("LOCAL_RANK", "0"))
    rank = a.rank_offset + local
    if not 0 <= rank < a.world_size:
        raise ValueError("Paired cache rank outside global world size")
    for directory in ("partials", "done", "media"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    protocol = dict(world_size=a.world_size, vae_seed=a.vae_seed,
                    h3_root=str(Path(a.h3_root).resolve()), sampling=SAMPLING,
                    input_sha256=hashlib.sha256((root / "input.jsonl").read_bytes()).hexdigest(),
                    encoder_code_sha256=hashlib.sha256(
                        (Path(__file__).resolve().parents[1] / "h3dmd/paired_cache.py").read_bytes()).hexdigest(),
                    worker_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    vae_config_sha256=hashlib.sha256(
                        (Path(a.h3_root) / "video_vae/config.json").read_bytes()).hexdigest())
    protocol_path = root / "partials" / f"rank{rank}.protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Cache worker input, partition or VAE configuration changed on resume")
    else:
        protocol_path.write_text(json.dumps(protocol))
    part = root / "partials" / f"rank{rank}.jsonl"
    completed = {r["id"] for r in rows(part)} if part.is_file() else set()
    source = [r for i, r in enumerate(rows(root / "input.jsonl"))
              if i % a.world_size == rank and r["id"] not in completed]
    torch.set_num_threads(2)
    started = time.monotonic()
    print(f"rank={rank} pending={len(source)} resumed={len(completed)} loading video VAE", flush=True)
    encoder = PairedVideoEncoder(a.h3_root, f"cuda:{local}")
    with part.open("a") as handle:
        for index, row in enumerate(source, 1):
            with tempfile.TemporaryDirectory(prefix="h3-crossres-source-") as temp:
                path = Path(row["source_video"])
                if not path.is_file():
                    path = Path(temp) / "source.mp4"
                    with zipfile.ZipFile(row["archive"]) as archive, archive.open(row["member"]) as src, path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                low, high, chosen, targets = sample_pair(str(path), float(row["source_time_range"][0]), row["num_frames"])
            pairing = dict(sampling=SAMPLING, fps=24, same_source_frames=True,
                           normalization=NORMALIZATION, source_video=row["source_video"],
                           vae_seed=a.vae_seed, posterior="sample_fp16_roundtrip",
                           source_time_range=row["source_time_range"],
                           canonical_source=row.get("canonical_source", row["source_video"]),
                           original_source_video=row.get("original_source_video", row["source_video"]),
                           original_time_range=row.get("original_time_range", row["source_time_range"]),
                           chosen_pts_sha256=hashlib.sha256(chosen.tobytes()).hexdigest(),
                           max_pts_error_ms=float(np.max(np.abs(chosen - targets)) * 1000))
            directory = root / "media" / row["id"][:2]
            directory.mkdir(parents=True, exist_ok=True)
            destinations = {}
            for tag, pixels in (("low", low), ("high", high)):
                destination = directory / f"{row['id']}_{tag}.pt"
                atomic_save(dict(video_latents=encoder.encode(pixels, a.vae_seed), pairing=pairing,
                                 alignment_provenance={"sample_id": row["id"]}), destination)
                destinations[tag + "_cache_path"] = str(destination)
            record = dict(id=row["id"], group=row["group"], split=row["split"],
                          num_frames=row["num_frames"], low_pixels=[320, 512], high_pixels=[480, 768],
                          pairing=pairing, **destinations)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            progress = dict(rank=rank, completed=len(completed) + index,
                            total=len(completed) + len(source), updated=time.time(),
                            elapsed_seconds=time.monotonic() - started, last_id=row["id"])
            temporary = root / "partials" / f"rank{rank}.progress.tmp.json"
            temporary.write_text(json.dumps(progress))
            temporary.replace(root / "partials" / f"rank{rank}.progress.json")
            print(f"rank={rank} encoded {row['id']}", flush=True)
    (root / "done" / f"rank{rank}.json").write_text(json.dumps(dict(rank=rank)))


if __name__ == "__main__":
    main()

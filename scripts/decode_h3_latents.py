"""Decode saved H3 callback latents on independent local GPUs (VAE only).

Example: torchrun --standalone --nproc_per_node=4 -m scripts.decode_h3_latents \
    --input-dir outputs/run/fixed_prompt/iter_000000500 --output-dir outputs/run/decoded
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained-path", default="/mnt/data/butong/mmh3/MiniMax-H3/FL2VA")
    args = parser.parse_args()
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    def log(message):
        print(f"{datetime.datetime.now().isoformat(timespec='seconds')} rank={rank} {message}", flush=True)

    source, destination = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    if source == destination:
        parser.error("output-dir must differ from input-dir to preserve training callback outputs")
    files = sorted(source.glob("*.pt"))[rank::world]
    if not files:
        log("No assigned PT files")
        return
    destination.mkdir(parents=True, exist_ok=True)
    log(f"Loading dependencies; {len(files)} files assigned, GPU {local_rank}")
    import torch
    from h3dmd.config import H3Config
    from h3dmd.decoders import load_h3_vaes
    from h3dmd.inference import H3InferenceOutput, H3InferencePipeline
    from h3dmd.cli.infer import _save_output

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    started = time.monotonic()
    log("Loading video/audio VAEs only")
    # No process group: each rank decodes its own clips with standalone VAEs.
    pipeline = H3InferencePipeline(None, H3Config(), load_h3_vaes(args.pretrained_path, device))
    log(f"VAEs ready after {time.monotonic() - started:.1f}s")
    records = []
    for path in files:
        target = destination / f"{path.stem}.mp4"
        record = {"source": str(path), "output": str(target), "rank": rank}
        if target.is_file() and target.stat().st_size > 0:
            record["status"] = "existing"
            records.append(record)
            continue
        temporary = destination / f"{path.stem}.tmp.mp4"
        sample_started = time.monotonic()
        log(f"Decoding {path.name}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            metadata = payload.get("metadata", {})
            video, audio = payload["video_latents"], payload["audio_latents"]
            if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
                raise ValueError("Nonfinite saved latents")
            result = pipeline.decode(H3InferenceOutput(video, audio))
            if not torch.isfinite(result.videos).all() or not torch.isfinite(result.audio).all():
                raise ValueError("Nonfinite decoded video/audio")
            fps = int(metadata.get("fps", 24))
            _save_output(temporary, result, fps)
            temporary.replace(target)
            record.update(status="ok", video_shape=list(result.videos.shape),
                audio_shape=list(result.audio.shape), sampling_rate=result.sampling_rate,
                fps=fps, seconds=time.monotonic() - sample_started,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30)
            (destination / f"{path.stem}.json").write_text(
                json.dumps({"metadata": metadata, "decode": record}, ensure_ascii=False, indent=2) + "\n")
            del result, payload, video, audio
            log(f"Saved {target.name}, {record['seconds']:.1f}s")
        except Exception:
            record.update(status="error", traceback=traceback.format_exc())
            log(record["traceback"])
            temporary.unlink(missing_ok=True)
        records.append(record)
        (destination / f"decode_rank_{rank:05d}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    if any(r["status"] == "error" for r in records):
        raise RuntimeError(f"rank {rank}: some files failed; see decode report")
    log(f"Finished {len(records)} clips")


if __name__ == "__main__":
    main()

"""Inventory bucketed source manifests and freeze all eligible H3 video pairs.

No prompt/audio cache is needed. Original source/time coordinates are preferred
over already resized clips, so both resolutions share one spatial crop.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from h3dmd.paired_cache import SAMPLING, group_split
from scripts.cache_h3_crossres_pairs import atomic_json_lines, rows

FRAMES = {5: 124, 10: 243, 15: 362}


def normalize(row, origin):
    source = row.get("original_source_video") or row.get("source_video")
    interval = (row.get("original_time_range") if row.get("original_source_video") else
                row.get("source_time_range") or row.get("time_range"))
    count = row.get("target_frames") or row.get("num_frames") or FRAMES.get(row.get("bucket_seconds"))
    if not source or not interval or count not in FRAMES.values():
        raise ValueError("Missing source/time range or unsupported 5/10/15s frame bucket")
    source = Path(source).expanduser()
    if not source.is_absolute():
        raise ValueError("Original source path must be absolute; resolve it before freezing inputs")
    source = str(source.resolve())
    start, end = map(float, interval)
    if not all(map(math.isfinite, (start, end))) or start < 0 or end <= start:
        raise ValueError("Invalid source time range")
    # A time-based identity deduplicates copied manifests and duplicate clip IDs.
    identity = json.dumps([source, round(start, 6), int(count)], separators=(",", ":"))
    sample_id = hashlib.sha256(identity.encode()).hexdigest()[:32]
    canonical = row.get("source_key") or source
    group, split = group_split(canonical)
    return dict(id=sample_id, source_video=source, source_time_range=[start, end],
                num_frames=int(count), group=group, split=split,
                archive=row.get("archive"), member=row.get("member"),
                original_id=row.get("id") or row.get("clip_id"),
                canonical_source=canonical, original_source_video=source,
                original_time_range=[start, end], rendered_video=row.get("video"),
                source_manifest=str(origin.resolve()), provider=row.get("provider") or row.get("dataset"))


def prepare(manifests, output, check_files=True):
    output = Path(output).resolve()
    if (output / "input.jsonl").exists():
        raise FileExistsError("Immutable input already exists; choose a new output version")
    accepted, rejected, counts, hashes, availability = {}, [], {}, {}, {}
    def exists(path):
        if path not in availability:
            availability[path] = Path(path).is_file()
        return availability[path]
    for manifest in map(Path, manifests):
        numbers = Counter()
        hashes[str(manifest.resolve())] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        for index, row in enumerate(rows(manifest)):
            numbers["rows"] += 1
            try:
                if row.get("accepted") is False or row.get("repair_issues"):
                    raise ValueError("Explicitly rejected or unrepaired source")
                # The caller supplies the accepted release, not raw candidates.
                # Keep slow-motion clips too: spatial mapping does not require
                # another motion-policy filter on top of the release's review.
                record = normalize(row, manifest)
                record["review"] = row.get("review")
                if check_files and not exists(record["source_video"]):
                    if not record["archive"] or not record["member"] or not exists(record["archive"]):
                        rendered = record["rendered_video"]
                        if not rendered or not Path(rendered).is_absolute() or not exists(rendered):
                            raise ValueError("Neither original source nor rendered clip is available")
                        record["source_video"] = str(Path(rendered).resolve())
                        record["source_time_range"] = [0, record["num_frames"] / 24]
                        record["media_origin"] = "reviewed_rendered_clip"
                record.setdefault("media_origin", "original_source")
                if record["id"] in accepted:
                    numbers["duplicates"] += 1
                    continue
                accepted[record["id"]] = record
                numbers["eligible"] += 1
            except (ValueError, TypeError, KeyError) as error:
                numbers["rejected"] += 1
                rejected.append(dict(manifest=str(manifest), line=index + 1,
                                     original_id=row.get("id") or row.get("clip_id"), reason=str(error)))
        counts[str(manifest.resolve())] = dict(numbers)
    records = sorted(accepted.values(), key=lambda row: row["id"])
    if not records:
        raise ValueError("No eligible source clips")
    durations = Counter(row["num_frames"] for row in records)
    # Both outputs currently store standardized FP32 latents; excludes pt overhead.
    payload_bytes = sum(24 * (((row["num_frames"] - 5) // 17) * 5 + 2) *
                        (20 * 32 + 30 * 48) * 4 for row in records)
    report = dict(source_manifests=counts, source_sha256=hashes, total_pairs=len(records),
                  total_latent_files=2 * len(records), rejected=len(rejected),
                  unique_source_videos=len({row["canonical_source"] for row in records}),
                  media_origins=dict(Counter(row["media_origin"] for row in records)),
                  frame_buckets=dict(durations), splits=dict(Counter(row["split"] for row in records)),
                  low_pixels=[320, 512], high_pixels=[480, 768], sampling=SAMPLING,
                  estimated_fp32_payload_GiB=payload_bytes / 1024**3,
                  file_existence_checked=check_files, policy="all_unique_eligible_bucketed_clips")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json_lines(records, output / "input.jsonl")
    atomic_json_lines(rejected, output / "rejected.jsonl")
    (output / "plan.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", action="append", required=True,
                        help="Repeat for all source manifests belonging to the intended release")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-file-check", action="store_true",
                        help="Inventory only; source existence remains unverified")
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.output, not args.skip_file_check),
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

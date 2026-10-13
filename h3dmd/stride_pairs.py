"""Refiner-compatible fixed-stride planning for long H3 spatial latent pairs.

Original source frame indices determine the pair. Output24Hz does not imply
original playback speed. This recipe never pads an incomplete source window.
"""
from __future__ import annotations

import math

SAMPLING = "refiner_fixed_stride_24fps_long_pair_v1"
FPS_STRIDE = {24: 1, 25: 1, 30: 1, 48: 2, 50: 2, 60: 2,
              72: 3, 96: 4, 100: 4, 120: 5}
FRAMES = {5: 124, 10: 243, 15: 362}
RESERVE_FRAMES = 2


def stride_for_fps(fps):
    if fps is None or not math.isfinite(fps) or fps <= 0:
        raise ValueError("review_invalid_fps")
    matches = [(abs(fps / nominal - 1), nominal, stride)
               for nominal, stride in FPS_STRIDE.items()
               if abs(fps / nominal - 1) <= .005]
    if not matches:
        raise ValueError("review_unsupported_fps")
    _, nominal, stride = min(matches)
    return dict(nominal_fps=nominal, source_stride=stride,
                sampled_fps=fps / stride, output_fps=24,
                speed_factor=24 * stride / fps)


def window_starts(native_frames, stride, payload_frames):
    """Same refiner window rule:2 reserve frames; advance by encoded payload."""
    if native_frames < 0 or stride < 1 or payload_frames not in FRAMES.values():
        raise ValueError("Invalid H3 long-pair frame plan")
    required = (payload_frames + RESERVE_FRAMES - 1) * stride + 1
    return range(0, max(0, native_frames - required + 1), payload_frames * stride)


def source_windows(row, bucket):
    """Metadata candidates; real workers must validate decoded PTS/geometry.

    Blacklist intervals are in original source seconds, before any speed change.
    Candidate capacities must not be reported as verified usable cache counts.
    """
    if row.get("status") != "ok" or row.get("blacklist_policy") == "all_blocked":
        return
    if row.get("width", 0) < 1920 or row.get("height", 0) < 1080:
        return
    sampling = stride_for_fps(row.get("fps"))
    fps, stride = row["fps"], sampling["source_stride"]
    duration = row.get("duration")
    if duration is None or not math.isfinite(duration) or duration <= 0:
        raise ValueError("review_invalid_duration")
    estimated = row.get("native_frames_estimate")
    native_frames = (int(estimated) if estimated is not None and estimated > 0
                     else math.floor(duration * fps))
    payload = FRAMES[bucket]
    guard = max(.0625, 1 / fps)
    intervals = row.get("blocked_intervals", [])
    if row.get("blacklist_policy") == "partial" and not intervals:
        raise ValueError("Partial blacklist source lacks original-time intervals")
    for start in window_starts(native_frames, stride, payload):
        first_seconds = start / fps
        last_frame = start + (payload + RESERVE_FRAMES - 1) * stride
        end_seconds = (last_frame + 1) / fps
        if any(first_seconds < end + guard and end_seconds > begin - guard
               for begin, end in intervals):
            continue
        yield dict(start_native_frame=start, source_stride=stride,
                   num_frames=payload, source_decode_frames=payload + RESERVE_FRAMES,
                   source_interval=[first_seconds, end_seconds],
                   source_fps=fps, speed_factor=sampling["speed_factor"],
                   blacklist_guard_seconds=guard)

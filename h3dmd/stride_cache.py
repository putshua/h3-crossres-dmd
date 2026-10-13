"""Long fixed-stride H3 pairs; refiner-compatible pixels, verified tar storage.

Native-plane transport/color options are ported from h3_refiner recut_production
(2026-10-09). No runtime import of that project's business code is required.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tarfile
import tempfile
import time

import numpy as np

from h3dmd.stride_pairs import SAMPLING, stride_for_fps


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_streams', '-of', 'json', str(path)],
                            capture_output=True, check=True, timeout=60)
    return json.loads(result.stdout.decode('utf-8', errors='replace'))['streams'][0]


def native_planes(frame):
    components = frame.format.components
    if any(c.bits != 8 for c in components) and not frame.format.is_planar:
        raise ValueError('review_unsupported_packed_high_depth')
    for number, plane in enumerate(frame.planes):
        row_bytes = sum(c.width * ((c.bits + 7) // 8) for c in components if c.plane == number)
        if frame.format.name in {'bgr0', 'rgb0', '0bgr', '0rgb'}:
            row_bytes = frame.width * 4
        if not row_bytes or row_bytes > plane.line_size:
            raise ValueError('review_unsupported_pixel_layout')
        yield np.frombuffer(plane, dtype=np.uint8).reshape(plane.height, plane.line_size)[:, :row_bytes].tobytes()


def sample_stride_pair(row, work):
    """Decode exact native indices; preserve approved native->bgr0 FFV1 route."""
    import av
    from PIL import Image
    from fractions import Fraction
    source = Path(row['source'])
    before = source.stat()
    info = probe(source)
    if int(info['width']) < 1920 or int(info['height']) < 1080:
        raise ValueError('review_source_resolution')
    fps = float(Fraction(info['avg_frame_rate']))
    if abs(fps / row['source_fps'] - 1) > .005:
        raise ValueError('review_fps_metadata_changed')
    if stride_for_fps(fps)['source_stride'] != row['source_stride']:
        raise ValueError('review_stride_mismatch')
    if info.get('sample_aspect_ratio', '1:1') not in {'1:1', '0:1', 'N/A'}:
        raise ValueError('review_non_square_pixels')
    if any(abs(float(s.get('rotation', 0))) > .001 for s in info.get('side_data_list', [])):
        raise ValueError('review_rotation')
    special = (info.get('color_transfer') in {'smpte2084', 'arib-std-b67'} or
               info.get('color_primaries') == 'bt2020' or
               info.get('color_space') in {'bt2020nc', 'bt2020c'})
    start, stride, count = row['start_native_frame'], row['source_stride'], row['num_frames'] + 2
    last = start + (count - 1) * stride
    pts, checked = [], []
    first, selected_count, guard_id3 = None, 0, False
    raw, master = Path(work) / 'native.raw', Path(work) / 'hr.mkv'
    origin = None
    deadline = time.monotonic() + 600
    with av.open(str(source), metadata_errors='replace') as container, raw.open('wb') as raw_file:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = 2
        for n, frame in enumerate(container.decode(stream)):
            if time.monotonic() > deadline:
                raise TimeoutError('source decode exceeded 600s')
            if frame.pts is None:
                raise ValueError('review_missing_pts')
            timestamp = float(frame.pts * stream.time_base)
            if origin is None:
                origin = timestamp
            if n >= start:
                checked.append(timestamp)
            if n >= start and (n - start) % stride == 0:
                if frame.interlaced_frame:
                    raise ValueError('review_interlaced')
                if (frame.width, frame.height) != (int(info['width']), int(info['height'])):
                    raise ValueError('review_dynamic_geometry')
                if first is None:
                    first = frame
                    guard_id3 = memoryview(first.planes[0])[:3].tobytes() == b'ID3'
                    if guard_id3:
                        for plane in native_planes(first):
                            raw_file.write(bytes(len(plane)))
                if frame.format.name != first.format.name:
                    raise ValueError('review_dynamic_pixel_format')
                for plane in native_planes(frame):
                    raw_file.write(plane)
                selected_count += 1
                pts.append(timestamp)
            if n == last:
                break
    if selected_count != count:
        raise ValueError('review_incomplete_window')
    deltas = np.diff(checked)
    period = float(statistics.median(deltas))
    if period <= 0 or np.min(deltas) <= 0 or np.max(np.abs(deltas / period - 1)) > .05:
        raise ValueError('review_variable_frame_rate')
    if abs((1 / period) / fps - 1) > .005:
        raise ValueError('review_measured_fps')
    interval = [pts[0] - origin, pts[-1] - origin + period]
    guard = max(.0625, 1 / fps)
    if any(interval[0] < b + guard and interval[1] > a - guard for a, b in row['blocked_intervals']):
        raise ValueError('review_blacklist_overlap')
    special = special or any(c.bits > 8 for c in first.format.components)
    color = []
    for key, flag in [('color_range', '-color_range'), ('color_space', '-colorspace'),
                      ('color_primaries', '-color_primaries'), ('color_transfer', '-color_trc'),
                      ('chroma_location', '-chroma_sample_location')]:
        if info.get(key) not in {None, '', 'unknown', 'unspecified', 'reserved'}:
            color += [flag, str(info[key])]
    trim = 'trim=start_frame=1,setpts=PTS-STARTPTS,' if guard_id3 else ''
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror', '-y',
               '-threads', '2', '-f', 'rawvideo', '-pixel_format', first.format.name,
               '-video_size', f'{first.width}x{first.height}', '-framerate', '24', *color,
               '-i', str(raw), '-vf', trim + 'scale=768:480:force_original_aspect_ratio=increase,crop=768:480:(iw-ow)/2:(ih-oh)/2,setsar=1',
               '-filter_threads', '2', '-frames:v', str(count), '-map_metadata', '-1',
               '-an', '-sn', '-dn', '-threads', '2', '-c:v', 'ffv1', '-level', '3',
               '-slicecrc', '1', '-pix_fmt', 'bgr0', '-colorspace', 'bt709',
               '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'pc',
               '-vsync', 'cfr', '-r', '24', str(master)]
    subprocess.run(command, check=True, capture_output=True, timeout=600)
    raw.unlink()
    high = []
    with av.open(str(master)) as container:
        for f in container.decode(video=0):
            high.append(f.to_ndarray(format='rgb24'))
    if len(high) != count:
        raise ValueError('review_master_count')
    high = np.stack(high[:row['num_frames']])
    low = np.stack([np.asarray(Image.fromarray(f).resize((512, 320), Image.Resampling.BICUBIC)) for f in high])
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('source_changed_during_decode')
    metadata = dict(sampling=SAMPLING, fps=24, same_source_frames=True,
                    normalization='h3_standardized_video_v1', source=str(source),
                    source_stat={'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns},
                    source_pts=pts, source_pts_origin=origin, source_interval=interval,
                    source_frame_indices=list(range(start, last + 1, stride)),
                    source_stride=stride, source_fps=fps, max_relative_pts_jitter=float(np.max(np.abs(deltas / period - 1))),
                    color_policy='special_legacy_old_artifact_bgr0_v1' if special else 'ordinary_no_extra_color_processing',
                    pixel_recipe='native_ffmpeg_bgr0_ffv1_hr480_pillow_bicubic_lr320',
                    posterior='sample_fp16_roundtrip', vae_seed=42,
                    reserve_frames=2, rawvideo_id3_transport_guard=guard_id3)
    return low, high, metadata


def read_member(archive, record, *, expected_id=None, expected_pairing=None):
    from safetensors.torch import load
    with open(archive, 'rb') as f:
        f.seek(record['offset'])
        data = f.read(record['size'])
    if hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('paired tar member SHA256 mismatch')
    if expected_id is not None:
        header_bytes = int.from_bytes(data[:8], 'little')
        metadata = json.loads(data[8:8+header_bytes]).get('__metadata__', {})
        if metadata.get('sample_id') != expected_id or json.loads(metadata.get('pairing', '{}')) != expected_pairing:
            raise ValueError('paired tar identity or sampling metadata mismatch')
    values = load(data)
    if set(values) != {'low', 'high'}:
        raise ValueError('paired tar tensor keys')
    return values


class PairArchive:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / 'pairs.tar'
        self.file = self.path.open('wb')
        self.tar = tarfile.open(fileobj=self.file, mode='w')
        self.rows = []

    def add(self, row, low, high, pairing):
        import torch
        from safetensors.torch import save, load
        data = save({'low': low.contiguous(), 'high': high.contiguous()},
                    metadata={'sample_id': row['id'], 'pairing': json.dumps(pairing, sort_keys=True)})
        check = load(data)
        if not torch.equal(check['low'], low) or not torch.equal(check['high'], high):
            raise ValueError('safetensors roundtrip changed pair')
        member = tarfile.TarInfo(row['id'] + '/pair.safetensors')
        member.size = len(data)
        header_offset = self.tar.offset
        self.tar.addfile(member, io.BytesIO(data))
        record = dict(row, pairing=pairing, low_pixels=[320, 512], high_pixels=[480, 768],
                      cache_path=str(self.path), cache_member={'name': member.name, 'offset': header_offset + 512,
                          'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        self.file.flush()
        actual = read_member(self.path, record['cache_member'])
        if not torch.equal(actual['low'], low) or not torch.equal(actual['high'], high):
            raise ValueError('actual tar roundtrip changed pair')
        self.rows.append(record)

    def close(self, failures, task_sha):
        self.tar.close()
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        manifest = self.directory / 'manifest.jsonl'
        with manifest.open('w') as f:
            for row in self.rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        return dict(archive=str(self.path), archive_sha256=sha_file(self.path),
                    manifest=str(manifest), manifest_sha256=sha_file(manifest),
                    count=len(self.rows), failures=failures, task_sha256=task_sha,
                    verified_member_reads=len(self.rows), committed_at=time.time())


def verify_commit(commit, task_path, deep=False):
    if commit['task_sha256'] != sha_file(task_path):
        raise ValueError('task changed after commit')
    for field in ['archive', 'manifest']:
        if not Path(commit[field]).is_file() or sha_file(commit[field]) != commit[field + '_sha256']:
            raise ValueError('committed shard missing or corrupt')
    if deep:
        with open(commit['manifest']) as f:
            for line in f:
                row = json.loads(line)
                read_member(commit['archive'], row['cache_member'])
    return commit

"""Same-source-frame, same-crop H3 VAE pairs for spatial mapper supervision."""
import hashlib
import json
from pathlib import Path

import numpy as np

SAMPLING = "nearest_pts_24hz_non_stride_v1"
NORMALIZATION = "h3_standardized_video_v1"


def group_split(source):
    key = hashlib.sha256(str(source).encode()).hexdigest()
    return key, "val" if int(key[:8], 16) % 20 == 0 else "train"


def fit(rgb, width, height):
    import cv2
    h, w = rgb.shape[:2]
    if w / h > width / height:
        count = max(2, int(round(h * width / height)))
        offset = (w - count) // 2
        rgb = rgb[:, offset:offset + count]
    elif w / h < width / height:
        count = max(2, int(round(w * height / width)))
        offset = (h - count) // 2
        rgb = rgb[offset:offset + count]
    return np.ascontiguousarray(cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LANCZOS4))


def sample_pair(path, start, count, low_size=(512, 320), high_size=(768, 480)):
    """Decode once; every selected source frame feeds both spatial encodes.

    Matches the existing non-stride sampler's target grid and nearest-PTS
    tie-breaking. This is time-based resampling, never an integer frame stride.
    """
    import av
    low, high, chosen = [], [], []
    targets = start + np.arange(count, dtype=np.float64) / 24
    index, previous = 0, None

    def append(frame, timestamp, rotation):
        rgb = frame.to_ndarray(format="rgb24")
        if rotation % 360 in (90, 180, 270):
            rgb = np.rot90(rgb, {90: 3, 180: 2, 270: 1}[rotation % 360])
        low.append(fit(rgb, *low_size))
        high.append(fit(rgb, *high_size))
        chosen.append(timestamp)

    with av.open(path) as container:
        stream = container.streams.video[0]
        rotation = int((stream.metadata or {}).get("rotate", 0) or 0)
        container.seek(int(max(0, start - 2) / float(stream.time_base)), stream=stream,
                       backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            current = timestamp, frame
            if previous is None:
                previous = current
            while index < count and timestamp >= targets[index]:
                best = previous if abs(previous[0] - targets[index]) <= abs(timestamp - targets[index]) else current
                append(best[1], best[0], rotation)
                index += 1
            previous = current
            if index == count:
                break
        if previous is None:
            raise ValueError("Source has no decodable timestamped frames")
        while index < count:
            append(previous[1], previous[0], rotation)
            index += 1
    return np.stack(low), np.stack(high), np.asarray(chosen), targets


class PairedVideoEncoder:
    def __init__(self, root, device, *, encoder_only=True):
        import importlib
        from safetensors import safe_open
        import torch
        from h3dmd.decoders import _local_vae_class
        root = Path(root) / "video_vae"
        self.config = json.loads((root / "config.json").read_text())
        cls = _local_vae_class(root, "minimax_h3_video_vae.MiniMaxH3VideoVAE")
        if encoder_only:
            # The released video VAE includes a 9+GiB ViT decoder. Encoding
            # never calls it; load only encoder/quantization tensors from the
            # same checkpoint, without paging decoder weights from shared NFS.
            module = importlib.import_module(cls.__module__)
            if self.config["vae_parallel_tiling"]:
                module._ensure_vae_parallel_state()
            source_cls = module._SOURCE_CLASSES[self.config["source_class_name"]]
            source_path = root / self.config["source_path"]
            names = ("clip_length", "token_drop", "encoder_tiling", "decoder_tiling",
                     "parallel_tiling", "tile_size", "tile_overlap_min",
                     "encoder_parallel", "decoder_parallel", "chunk_dim")
            options = {name: self.config["vae_" + name] for name in names}
            # Construct the unused multi-billion-parameter decoder on meta;
            # only the retained encoder checkpoint is materialized below.
            with torch.device('meta'):
                model = source_cls.from_config(source_cls.load_config(str(source_path)), **options)
            model.decoder = torch.nn.Identity()
            model.post_quant_conv = torch.nn.Identity()
            expected = set(model.state_dict())
            with safe_open(str(source_path / self.config["source_safetensors_path"]),
                           framework="pt", device="cpu") as weights:
                keys = set(weights.keys())
                discarded = keys - expected
                if expected - keys or any(not key.startswith(("decoder.", "post_quant_conv."))
                                          for key in discarded):
                    raise ValueError("Unexpected H3 encoder-only checkpoint structure")
                model.load_state_dict({key: weights.get_tensor(key) for key in expected}, strict=True, assign=True)
            if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):
                raise ValueError('H3 encoder checkpoint left unmaterialized tensors')
            self.model = cls(model)
        else:
            self.model = cls.from_pretrained(str(root))
        self.model = self.model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
        self.device = torch.device(device)

    def encode(self, pixels, seed=42):
        import torch
        n, height, width, _ = pixels.shape
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            value = self.model.encode_videos([pixels], use_fp16_latent=True)[0].float().cpu()
        expected = (24, ((n - 5) // 17) * 5 + 2, height // 16, width // 16)
        if tuple(value.shape) != expected or not torch.isfinite(value).all():
            raise ValueError("Invalid H3 paired VAE latent")
        mean = torch.tensor(self.config["latents_mean"]).view(24, 1, 1, 1)
        std = torch.tensor(self.config["latents_std"]).view(24, 1, 1, 1)
        value = ((value - mean) / std).contiguous()
        if not torch.isfinite(value).all():
            raise ValueError("Invalid standardized paired H3 latent")
        return value

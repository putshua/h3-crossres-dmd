"""Frame-local H3 clean-latent upsamplers adapted from the Wan crossres experiment.

Source: /mnt/data/sl/wan21_crossres_distill/src/crossres_distill/neural_upsampling.py
Source SHA256: 241a50ae9ec163395735ef25db3b43aa9702370be8e1e38364c6984bbf5d718c
Only spatial axes are mapped; audio and temporal positions are untouched.
H3 uses 24 video latent channels, rather than Wan's 16.
"""
import torch
from torch import nn
from torch.nn import functional as F
import hashlib
import math
from pathlib import Path


NEURAL_BRIDGES = {"dysample", "carafe", "liif", "rcan"}


def frames(x):
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)


def video(x, batch, time):
    return x.reshape(batch, time, *x.shape[1:]).permute(0, 2, 1, 3, 4).contiguous()


class EncoderBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.SiLU(), nn.Conv2d(width, width, 3, padding=1),
                                    nn.SiLU(), nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return x + 0.1 * self.layers(x)


class ChannelAttentionBlock(nn.Module):
    """Compact RCAN-style block; channel pooling is spatial and frame-local."""
    def __init__(self, width, reduction, residual_scale):
        super().__init__()
        self.residual_scale = residual_scale
        self.features = nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                      nn.ReLU(), nn.Conv2d(width, width, 3, padding=1))
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, max(1, width // reduction), 1), nn.ReLU(),
            nn.Conv2d(max(1, width // reduction), width, 1), nn.Sigmoid())

    def forward(self, x):
        residual = self.features(x)
        return x + self.residual_scale * residual * self.attention(residual)


class ResidualAttentionGroup(nn.Module):
    def __init__(self, width, blocks, reduction, residual_scale):
        super().__init__()
        self.body = nn.Sequential(*[
            ChannelAttentionBlock(width, reduction, residual_scale) for _ in range(blocks)
        ], nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class ResidualAttentionEncoder(nn.Module):
    """Residual-in-residual features, without RGB mean shift or BatchNorm."""
    def __init__(self, channels, width, groups, blocks, reduction, residual_scale):
        super().__init__()
        self.head = nn.Conv2d(channels, width, 3, padding=1)
        self.body = nn.Sequential(*[
            ResidualAttentionGroup(width, blocks, reduction, residual_scale) for _ in range(groups)
        ], nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        features = self.head(x)
        return features + self.body(features)


class NeuralCleanMap(nn.Module):
    """Fixed-size clean-latent map with a bicubic skip and learned residual.

Only the low-resolution latent is an input. Training targets or RGB guidance
are never available to the map during DMD. ``calibrated`` keeps the existing
objective's native-grid identity shortcut from misclassifying a nonlinear map.
"""
    calibrated = True
    interpolation = "bicubic"

    def __init__(self, size, kind, channels=24, width=64, groups=4, kernel_size=5,
                 max_offset=0.5, rcan_groups=2, rcan_blocks=4,
                 attention_reduction=16, residual_scale=0.1):
        super().__init__()
        if kind not in NEURAL_BRIDGES:
            raise ValueError(f"Unknown neural bridge: {kind}")
        if width % groups or kernel_size % 2 != 1 or max_offset <= 0:
            raise ValueError("Neural bridge needs divisible feature groups, an odd kernel and positive offset bound")
        self.size, self.kind = tuple(size), kind
        self.channels, self.width, self.groups = channels, width, groups
        self.kernel_size, self.max_offset = kernel_size, max_offset
        if kind == "rcan":
            if (min(rcan_groups, rcan_blocks, attention_reduction) <= 0
                    or not 0 < residual_scale <= 1):
                raise ValueError("Invalid residual channel attention architecture")
            self.encoder = ResidualAttentionEncoder(channels, width, rcan_groups, rcan_blocks,
                                                     attention_reduction, residual_scale)
        else:
            self.encoder = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1),
                                         EncoderBlock(width), EncoderBlock(width))
        if kind == "dysample":
            self.offset = nn.Conv2d(width, 2 * groups, 1)
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
        elif kind == "carafe":
            self.kernel = nn.Conv2d(width, groups * kernel_size ** 2, 1)
            nn.init.zeros_(self.kernel.weight)
            nn.init.zeros_(self.kernel.bias)
        if kind == "liif":
            self.decoder = nn.Sequential(nn.Conv2d(width + 4, width, 1), nn.SiLU(),
                                         nn.Conv2d(width, width, 1), nn.SiLU(),
                                         nn.Conv2d(width, channels, 1))
            last = self.decoder[-1]
        elif kind == "rcan":
            # Direct target-grid resize-convolution avoids an integer PixelShuffle grid
            # followed by a second resampling operation.
            self.decoder = nn.Conv2d(width, channels, 3, padding=1)
            last = self.decoder
        else:
            self.decoder = nn.Conv2d(width, channels, 1)
            last = self.decoder
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _grid(self, value):
        h, w = self.size
        yy = (torch.arange(h, device=value.device, dtype=torch.float32) + 0.5) * (2 / h) - 1
        xx = (torch.arange(w, device=value.device, dtype=torch.float32) + 0.5) * (2 / w) - 1
        y, x = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack([x, y], -1)

    def _dynamic_sample(self, value):
        n, c, h, w = value.shape
        offsets = F.interpolate(self.offset(value).float(), size=self.size, mode="bilinear", align_corners=False)
        offsets = self.max_offset * offsets.tanh().reshape(n, self.groups, 2, *self.size)
        scale = value.new_tensor([2 / w, 2 / h], dtype=torch.float32).view(1, 1, 2, 1, 1)
        grid = self._grid(value)[None, None] + (offsets * scale).permute(0, 1, 3, 4, 2)
        sampled = F.grid_sample(value.float().reshape(n * self.groups, c // self.groups, h, w),
                                grid.reshape(n * self.groups, *self.size, 2), mode="bilinear",
                                padding_mode="border", align_corners=False)
        return self.decoder(sampled.reshape(n, c, *self.size))

    def _reassemble(self, value):
        n, c, h, w = value.shape
        count, radius = self.kernel_size ** 2, self.kernel_size // 2
        logits = F.interpolate(self.kernel(value).float(), size=self.size, mode="bilinear", align_corners=False)
        weights = logits.reshape(n, self.groups, count, *self.size).softmax(2)
        # Gather one neighborhood offset at a time to avoid a materialized
        # [frames, channels, kernel_area, target_h, target_w] feature tensor.
        padded = F.pad(value.float(), (radius,) * 4, mode="replicate")
        assembled = torch.zeros(n, self.groups, c // self.groups, *self.size,
                                device=value.device, dtype=torch.float32)
        for ky in range(self.kernel_size):
            for kx in range(self.kernel_size):
                neighbor = F.interpolate(padded[:, :, ky:ky+h, kx:kx+w], size=self.size, mode="nearest")
                assembled = assembled + (neighbor.reshape(n, self.groups, c // self.groups, *self.size)
                                           * weights[:, :, ky * self.kernel_size + kx].unsqueeze(2))
        return self.decoder(assembled.reshape(n, c, *self.size))

    def _implicit(self, value):
        n, c, h, w = value.shape
        target_h, target_w = self.size
        grid = self._grid(value)
        x = (grid[..., 0] + 1) * w / 2 - 0.5
        y = (grid[..., 1] + 1) * h / 2 - 0.5
        x0, y0 = x.floor(), y.floor()
        dx, dy = x - x0, y - y0
        cell = value.new_tensor([h / target_h, w / target_w], dtype=torch.float32)
        cell = cell.view(1, 2, 1, 1).expand(n, 2, target_h, target_w)
        result = 0
        for oy, ox in ((0, 0), (0, 1), (1, 0), (1, 1)):
            ix, iy = (x0 + ox).clamp(0, w - 1), (y0 + oy).clamp(0, h - 1)
            index = (iy.long() * w + ix.long()).flatten().view(1, 1, -1).expand(n, c, -1)
            local = value.flatten(2).gather(2, index).reshape(n, c, target_h, target_w)
            relative = torch.stack([y - iy, x - ix], 0)[None].expand(n, 2, target_h, target_w)
            decoded = self.decoder(torch.cat([local.float(), relative, cell], 1))
            weight = (dy if oy else 1 - dy) * (dx if ox else 1 - dx)
            result = result + decoded * weight[None, None]
        return result

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != self.channels:
            raise ValueError("Expected a B,C,T,H,W clean latent with the configured channel count")
        if x.shape[-2] > self.size[0] or x.shape[-1] > self.size[1]:
            raise ValueError("A neural upsampler cannot downsample")
        b, _, t, _, _ = x.shape
        baseline = F.interpolate(frames(x.float()), size=self.size, mode="bicubic", align_corners=False)
        # The frame axis is folded into batch: score mapping cannot hide native
        # student flicker by reading or averaging neighboring latent frames.
        encoded = self.encoder(frames(x.float()))
        if self.kind == "dysample":
            residual = self._dynamic_sample(encoded)
        elif self.kind == "carafe":
            residual = self._reassemble(encoded)
        elif self.kind == "rcan":
            enlarged = F.interpolate(encoded.float(), size=self.size,
                                     mode="bicubic", align_corners=False)
            residual = self.decoder(enlarged)
        else:
            residual = self._implicit(encoded)
        return video(baseline + residual.float(), b, t)


def reconstruction_losses(prediction, target, spatial_weight=0.1, temporal_weight=0.1):
    """Match target edges and motion; never penalize motion toward a static clip."""
    error = prediction.float() - target.float()
    mse = error.square().mean()
    spatial = sum(error.diff(dim=dim).square().mean() for dim in (-2, -1)) * 0.5
    temporal = error.diff(dim=2).square().mean() if error.shape[2] > 1 else mse * 0
    loss = mse + spatial_weight * spatial + temporal_weight * temporal
    return loss, {"mse": mse, "spatial_mse": spatial, "temporal_mse": temporal}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapper_spec(config):
    c = config.crossres
    spec = dict(kind=c.mapper_kind, channels=24, width=c.mapper_width,
                groups=c.mapper_groups, low_pixels=[config.data.height, config.data.width],
                high_pixels=[c.score_height, c.score_width], spatial_compression=16,
                fps=24, temporal_mixing=False, normalization="h3_standardized_video_v1")
    if c.mapper_kind == "ltx":
        spec.update(temporal_mixing=True, blocks_per_stage=c.mapper_blocks_per_stage,
                    architecture="h3_ltx_3d_gn32_shuffle3_blur2_v1", spatial_scale=1.5)
    return spec


def new_mapper(config):
    c = config.crossres
    if c.mapper_kind == "ltx":
        from h3dmd.ltx_upsampling import LTXLatentUpsampler
        return LTXLatentUpsampler((c.score_height//16, c.score_width//16),
                                 width=c.mapper_width, blocks=c.mapper_blocks_per_stage,
                                 gradient_checkpointing=c.mapper_gradient_checkpointing)
    return NeuralCleanMap((c.score_height // 16, c.score_width // 16), c.mapper_kind,
                          channels=24, width=c.mapper_width, groups=c.mapper_groups)


class FrozenScoreMapper(nn.Module):
    """Immutable artifact binding survives DCP save/resume alongside its weights."""
    def __init__(self, network, sha256):
        super().__init__()
        self.network = network.eval().requires_grad_(False)
        # DCP can populate state tensors in-place before load_state_dict.
        # Keep the expected binding separately from its mutable saved buffer.
        self._expected_artifact_sha256 = bytes.fromhex(sha256)
        self.register_buffer("artifact_sha256", torch.tensor(list(self._expected_artifact_sha256), dtype=torch.uint8))

    def forward(self, value):
        return self.network(value)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "artifact_sha256"
        expected = torch.tensor(list(self._expected_artifact_sha256), dtype=torch.uint8)
        if key not in state_dict or not torch.equal(state_dict[key].cpu(), expected):
            error_msgs.append("Frozen score mapper differs from the mapper bound to this DMD checkpoint")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


class BicubicCleanMap(nn.Module):
    """Frame-local fixed interpolation, with its exact autograd adjoint."""
    def __init__(self, size):
        super().__init__()
        self.size = tuple(size)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("Expected B,C,T,H,W clean video latent")
        if tuple(x.shape[-2:]) == self.size:
            return x
        b, _, t, _, _ = x.shape
        mapped = F.interpolate(frames(x.float()), size=self.size,
                               mode="bicubic", align_corners=False)
        return video(mapped, b, t)


def load_frozen_mapper(config, device="cpu"):
    c = config.crossres
    if c.mapper_kind == "bicubic":
        if c.mapper_checkpoint or c.mapper_sha256:
            raise ValueError("Bicubic mapping cannot use neural artifact weights")
        return BicubicCleanMap((c.score_height // 16, c.score_width // 16)).to(device)
    path = Path(c.mapper_checkpoint).expanduser()
    if not c.mapper_checkpoint or not path.is_file():
        raise FileNotFoundError("Cross-resolution DMD requires a validated pretrained mapper_checkpoint")
    digest = file_sha256(path)
    if c.mapper_sha256 and digest != c.mapper_sha256:
        raise ValueError("Frozen mapper SHA256 changed")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("spec") != mapper_spec(config):
        raise ValueError("Frozen mapper architecture, geometry or H3 normalization mismatch")
    report = artifact.get("report", {})
    limits = {"mse_ratio": c.mse_ratio_limit, "temporal_mse_ratio": c.temporal_ratio_limit}
    if (not artifact.get("accepted") or artifact.get("pretrain_steps_completed", 0) < c.pretrain_steps
            or any(not isinstance(report.get(k), (int, float)) or
                   not math.isfinite(report[k]) or report[k] > limit for k, limit in limits.items())):
        raise ValueError("Frozen mapper has not passed heldout reconstruction acceptance")
    network = new_mapper(config)
    network.load_state_dict(artifact["model"], strict=True)
    if not all(torch.isfinite(p).all() for p in network.parameters()):
        raise ValueError("Frozen mapper contains non-finite weights")
    return FrozenScoreMapper(network, digest).eval().requires_grad_(False).to(device)

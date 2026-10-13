"""LTX-style spatial bridge trained from scratch on standardized H3 latents.

Architecture reference (no LTX weights or runtime dependency):
https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core/src/ltx_core/model/upsampler
3D residual stages surround a 3/2 spatial resampler. Unlike the official
LTX wrapper, this bridge directly consumes/returns H3-standardized latents.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class VideoResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1 = nn.Conv3d(width, width, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, width)
        self.conv2 = nn.Conv3d(width, width, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, width)

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        return F.silu(x + self.norm2(self.conv2(h)))


class RationalSpatialUpsample(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.expand = nn.Conv2d(width, 9*width, 3, padding=1)
        self.shuffle = nn.PixelShuffle(3)
        row = torch.tensor([1., 4., 6., 4., 1.])
        self.register_buffer('blur_kernel', (row[:, None]*row[None, :]/256)[None, None])

    def forward(self, x):
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
        y = self.shuffle(self.expand(y))
        kernel = self.blur_kernel.to(dtype=y.dtype).expand(c, 1, 5, 5)
        y = F.conv2d(y, kernel, stride=2, padding=2, groups=c)
        return y.reshape(b, t, c, *y.shape[-2:]).permute(0, 2, 1, 3, 4).contiguous()


class LTXLatentUpsampler(nn.Module):
    temporal_mixing = True
    calibrated = True

    def __init__(self, size, channels=24, width=128, blocks=4, gradient_checkpointing=False):
        super().__init__()
        if width < 32 or width % 32 or blocks < 1:
            raise ValueError('LTX width must be a positive multiple of32; blocks must be positive')
        self.size = tuple(size)
        self.channels = channels
        self.gradient_checkpointing = gradient_checkpointing
        self.head = nn.Conv3d(channels, width, 3, padding=1)
        self.head_norm = nn.GroupNorm(32, width)
        self.before = nn.ModuleList(VideoResidualBlock(width) for _ in range(blocks))
        self.resize = RationalSpatialUpsample(width)
        self.after = nn.ModuleList(VideoResidualBlock(width) for _ in range(blocks))
        self.tail = nn.Conv3d(width, channels, 3, padding=1)

    def _block(self, block, x):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != self.channels:
            raise ValueError('Expected B,C,T,H,W H3 video latents')
        if any(s % 2 for s in x.shape[-2:]) or tuple(3*s//2 for s in x.shape[-2:]) != self.size:
            raise ValueError('LTX bridge requires even source dimensions and an exact 1.5x target')
        h = F.silu(self.head_norm(self.head(x)))
        for block in self.before:
            h = self._block(block, h)
        h = self.resize(h)
        for block in self.after:
            h = self._block(block, h)
        return self.tail(h)

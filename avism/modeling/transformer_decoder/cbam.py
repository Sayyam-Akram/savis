"""
CBAM: Convolutional Block Attention Module
Adapted for both 2D spatial features (frame-level) and 1D temporal features (video-level).

Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 2D CBAM — For frame-level per-pixel features [B, C, H, W]
# ──────────────────────────────────────────────────────────────────────────────

class ChannelAttention2D(nn.Module):
    """Channel attention: learns WHICH feature channels are important."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Dual pooling across spatial dimensions
        avg_out = self.mlp(x.mean(dim=[2, 3]))   # [B, C]
        max_out = self.mlp(x.amax(dim=[2, 3]))   # [B, C]
        scale = torch.sigmoid(avg_out + max_out)  # [B, C]
        return x * scale[:, :, None, None]


class SpatialAttention2D(nn.Module):
    """Spatial attention: learns WHERE in H×W the important features are."""

    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        max_out = x.amax(dim=1, keepdim=True)  # [B, 1, H, W]
        scale = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAM2D(nn.Module):
    """Full CBAM for 2D spatial features: Channel → Spatial."""

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention2D(channels, reduction)
        self.spatial_attn = SpatialAttention2D(spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 1D CBAM — For video-level temporal features [B, C, T]
# ──────────────────────────────────────────────────────────────────────────────

class ChannelAttention1D(nn.Module):
    """Channel attention for 1D: learns WHICH channels are important."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        # x: [B, C, T]
        avg_out = self.mlp(x.mean(dim=2))   # [B, C]
        max_out = self.mlp(x.amax(dim=2))   # [B, C]
        scale = torch.sigmoid(avg_out + max_out)  # [B, C]
        return x * scale[:, :, None]


class TemporalAttention1D(nn.Module):
    """Temporal attention: learns WHICH time steps have reliable features."""

    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x):
        # x: [B, C, T]
        avg_out = x.mean(dim=1, keepdim=True)  # [B, 1, T]
        max_out = x.amax(dim=1, keepdim=True)  # [B, 1, T]
        scale = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAM1D(nn.Module):
    """Full CBAM for 1D temporal features: Channel → Temporal."""

    def __init__(self, channels, reduction=16, temporal_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention1D(channels, reduction)
        self.temporal_attn = TemporalAttention1D(temporal_kernel)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.temporal_attn(x)
        return x

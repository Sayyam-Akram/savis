"""
Audio-Conditioned CBAM (AC-CBAM) — v2
======================================
Extends CBAM with a LEARNED audio-visual spatial interaction for
audio-visual instance segmentation.

Channel Attention: Standard CBAM — learns WHICH feature channels matter.
Spatial Attention: Audio-Conditioned — combines visual pooling statistics
    with a learned audio-visual interaction map that captures complex
    spatial correspondences between audio and visual features.

Key difference from v1: Instead of a parameter-free cosine similarity
map (too weak), we use a learned convolutional network to compute the
audio-visual spatial map. This provides sufficient capacity to capture
rich audio-visual spatial patterns.

Initialization strategy:
    - External alpha (zero-init): module starts as identity
    - Internal av_interact last conv (zero-init, bias=-5): audio-visual
      map starts near-zero, so spatial attention initially behaves like
      standard CBAM and gradually learns audio conditioning

Reference:
    - Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
    - Extended with learned audio-visual spatial conditioning (ours).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Standard CBAM channel attention: learns WHICH channels are important."""

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
        avg_out = self.mlp(x.mean(dim=[2, 3]))   # [B, C]
        max_out = self.mlp(x.amax(dim=[2, 3]))   # [B, C]
        scale = torch.sigmoid(avg_out + max_out)  # [B, C]
        return x * scale[:, :, None, None]


class AudioConditionedSpatialAttention(nn.Module):
    """
    Audio-conditioned spatial attention with LEARNED interaction.

    Uses 3 spatial channels:
        1. Visual channel-wise average pooling     [B, 1, H, W]
        2. Visual channel-wise max pooling          [B, 1, H, W]
        3. Learned audio-visual interaction map     [B, 1, H, W]

    The interaction map is computed by a small conv network that takes
    concatenated visual features and broadcast audio features as input.
    This provides sufficient capacity to learn complex audio-visual
    spatial correspondences (unlike parameter-free cosine similarity).

    GroupNorm is used instead of BatchNorm for stability with batch_size=1.
    """

    def __init__(self, channels, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2

        # Audio projection to match visual feature space
        self.audio_proj = nn.Linear(channels, channels)

        # Learned audio-visual spatial interaction network
        # Input: concat(visual, audio_broadcast) → [B, 2C, H, W]
        # Output: spatial map → [B, 1, H, W]
        self.av_interact = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
        )

        # Combined spatial gate: visual_avg + visual_max + av_map → gate
        self.conv = nn.Conv2d(3, 1, kernel_size=kernel_size, padding=pad, bias=False)

        # NOTE: No zero-init here. External alpha=0 ensures identity start.
        # Internal weights use default Kaiming init so audio gradients flow
        # freely once alpha grows above zero during training.

    def forward(self, x, audio_features):
        """
        Args:
            x: Visual features [B, C, H, W]
            audio_features: Audio embeddings [B, C]
        Returns:
            Spatially attended features [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Visual spatial statistics (standard CBAM)
        vis_avg = x.mean(dim=1, keepdim=True)     # [B, 1, H, W]
        vis_max = x.amax(dim=1, keepdim=True)     # [B, 1, H, W]

        # Learned audio-visual interaction map
        a_proj = self.audio_proj(audio_features)                     # [B, C]
        a_spatial = a_proj[:, :, None, None].expand(-1, -1, H, W)   # [B, C, H, W]
        av_cat = torch.cat([x, a_spatial], dim=1)                   # [B, 2C, H, W]
        av_map = self.av_interact(av_cat)                            # [B, 1, H, W]

        # Combine 3 signals → spatial gate
        spatial_input = torch.cat([vis_avg, vis_max, av_map], dim=1)  # [B, 3, H, W]
        spatial_gate = torch.sigmoid(self.conv(spatial_input))        # [B, 1, H, W]

        return x * spatial_gate


class AudioConditionedCBAM(nn.Module):
    """
    Audio-Conditioned CBAM (AC-CBAM).

    Channel attention identifies important feature channels (visual-only).
    Spatial attention identifies important regions using a learned
    audio-visual interaction map alongside visual pooling statistics.

    Applied per feature scale with zero-initialized residual:
        x_out = x + alpha * AC-CBAM(x, audio)    [alpha init=0]

    At initialization, AC-CBAM ≈ standard CBAM (because av_map ≈ 0).
    Over training, the audio-visual map gradually learns to highlight
    sounding object regions.
    """

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = AudioConditionedSpatialAttention(channels, spatial_kernel)

    def forward(self, x, audio_features):
        """
        Args:
            x: Visual features [B, C, H, W]
            audio_features: Audio embeddings [B, C]
        Returns:
            Refined features [B, C, H, W]
        """
        x = self.channel_attn(x)
        x = self.spatial_attn(x, audio_features)
        return x

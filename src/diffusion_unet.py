# src/diffusion_unet.py

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 1. Timestep embedding
# -------------------------

def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    timesteps: (B,) int64
    dim: embedding dimension (must be even ideally)

    Returns: (B, dim)
    """
    half = dim // 2
    timesteps = timesteps.float()
    freqs = torch.exp(
        torch.arange(half, device=timesteps.device, dtype=torch.float32)
        * -(math.log(10000.0) / (half - 1))
    )
    args = timesteps[:, None] * freqs[None, :]  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)


# -------------------------
# 2. Building blocks
# -------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        t = self.time_mlp(t_emb)  # (B, out_ch)
        h = h + t[:, :, None, None]

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


# -------------------------
# 3. UNet model (fixed)
# -------------------------

class UNetModel(nn.Module):
    """
    Class-conditional UNet for DDPM on 224x224 RGB images.

    Args:
        in_channels: 3
        base_channels: starting channels (e.g. 64)
        channel_mults: e.g. (1, 2, 4, 8) -> resolutions: 224,112,56,28,14
        num_res_blocks: residual blocks per level
        num_classes: if > 0, class-conditional on labels [0..num_classes-1]
        time_emb_dim: time (and label) embedding dimension
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        num_classes: int = 0,
        time_emb_dim: int = 256,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.time_emb_dim = time_emb_dim

        # time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # optional class embedding
        if num_classes > 0:
            self.label_emb = nn.Embedding(num_classes, time_emb_dim)
        else:
            self.label_emb = None

        # channel schedule
        chs = [base_channels * m for m in channel_mults]  # per-level channels
        assert channel_mults[0] == 1, "channel_mults[0] must be 1"
        self.chs = chs
        num_levels = len(chs)
        self.num_levels = num_levels

        # input conv: 3 -> base_channels
        self.init_conv = nn.Conv2d(in_channels, chs[0], kernel_size=3, padding=1)

        # encoder: down_res_blocks[i] -> list of ResidualBlock for level i
        self.down_res_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()  # list of Downsample, length num_levels-1

        in_ch = chs[0]
        for i in range(num_levels):
            out_ch = chs[i]
            blocks = nn.ModuleList()
            for k in range(num_res_blocks):
                blocks.append(
                    ResidualBlock(
                        in_ch if k == 0 else out_ch,
                        out_ch,
                        time_emb_dim,
                    )
                )
            self.down_res_blocks.append(blocks)
            in_ch = out_ch
            if i != num_levels - 1:
                self.downs.append(Downsample(in_ch))

        # bottleneck at lowest resolution
        self.mid_block1 = ResidualBlock(in_ch, in_ch, time_emb_dim)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, time_emb_dim)

        # decoder: up_res_blocks[i] corresponds to level i (same spatial size as down_res_blocks[i])
        self.up_res_blocks = nn.ModuleList([None] * num_levels)
        self.ups = nn.ModuleList()  # length num_levels-1, but indexed as ups[j-1] in forward

        # create Upsample modules for levels 1..num_levels-1
        for i in range(1, num_levels):
            self.ups.append(Upsample(chs[i]))

        # create per-level up ResBlocks
        for j in reversed(range(num_levels)):
            out_ch = chs[j]
            if j == num_levels - 1:
                # deepest level: current chs[-1] + skip chs[-1]
                in_ch0 = chs[j] + chs[j]
            else:
                # at level j, input is upsampled from level j+1 (chs[j+1]) + skip chs[j]
                in_ch0 = chs[j + 1] + chs[j]

            blocks = nn.ModuleList()
            blocks.append(ResidualBlock(in_ch0, out_ch, time_emb_dim))
            for _ in range(num_res_blocks - 1):
                blocks.append(ResidualBlock(out_ch, out_ch, time_emb_dim))

            self.up_res_blocks[j] = blocks

        # final conv
        self.final_norm = nn.GroupNorm(8, chs[0])
        self.final_conv = nn.Conv2d(chs[0], in_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B,3,H,W)
        t: (B,) timesteps
        y: (B,) labels in [0,num_classes-1] or None
        returns: predicted noise eps_theta(x_t, t, y)
        """

        # --- time (and label) embedding ---
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        if self.label_emb is not None and y is not None:
            t_emb = t_emb + self.label_emb(y)
        t_emb = self.time_mlp(t_emb)  # (B, time_emb_dim)

        # --- encoder ---
        h = self.init_conv(x)
        skips = []

        # go down
        for i in range(self.num_levels):
            blocks = self.down_res_blocks[i]
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            if i != self.num_levels - 1:
                h = self.downs[i](h)

        # --- bottleneck ---
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # --- decoder ---
        # j = level index, from deepest (num_levels-1) to 0
        for j in reversed(range(self.num_levels)):
            skip = skips[j]
            # concatenate along channels (sizes now guaranteed to match in H,W)
            h = torch.cat([h, skip], dim=1)

            blocks = self.up_res_blocks[j]
            for block in blocks:
                h = block(h, t_emb)

            # upsample except for top level (j == 0)
            if j > 0:
                up = self.ups[j - 1]
                h = up(h)

        # final
        h = self.final_norm(h)
        h = self.act(h)
        out = self.final_conv(h)
        return out

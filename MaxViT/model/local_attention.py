from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from model.attention import *
from model.window_partition import * 
from model.grid_partition import *

AttnMode = Literal["window", "grid"]

@dataclass(frozen=True)
class LocalAttention2DConfig:
    mode: AttnMode                    # "window" or "grid"
    dim: int                          # embedding dim C
    num_heads: int
    # window params
    window_size: int = 4
    # grid params
    grid_size: int = 4
    # attention params
    qkv_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0

class LocalAttention2D(nn.Module):
    """
    Wrapper that applies MHSA over 2D groups (windows or grids).

    Input/Output: x in BHWC format:
        x: [B, H, W, C]  ->  y: [B, H, W, C]

    Pipeline:
      - Partition (window or grid) -> groups [Bgrp, h, w, C]
      - Flatten -> [Bgrp, N, C]
      - MHSA -> [Bgrp, N, C]
      - Unflatten -> [Bgrp, h, w, C]
      - Unpartition -> [B, H, W, C]
    """

    def __init__(self, cfg: LocalAttention2DConfig):
        super().__init__()

        if cfg.dim <= 0:
            raise ValueError("cfg.dim must be > 0")
        if cfg.num_heads <= 0:
            raise ValueError("cfg.num_heads must be > 0")
        if cfg.mode not in ("window", "grid"):
            raise ValueError("cfg.mode must be 'window' or 'grid'")

        self.cfg = cfg
        self.mode: AttnMode = cfg.mode

        self.mhsa = MultiHeadSelfAttention(
            AttentionConfig(
                dim=cfg.dim,
                num_heads=cfg.num_heads,
                qkv_bias=cfg.qkv_bias,
                attn_drop=cfg.attn_drop,
                proj_drop=cfg.proj_drop))

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected x.ndim==4 (BHWC). Got {tuple(x.shape)}")

        B, H, W, C = x.shape

        if C != self.cfg.dim:
            raise ValueError(f"Expected C=={self.cfg.dim}. Got C={C}")

        if self.mode == "window":
            ws = self.cfg.window_size
            groups = window_partition(x, ws)              # [B*nW, ws, ws, C]
            Bgrp, gh, gw, _ = groups.shape
            tokens = groups.view(Bgrp, gh * gw, C)         # [Bgrp, N, C]
            tokens = self.mhsa(tokens)
            groups = tokens.view(Bgrp, gh, gw, C)          # [Bgrp, ws, ws, C]
            out = window_unpartition(groups, ws, H=H, W=W, B=B)
            return out

        # grid mode
        g = self.cfg.grid_size
        grids, meta = grid_partition(x, g)                 # [B*g*g, Hg, Wg, C]
        Bgrp, gh, gw, _ = grids.shape
        tokens = grids.view(Bgrp, gh * gw, C)              # [Bgrp, N, C]
        tokens = self.mhsa(tokens)
        grids = tokens.view(Bgrp, gh, gw, C)
        out = grid_unpartition(grids, meta)
        return out
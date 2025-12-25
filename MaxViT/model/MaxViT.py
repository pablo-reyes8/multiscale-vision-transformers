from dataclasses import dataclass
from typing import Sequence, Optional, Literal, List, Dict, Any

import torch
import torch.nn as nn

from model.MaxViT_block import *


@dataclass(frozen=True)
class MaxViTConfig:
    # dataset/head
    num_classes: int = 100
    in_chans: int = 3

    # stem
    stem_type: Literal["A", "B"] = "A"
    stem_out_ch: int = 64
    stem_act: Literal["silu", "gelu", "relu"] = "silu"
    stem_use_bn: bool = True
    stem_mid_ch: Optional[int] = None  # only for stem B

    # pyramid
    dims: Sequence[int] = (64, 128, 256, 512)
    depths: Sequence[int] = (2, 2, 3, 2)
    heads: Sequence[int] = (2, 4, 8, 16)

    # attention partitions
    window_size: int = 4
    grid_size: int = 4

    # regularization
    drop_path_rate: float = 0.1
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    ffn_drop: float = 0.0

    # conv / ffn
    mbconv_expand_ratio: float = 4.0
    mbconv_se_ratio: float = 0.25
    mbconv_act: Literal["silu", "gelu", "relu"] = "silu"
    use_bn: bool = True

    mlp_ratio: float = 4.0
    mlp_act: Literal["silu", "gelu", "relu"] = "gelu"

    # downsample
    downsample_kind: Literal["conv", "pool"] = "conv"
    downsample_act: Literal["silu", "gelu", "relu"] = "silu"
    downsample_use_bn: bool = True


class MaxViT(nn.Module):
    """
    Full MaxViT for CIFAR-like inputs (32x32), assembled strictly from our lego pieces:

      Stem -> [Stage0] -> Down -> [Stage1] -> Down -> [Stage2] -> Down -> [Stage3] -> Head

    All stages keep resolution internally; downsampling happens only between stages.
    """

    def __init__(self, cfg: MaxViTConfig):
        super().__init__()
        self.cfg = cfg

        if not (len(cfg.dims) == len(cfg.depths) == len(cfg.heads)):
            raise ValueError("dims, depths, heads must have the same length")

        # Stem
        self.stem = MaxViTStem(
            StemConfig(
                stem_type=cfg.stem_type,
                out_ch=cfg.stem_out_ch,
                act=cfg.stem_act,
                use_bn=cfg.stem_use_bn,
                in_chans=cfg.in_chans,
                mid_ch=cfg.stem_mid_ch,))

        # DropPath schedule across all blocks
        total_blocks = int(sum(cfg.depths))
        dprs = _linspace_drop_path(total_blocks, cfg.drop_path_rate)

        # Build stages + downsamples
        stages = []
        downs = []

        cur = 0
        in_dim = cfg.dims[0]
        if cfg.stem_out_ch != in_dim:
            raise ValueError(f"stem_out_ch ({cfg.stem_out_ch}) must match dims[0] ({in_dim})")

        for si, (dim, depth, heads) in enumerate(zip(cfg.dims, cfg.depths, cfg.heads)):
            stage_dpr = dprs[cur : cur + depth]
            cur += depth

            stages.append(
                MaxViTStage(
                    dim=dim,
                    depth=depth,
                    num_heads=heads,
                    window_size=cfg.window_size,
                    grid_size=cfg.grid_size,
                    drop_path_rates=stage_dpr,
                    attn_drop=cfg.attn_drop,
                    proj_drop=cfg.proj_drop,
                    ffn_drop=cfg.ffn_drop,
                    mbconv_expand_ratio=cfg.mbconv_expand_ratio,
                    mbconv_se_ratio=cfg.mbconv_se_ratio,
                    mbconv_act=cfg.mbconv_act,
                    use_bn=cfg.use_bn,
                    mlp_ratio=cfg.mlp_ratio,
                    mlp_act=cfg.mlp_act,))

            # add downsample between stages (except after last stage)
            if si < len(cfg.dims) - 1:

                next_dim = cfg.dims[si + 1]
                downs.append(
                    Downsample(
                        in_ch=dim,
                        out_ch=next_dim,
                        cfg=DownsampleConfig(
                            kind=cfg.downsample_kind,
                            act=cfg.downsample_act,
                            use_bn=cfg.downsample_use_bn),))

        self.stages = nn.ModuleList(stages)
        self.downsamples = nn.ModuleList(downs)

        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(cfg.dims[-1], cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 32, 32]
        x = self.stem(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x
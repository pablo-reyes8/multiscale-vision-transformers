from dataclasses import dataclass
from typing import Sequence, Optional, Literal, List, Dict, Any

import torch
import torch.nn as nn

from model.MaxViT_block import *

def linspace_drop_path(total_blocks: int, drop_path_rate: float):
    """
    Linearly increases drop_path from 0 -> drop_path_rate across all blocks.
    """
    if total_blocks <= 0:
        return []
    if drop_path_rate <= 0:
        return [0.0] * total_blocks
    return torch.linspace(0.0, float(drop_path_rate), total_blocks).tolist()

def make_mbconv_cfg_kwargs(base: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make MBConvConfig kwargs robust to different MBConvConfig definitions.
    (Your earlier error suggests MBConvConfig may not have 'act' in some versions.)
    """
    ann = getattr(MBConvConfig, "__annotations__", {})
    return {k: v for k, v in base.items() if k in ann}

class MaxViTStage(nn.Module):
    """
    A stage = repeat MaxViTBlock 'depth' times at fixed resolution and channels.
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        grid_size: int,
        drop_path_rates: Sequence[float],
        # regularization
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        # conv
        mbconv_expand_ratio: float = 4.0,
        mbconv_se_ratio: float = 0.25,
        mbconv_act: Literal["silu", "gelu", "relu"] = "silu",
        use_bn: bool = True,
        # ffn
        mlp_ratio: float = 4.0,
        mlp_act: Literal["silu", "gelu", "relu"] = "gelu",):
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be > 0")
        if len(drop_path_rates) != depth:
            raise ValueError(f"drop_path_rates must have length=depth ({depth}). Got {len(drop_path_rates)}")

        blocks = []
        for i in range(depth):
            blocks.append(
                MaxViTBlock(
                    MaxViTBlockConfig(
                        dim=dim,
                        num_heads=num_heads,
                        window_size=window_size,
                        grid_size=grid_size,
                        mbconv_expand_ratio=mbconv_expand_ratio,
                        mbconv_se_ratio=mbconv_se_ratio,
                        mbconv_act=mbconv_act,
                        use_bn=use_bn,
                        mlp_ratio=mlp_ratio,
                        mlp_act=mlp_act,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        ffn_drop=ffn_drop,
                        drop_path=float(drop_path_rates[i]),)))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
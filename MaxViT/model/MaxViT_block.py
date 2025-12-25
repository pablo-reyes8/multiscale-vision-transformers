from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from model.max_vit_stem import *
from model.downsample import *
from model.MBConv import *
from model.window_partition import * 
from model.grid_partition import *
from model.attention import *
from model.local_attention import *


ActType = Literal["silu", "gelu", "relu"]


def _make_activation(act) -> nn.Module:
    act = act.lower()
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation '{act}'. Use one of: silu|gelu|relu")


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0, act= "gelu"):
        super().__init__()
        hidden = max(1, int(dim * mlp_ratio))
        self.fc1 = nn.Linear(dim, hidden)
        self.act = _make_activation(act)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


@dataclass(frozen=True)
class MaxViTBlockConfig:
    dim: int
    num_heads: int
    window_size: int = 4
    grid_size: int = 4

    # MBConv
    mbconv_expand_ratio: float = 4.0
    mbconv_se_ratio: float = 0.25
    mbconv_act: ActType = "silu"
    use_bn: bool = True

    # Transformer-ish
    mlp_ratio: float = 4.0
    mlp_act: ActType = "gelu"
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    ffn_drop: float = 0.0

    # regularization
    drop_path: float = 0.0



class MaxViTBlock(nn.Module):
    """
    MaxViT block (no downsampling inside; keep H,W constant):

      1) MBConv (NCHW)
      2) Block Attention (window) with Pre-LN + residual (BHWC)
      3) Grid  Attention (grid)   with Pre-LN + residual (BHWC)
      4) FFN/MLP                 with Pre-LN + residual (BHWC)

    Input/Output: [B, C, H, W] with C = dim
    """

    def __init__(self, cfg: MaxViTBlockConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.dim

        # MBConv
        self.mbconv = MBConv(
            in_ch=C,
            out_ch=C,
            stride=1,
            cfg=MBConvConfig(
                expand_ratio=cfg.mbconv_expand_ratio,
                se_ratio=cfg.mbconv_se_ratio,
                act=cfg.mbconv_act,
                use_bn=cfg.use_bn,
                drop_path=0.0,  # drop_path on residual branches below (common practice)
            ),)

        # Window attention branch (BHWC)
        self.norm1 = nn.LayerNorm(C)
        self.window_attn = LocalAttention2D(
            LocalAttention2DConfig(
                mode="window",
                dim=C,
                num_heads=cfg.num_heads,
                window_size=cfg.window_size,
                grid_size=cfg.grid_size,
                qkv_bias=True,
                attn_drop=cfg.attn_drop,
                proj_drop=cfg.proj_drop))

        self.drop_path1 = DropPath(cfg.drop_path) if cfg.drop_path > 0 else nn.Identity()

        # Grid attention branch (BHWC)
        self.norm2 = nn.LayerNorm(C)
        self.grid_attn = LocalAttention2D(
            LocalAttention2DConfig(
                mode="grid",
                dim=C,
                num_heads=cfg.num_heads,
                window_size=cfg.window_size,
                grid_size=cfg.grid_size,
                qkv_bias=True,
                attn_drop=cfg.attn_drop,
                proj_drop=cfg.proj_drop,))

        self.drop_path2 = DropPath(cfg.drop_path) if cfg.drop_path > 0 else nn.Identity()

        # FFN branch (BHWC)
        self.norm3 = nn.LayerNorm(C)
        self.mlp = MLP(dim=C, mlp_ratio=cfg.mlp_ratio, drop=cfg.ffn_drop, act=cfg.mlp_act)
        self.drop_path3 = DropPath(cfg.drop_path) if cfg.drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected x.ndim==4 (NCHW). Got {tuple(x.shape)}")
        B, C, H, W = x.shape
        if C != self.cfg.dim:
            raise ValueError(f"Expected C=={self.cfg.dim}. Got C={C}")

        # MBConv in NCHW
        x = self.mbconv(x)

        # convert to BHWC for attention & LN
        x_bhwc = x.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]

        # Window attention (Pre-LN)
        y = self.norm1(x_bhwc)
        y = self.window_attn(y)

        x_bhwc = x_bhwc + self.drop_path1(y)

        # Grid attention (Pre-LN)
        y = self.norm2(x_bhwc)
        y = self.grid_attn(y)
        x_bhwc = x_bhwc + self.drop_path2(y)

        # FFN (Pre-LN): apply MLP over tokens, i.e. last dim
        y = self.norm3(x_bhwc)
        y = self.mlp(y)
        x_bhwc = x_bhwc + self.drop_path3(y)

        # back to NCHW
        x_out = x_bhwc.permute(0, 3, 1, 2).contiguous()
        return x_out
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

StemType = Literal["A", "B"]
ActType = Literal["silu", "gelu", "relu"]

def _make_activation(act) -> nn.Module:
    act = act.lower()
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        # GELU no tiene inplace
        return nn.GELU()
    raise ValueError(f"Unknown activation '{act}'. Use one of: silu|gelu|relu")

@dataclass(frozen=True)
class StemConfig:
    stem_type: StemType = "A"     # "A" or "B"
    out_ch: int = 64              # C0
    act: ActType = "silu"
    use_bn: bool = True
    in_chans: int = 3
    mid_ch: Optional[int] = None  # if None -> out_ch // 2 (min 1)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1,
                 use_bn: bool = True, act: ActType = "silu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = _make_activation(act)

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))

class MaxViTStem(nn.Module):
    """
    CIFAR-100 stem (32x32), parametrizable:
      - Type A: 1x (Conv3x3 -> BN -> Act), stride=1
      - Type B: 2x (Conv3x3 -> BN -> Act) + (Conv3x3 -> BN -> Act), stride=1

    Output keeps spatial resolution: 32x32.
    """

    def __init__(self, cfg: StemConfig):
        super().__init__()
        if cfg.out_ch <= 0:
            raise ValueError("cfg.out_ch must be > 0")
        if cfg.in_chans <= 0:
            raise ValueError("cfg.in_chans must be > 0")

        stem_type = cfg.stem_type.upper()
        if stem_type not in ("A", "B"):
            raise ValueError("cfg.stem_type must be 'A' or 'B'")

        if stem_type == "A":
            self.stem = ConvBNAct(
                in_ch=cfg.in_chans,
                out_ch=cfg.out_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                use_bn=cfg.use_bn,
                act=cfg.act)
        else:
            mid = cfg.mid_ch if cfg.mid_ch is not None else max(1, cfg.out_ch // 2)
            self.stem = nn.Sequential(
                ConvBNAct(cfg.in_chans, mid, kernel_size=3, stride=1, padding=1, use_bn=cfg.use_bn, act=cfg.act),
                ConvBNAct(mid, cfg.out_ch, kernel_size=3, stride=1, padding=1, use_bn=cfg.use_bn, act=cfg.act))


        self.out_channels = cfg.out_ch
        self.stem_type = stem_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 32, 32] -> [B, C0, 32, 32]
        return self.stem(x)
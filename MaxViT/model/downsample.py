
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import torch
import torch.nn as nn


DownsampleType = Literal["conv", "pool"]
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

@dataclass(frozen=True)
class DownsampleConfig:
    kind: DownsampleType = "conv"  # "conv" or "pool"
    act: ActType = "silu"
    use_bn: bool = True


class Downsample(nn.Module):
    """
    Downsample block:
      - "conv": Conv3x3 stride2 padding1 (in_ch -> out_ch) + BN + Act
      - "pool": AvgPool2x2 + Conv1x1 (in_ch -> out_ch) + BN + Act

    Input:  [B, in_ch, H, W]
    Output: [B, out_ch, H/2, W/2]
    """

    def __init__(self, in_ch: int, out_ch: int, cfg: DownsampleConfig = DownsampleConfig()):
        super().__init__()
        if in_ch <= 0 or out_ch <= 0:
            raise ValueError("in_ch and out_ch must be > 0")

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kind = cfg.kind

        bn = (lambda c: nn.BatchNorm2d(c)) if cfg.use_bn else (lambda c: nn.Identity())
        act = _make_activation(cfg.act)

        if cfg.kind == "conv":
            self.op = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=not cfg.use_bn),
                bn(out_ch),
                act,)
        elif cfg.kind == "pool":
            self.op = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=not cfg.use_bn),
                bn(out_ch),
                act,)
        else:
            raise ValueError("cfg.kind must be 'conv' or 'pool'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)
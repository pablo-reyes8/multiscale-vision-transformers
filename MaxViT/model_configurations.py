from dataclasses import dataclass
from typing import Sequence, Optional, Literal, List, Dict, Any

from model.MaxViT import *

def maxvit_cifar100_tiny(
    stem_type: Literal["A", "B"] = "A",
    drop_path_rate: float = 0.1) -> MaxViTConfig:

    """
    A sane CIFAR-100 preset that plays nicely with 32->16->8->4 and ws=g=4.

    dims/depths/heads are chosen to be "MaxViT-ish" but CIFAR-friendly.
    """
    return MaxViTConfig(
        num_classes=100,
        in_chans=3,
        stem_type=stem_type,
        stem_out_ch=64,
        dims=(64, 128, 256, 512),
        depths=(2, 2, 3, 2),
        heads=(2, 4, 8, 16),
        window_size=4,
        grid_size=4,
        drop_path_rate=drop_path_rate,
        downsample_kind="conv",)




def maxvit_cifar100_small(
    stem_type: Literal["A", "B"] = "A",
    drop_path_rate: float = 0.15,):
    """
    Larger than tiny: more depth + a bit wider.
    Still safe for CIFAR pyramid 32->16->8->4 with ws=g=4.
    """
    return MaxViTConfig(
        num_classes=100,
        in_chans=3,
        stem_type=stem_type,
        stem_out_ch=80,                 # wider stem
        dims=(80, 160, 320, 640),       # wider pyramid
        depths=(3, 3, 5, 3),            # deeper
        heads=(4, 5, 10, 20),           # keep head_dim ~16-32
        window_size=4,
        grid_size=4,
        drop_path_rate=drop_path_rate,
        downsample_kind="conv",)




def maxvit_cifar100_base(
    stem_type: Literal["A", "B"] = "A",
    drop_path_rate: float = 0.20,) -> MaxViTConfig:
    """
    Even larger: comparable to a "base-ish" config for CIFAR.
    Heavier compute/memory; consider batch_size down (e.g., 64) if needed.
    """
    return MaxViTConfig(
        num_classes=100,
        in_chans=3,
        stem_type=stem_type,
        stem_out_ch=96,
        dims=(96, 192, 384, 768),
        depths=(4, 4, 8, 4),
        heads=(4, 6, 12, 24),
        window_size=4,
        grid_size=4,
        drop_path_rate=drop_path_rate,
        downsample_kind="conv",)
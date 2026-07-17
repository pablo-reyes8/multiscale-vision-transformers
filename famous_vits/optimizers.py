"""Optimizer factory shared by the high-level API and CLIs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from torch import Tensor, optim


def create_optimizer(
    parameters: Iterable[Tensor],
    name: str = "adamw",
    *,
    lr: float = 5e-4,
    **kwargs: Any,
):
    normalized = name.lower().replace("-", "")
    factories = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
    }
    if normalized not in factories:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {', '.join(factories)}.")
    return factories[normalized](parameters, lr=lr, **kwargs)

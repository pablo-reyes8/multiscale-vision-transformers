import os, math, random, inspect
from contextlib import contextmanager, nullcontext
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE_MAP = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16,  "float16": torch.float16,
    "fp32": torch.float32,  "float32": torch.float32,}

def _cuda_dtype_supported(dtype: torch.dtype) -> bool:
    if not torch.cuda.is_available():
        return False
    return dtype in (torch.float16, torch.bfloat16)

def make_grad_scaler(device: str = "cuda", enabled: bool = True):
    if not enabled:
        return None

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            sig = inspect.signature(torch.amp.GradScaler)
            if len(sig.parameters) >= 1:
                return torch.amp.GradScaler(device if device in ("cuda", "cpu") else "cuda")
            return torch.amp.GradScaler()
        except Exception:
            pass

    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler()
    return None


@contextmanager
def autocast_ctx(
    device: str = "cuda",
    enabled: bool = True,
    dtype: str = "fp16",
    cache_enabled: bool = True,):
    """
    Context manager de autocast:
      - cuda: fp16 por defecto (ideal en T4)
      - cpu: bfloat16 si está disponible
    """
    if not enabled:
        with nullcontext():
            yield
        return

    if device == "cuda":
        want = DTYPE_MAP.get(dtype.lower(), torch.float16)
        use = want if _cuda_dtype_supported(want) else torch.float16
        with torch.amp.autocast(device_type="cuda", dtype=use, cache_enabled=cache_enabled):
            yield
        return

    if device == "cpu":
        try:
            with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, cache_enabled=cache_enabled):
                yield
        except Exception:
            with nullcontext():
                yield
        return

    with nullcontext():
        yield
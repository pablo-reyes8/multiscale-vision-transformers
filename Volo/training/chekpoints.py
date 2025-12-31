import os, math, random, inspect
from contextlib import contextmanager, nullcontext
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_top1: float,
    extra: dict | None = None,):

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_top1": best_top1,
        "extra": extra or {},}
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
    strict: bool = True,):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=strict)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt
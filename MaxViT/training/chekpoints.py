import torch
from training.ema import *

def save_checkpoint(path: str, model, optimizer, scheduler, scaler, epoch: int, best_top1: float,
                    extra: dict | None = None, ema: ModelEMA | None = None):
    ckpt = {
        "model": model.state_dict(),
        "model_ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_top1": best_top1,
        "extra": extra or {}}
    torch.save(ckpt, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None, ema: ModelEMA | None = None,
                    map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)

    if ema is not None and ckpt.get("model_ema") is not None:
        ema.load_state_dict(ckpt["model_ema"])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt
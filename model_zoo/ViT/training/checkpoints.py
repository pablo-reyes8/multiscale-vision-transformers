import torch


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_top1: float,
    extra: dict | None = None,
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_top1": best_top1,
        "extra": extra or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"], strict=True)

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    return checkpoint

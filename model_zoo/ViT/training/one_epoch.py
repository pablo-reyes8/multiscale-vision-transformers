import time

import torch
import torch.nn as nn

from training.autocast import autocast_ctx
from training.training_metrics import accuracy_topk


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str = "cuda",
    scaler=None,
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    label_smoothing: float = 0.1,
    print_every: int = 100,
):
    model.train().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    use_scaler = (scaler is not None) and use_amp and autocast_dtype.lower() in ("fp16", "float16")

    running_loss = 0.0
    total = 0
    top1_correct = 0.0
    top3_correct = 0.0
    top5_correct = 0.0

    start_time = time.time()
    for step, (images, targets) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.size(0)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx(device=device, enabled=use_amp, dtype=autocast_dtype, cache_enabled=True):
            logits = model(images)

        loss = criterion(logits.float(), targets)

        if use_scaler:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        total += batch_size

        accs = accuracy_topk(logits.detach(), targets, ks=(1, 3, 5))
        top1_correct += accs[1] * batch_size / 100.0
        top3_correct += accs[3] * batch_size / 100.0
        top5_correct += accs[5] * batch_size / 100.0

        if print_every and step % print_every == 0:
            elapsed = time.time() - start_time
            imgs_sec = total / max(elapsed, 1e-9)
            print(
                f"[train step {step}/{len(dataloader)}] "
                f"loss {running_loss / total:.4f} | "
                f"top1 {100.0 * top1_correct / total:.2f}% | "
                f"top3 {100.0 * top3_correct / total:.2f}% | "
                f"top5 {100.0 * top5_correct / total:.2f}% | "
                f"{imgs_sec:.1f} img/s"
            )

    avg_loss = running_loss / total
    metrics = {
        "top1": 100.0 * top1_correct / total,
        "top3": 100.0 * top3_correct / total,
        "top5": 100.0 * top5_correct / total,
    }
    return avg_loss, metrics


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    label_smoothing: float = 0.1,
):
    model.eval().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    running_loss = 0.0
    total = 0
    top1_correct = 0.0
    top3_correct = 0.0
    top5_correct = 0.0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.size(0)

        with autocast_ctx(device=device, enabled=use_amp, dtype=autocast_dtype, cache_enabled=True):
            logits = model(images)

        loss = criterion(logits.float(), targets)

        running_loss += loss.item() * batch_size
        total += batch_size

        accs = accuracy_topk(logits, targets, ks=(1, 3, 5))
        top1_correct += accs[1] * batch_size / 100.0
        top3_correct += accs[3] * batch_size / 100.0
        top5_correct += accs[5] * batch_size / 100.0

    avg_loss = running_loss / total
    metrics = {
        "top1": 100.0 * top1_correct / total,
        "top3": 100.0 * top3_correct / total,
        "top5": 100.0 * top5_correct / total,
    }
    return avg_loss, metrics

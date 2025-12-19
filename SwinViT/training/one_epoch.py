import torch.nn as nn
import torch.nn.functional as F
import torch
import time

from training.training_metrics import * 
from training.cosine_lr import * 
from training.autocast import *

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str = "cuda",
    scaler=None,                       # GradScaler (solo FP16)
    autocast_dtype: str = "fp16",      # "fp16" o "bf16"
    use_amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    label_smoothing: float = 0.1,
    print_every: int = 100,):

    model.train().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    use_scaler = (scaler is not None) and use_amp and autocast_dtype.lower() in ("fp16", "float16")

    running_loss = 0.0
    total = 0
    c1 = c3 = c5 = 0.0

    t0 = time.time()
    for step, (images, targets) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        B = targets.size(0)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx(device=device, enabled=use_amp, dtype=autocast_dtype, cache_enabled=True):
            logits = model(images)  # [B, num_classes]

        # LOSS EN FP32 (clave para FP16 estable)
        loss = criterion(logits.float(), targets)

        if use_scaler:
            scaler.scale(loss).backward()
            # para clip hay que unscale primero
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

        # metricas
        running_loss += loss.item() * B
        total += B
        accs = accuracy_topk(logits.detach(), targets, ks=(1, 3, 5))
        c1 += accs[1] * B / 100.0
        c3 += accs[3] * B / 100.0
        c5 += accs[5] * B / 100.0

        if print_every and (step % print_every == 0):
            dt = time.time() - t0
            imgs_sec = total / max(dt, 1e-9)
            print(
                f"[train step {step}/{len(dataloader)}] "
                f"loss {running_loss/total:.4f} | "
                f"top1 {100*c1/total:.2f}% | top3 {100*c3/total:.2f}% | top5 {100*c5/total:.2f}% | "
                f"{imgs_sec:.1f} img/s")

    avg_loss = running_loss / total
    metrics = {"top1": 100.0 * c1 / total, "top3": 100.0 * c3 / total, "top5": 100.0 * c5 / total}
    return avg_loss, metrics


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    label_smoothing: float = 0.1):
    model.eval().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    running_loss = 0.0
    total = 0
    c1 = c3 = c5 = 0.0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        B = targets.size(0)

        with autocast_ctx(device=device, enabled=use_amp, dtype=autocast_dtype, cache_enabled=True):
            logits = model(images)

        loss = criterion(logits.float(), targets)

        running_loss += loss.item() * B
        total += B

        accs = accuracy_topk(logits, targets, ks=(1, 3, 5))
        c1 += accs[1] * B / 100.0
        c3 += accs[3] * B / 100.0
        c5 += accs[5] * B / 100.0

    avg_loss = running_loss / total
    metrics = {"top1": 100.0 * c1 / total, "top3": 100.0 * c3 / total, "top5": 100.0 * c5 / total}
    return avg_loss, metrics
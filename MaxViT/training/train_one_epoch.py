
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Any
from training.ema import ModelEMA
import time

from training.amp import * 
from training.cosine_lr import * 
from training.metrics import * 
from training.mixup_cutmix import * 


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str = "cuda",
    scaler=None,
    autocast_dtype: str = "bf16",
    use_amp: bool = True,
    grad_clip_norm: Optional[float] = 1.0,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mix_prob: float = 1.0,
    num_classes: int = 100,
    ema: Optional[ModelEMA] = None,
    channels_last: bool = False,
    print_every: int = 100):

    model.train()

    use_scaler = (scaler is not None) and use_amp and autocast_dtype.lower() in ("fp16", "float16")

    running_loss = 0.0
    total = 0
    c1 = c3 = c5 = 0.0

    t0 = time.time()
    for step, (images, targets) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        B = targets.size(0)

        # mixup/cutmix => soft targets
        images_aug, targets_soft = apply_mixup_cutmix(
            images, targets, num_classes=num_classes,
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, prob=mix_prob)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx(device=device, enabled=use_amp, dtype=autocast_dtype, cache_enabled=True):
            logits = model(images_aug)  # [B, K]

        # loss in fp32
        if (mixup_alpha > 0.0) or (cutmix_alpha > 0.0):
            # soft targets; label smoothing is usually redundant with mixup/cutmix.
            loss = soft_target_cross_entropy(logits.float(), targets_soft)
        else:
            loss = F.cross_entropy(logits.float(), targets, label_smoothing=label_smoothing)

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

        if ema is not None:
            ema.update(model)

        # metrics
        running_loss += loss.item() * B
        total += B
        accs = accuracy_topk(logits.detach(), targets_soft if targets_soft.ndim == 2 else targets, ks=(1, 3, 5))
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
                f"{imgs_sec:.1f} img/s | lr {optimizer.param_groups[0]['lr']:.2e}")

    avg_loss = running_loss / total
    metrics = {"top1": 100.0 * c1 / total, "top3": 100.0 * c3 / total, "top5": 100.0 * c5 / total}
    return avg_loss, metrics


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    autocast_dtype: str = "bf16",
    use_amp: bool = True,
    label_smoothing: float = 0.0,  # usually 0.0 for eval loss
    channels_last: bool = False):

    model.eval().to(device)

    running_loss = 0.0
    total = 0
    c1 = c3 = c5 = 0.0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        B = targets.size(0)

        with autocast_ctx(device=device, enabled=use_amp, dtype=autocast_dtype, cache_enabled=True):
            logits = model(images)

        loss = F.cross_entropy(logits.float(), targets, label_smoothing=label_smoothing)

        running_loss += loss.item() * B
        total += B

        accs = accuracy_topk(logits, targets, ks=(1, 3, 5))
        c1 += accs[1] * B / 100.0
        c3 += accs[3] * B / 100.0
        c5 += accs[5] * B / 100.0

    avg_loss = running_loss / total
    metrics = {"top1": 100.0 * c1 / total, "top3": 100.0 * c3 / total, "top5": 100.0 * c5 / total}
    return avg_loss, metrics
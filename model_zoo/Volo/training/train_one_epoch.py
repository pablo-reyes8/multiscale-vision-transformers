from typing import Optional
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.metrics import * 
from training.ddp_utils import * 
from training.autocast import * 
from training.mixup_cutmix_loss import *

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str = "cuda",
    scaler=None,
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    grad_clip_norm: Optional[float] = 1.0,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mix_prob: float = 1.0,
    num_classes: int = 100,
    channels_last: bool = False,
    print_every: int = 100,
):
    model.train()

    use_scaler = (scaler is not None) and use_amp and autocast_dtype.lower() in ("fp16", "float16")

    running_loss = 0.0
    total = 0
    c1 = c3 = c5 = 0.0  # “correct counts” acumulados (no %)

    t0 = time.time()
    for step, (images, targets) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        B = targets.size(0)

        images_aug, targets_soft = apply_mixup_cutmix(
            images, targets,
            num_classes=num_classes,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=mix_prob
        )

        use_mix = (mixup_alpha > 0.0) or (cutmix_alpha > 0.0)
        targets_for_acc = targets_soft if use_mix else targets

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx(device=device, enabled=use_amp, dtype=autocast_dtype, cache_enabled=True):
            logits = model(images_aug)

        if use_mix:
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

        # métricas “locales”
        running_loss += loss.item() * B
        total += B
        accs = accuracy_topk(logits.detach(), targets_for_acc, ks=(1, 3, 5))
        c1 += accs[1] * B / 100.0
        c3 += accs[3] * B / 100.0
        c5 += accs[5] * B / 100.0

        # log solo en rank0 (si DDP)
        if print_every and (step % print_every == 0) and is_main_process():
            dt = time.time() - t0
            imgs_sec = total / max(dt, 1e-9)
            print(
                f"[train step {step}/{len(dataloader)}] "
                f"loss {running_loss/total:.4f} | "
                f"top1 {100*c1/total:.2f}% | top3 {100*c3/total:.2f}% | top5 {100*c5/total:.2f}% | "
                f"{imgs_sec:.1f} img/s | lr {optimizer.param_groups[0]['lr']:.2e}"
            )

    # ---- REDUCCIÓN GLOBAL (DDP) ----
    stats = torch.tensor([running_loss, total, c1, c3, c5], device=device, dtype=torch.float64)
    ddp_sum_(stats)
    running_loss_g, total_g, c1_g, c3_g, c5_g = stats.tolist()

    avg_loss = running_loss_g / max(total_g, 1e-12)
    metrics = {
        "top1": 100.0 * c1_g / max(total_g, 1e-12),
        "top3": 100.0 * c3_g / max(total_g, 1e-12),
        "top5": 100.0 * c5_g / max(total_g, 1e-12),}
    
    return avg_loss, metrics



@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    label_smoothing: float = 0.0,
    channels_last: bool = False,
):
    model.eval()

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

    # ---- REDUCCIÓN GLOBAL (DDP) ----
    stats = torch.tensor([running_loss, total, c1, c3, c5], device=device, dtype=torch.float64)
    ddp_sum_(stats)
    running_loss_g, total_g, c1_g, c3_g, c5_g = stats.tolist()

    avg_loss = running_loss_g / max(total_g, 1e-12)
    metrics = {
        "top1": 100.0 * c1_g / max(total_g, 1e-12),
        "top3": 100.0 * c3_g / max(total_g, 1e-12),
        "top5": 100.0 * c5_g / max(total_g, 1e-12),}
    
    return avg_loss, metrics
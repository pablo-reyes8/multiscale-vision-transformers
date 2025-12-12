import torch.nn as nn
import torch 

from training.autocast import * 
from training.training_utils import *


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
    scaler=None,
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    print_every: int = 100):

    """
    Entrena un epoch sobre dataloader usando AMP (fp16/bf16) si se desea.

    Args:
        model: modelo (HierarchicalViT, ResNet, etc.)
        dataloader: DataLoader de training
        optimizer: optimizador
        device: "cuda" o "cpu"
        scaler: GradScaler opcional (para FP16). Si None y autocast_dtype es FP16,
                puedes crear uno fuera con make_grad_scaler.
        autocast_dtype: "fp16", "float16", "bf16"...
        use_amp: si False, entrena en full precision.
        print_every: cada cuántos batches imprimir stats.

    Returns:
        avg_loss, metrics_dict
        donde metrics_dict = {"top1": ..., "top3": ..., "top5": ...}
    """
    model.train()
    model.to(device)

    # Criterio
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    running_loss = 0.0
    total_samples = 0
    correct_top1 = 0.0
    correct_top3 = 0.0
    correct_top5 = 0.0

    # Banderas para saber si usamos scaler realmente
    use_scaler = (scaler is not None) and use_amp and (
        autocast_dtype.lower() in ("fp16", "float16"))

    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx(
            device=device,
            enabled=use_amp,
            dtype=autocast_dtype,
            cache_enabled=True):

            logits = model(images)           # [B, num_classes]
            loss = criterion(logits, targets)

        if use_scaler:
            # FP16 con GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # BF16 o full precision
            loss.backward()
            optimizer.step()

        # Métricas
        B = targets.size(0)
        total_samples += B
        running_loss += loss.item() * B

        accs = accuracy_topk(logits, targets, ks=(1, 3, 5))
        correct_top1 += accs[1] * B / 100.0
        correct_top3 += accs[3] * B / 100.0
        correct_top5 += accs[5] * B / 100.0

        if (step + 1) % print_every == 0:
            avg_loss_step = running_loss / total_samples
            top1_step = 100.0 * correct_top1 / total_samples
            top3_step = 100.0 * correct_top3 / total_samples
            top5_step = 100.0 * correct_top5 / total_samples

            print(
                f"[Step {step+1}/{len(dataloader)}] "
                f"Loss: {avg_loss_step:.4f} | "
                f"Top-1: {top1_step:.2f}% | "
                f"Top-3: {top3_step:.2f}% | "
                f"Top-5: {top5_step:.2f}%")

    avg_loss = running_loss / total_samples
    top1 = 100.0 * correct_top1 / total_samples
    top3 = 100.0 * correct_top3 / total_samples
    top5 = 100.0 * correct_top5 / total_samples

    metrics = {
        "top1": top1,
        "top3": top3,
        "top5": top5}

    return avg_loss, metrics


def evaluate_one_epoch(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    autocast_dtype: str = "fp16",
    use_amp: bool = True,):

    """
    Evalúa un modelo en un dataloader (val/test) sin actualizar pesos.

    Returns:
        avg_loss, metrics_dict  (top1, top3, top5)
    """
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    running_loss = 0.0
    total_samples = 0
    correct_top1 = 0.0
    correct_top3 = 0.0
    correct_top5 = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast_ctx(
                device=device,
                enabled=use_amp,
                dtype=autocast_dtype,
                cache_enabled=True):

                logits = model(images)
                loss = criterion(logits, targets)

            B = targets.size(0)
            total_samples += B
            running_loss += loss.item() * B

            accs = accuracy_topk(logits, targets, ks=(1, 3, 5))
            correct_top1 += accs[1] * B / 100.0
            correct_top3 += accs[3] * B / 100.0
            correct_top5 += accs[5] * B / 100.0

    avg_loss = running_loss / total_samples
    top1 = 100.0 * correct_top1 / total_samples
    top3 = 100.0 * correct_top3 / total_samples
    top5 = 100.0 * correct_top5 / total_samples

    metrics = {
        "top1": top1,
        "top3": top3,
        "top5": top5}

    return avg_loss, metrics
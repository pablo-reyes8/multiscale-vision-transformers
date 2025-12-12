import torch 
import torch.nn as nn

from training.autocast import * 
from training.training_utils import *
from training.train_one_epoch import *


def train_vit(
    model: nn.Module,
    train_loader,
    val_loader=None,
    epochs: int = 10,
    device: str = "cuda",
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    use_grad_scaler: bool = True,
    save_best: bool = True,
    save_path: str | None = "best_vit.pth",
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    print_every: int = 100,
):
    """
    Entrena un ViT (o cualquier modelo) varios epochs y opcionalmente:
    - evalúa en val_loader por época,
    - guarda el mejor modelo en validación (top1 más alto).
    """
    model.to(device)

    # --- OPTIMIZER ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,)

    # --- SCHEDULER ---
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,)

    # --- AMP / SCALER ---
    scaler = None
    if use_grad_scaler and use_amp and autocast_dtype.lower() in ("fp16", "float16"):
        scaler = make_grad_scaler(device=device, enabled=True)

    best_val_top1 = -float("inf")
    history = {
        "train_loss": [],
        "train_top1": [],
        "train_top3": [],
        "train_top5": [],
        "val_loss": [],
        "val_top1": [],
        "val_top3": [],
        "val_top5": [],}

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")

        # ---- TRAIN ----
        train_loss, train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            autocast_dtype=autocast_dtype,
            use_amp=use_amp,
            print_every=print_every,)

        history["train_loss"].append(train_loss)
        history["train_top1"].append(train_metrics["top1"])
        history["train_top3"].append(train_metrics["top3"])
        history["train_top5"].append(train_metrics["top5"])

        # Scheduler por época
        if scheduler is not None:
            scheduler.step()

        print(
            f"[Train] Loss: {train_loss:.4f} | "
            f"Top-1: {train_metrics['top1']:.2f}% | "
            f"Top-3: {train_metrics['top3']:.2f}% | "
            f"Top-5: {train_metrics['top5']:.2f}%")

        if val_loader is not None:
            val_loss, val_metrics = evaluate_one_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                autocast_dtype=autocast_dtype,
                use_amp=use_amp)

            history["val_loss"].append(val_loss)
            history["val_top1"].append(val_metrics["top1"])
            history["val_top3"].append(val_metrics["top3"])
            history["val_top5"].append(val_metrics["top5"])

            print(
                f"[Val]   Loss: {val_loss:.4f} | "
                f"Top-1: {val_metrics['top1']:.2f}% | "
                f"Top-3: {val_metrics['top3']:.2f}% | "
                f"Top-5: {val_metrics['top5']:.2f}%")

            if save_best and save_path is not None:
                if val_metrics["top1"] > best_val_top1:
                    best_val_top1 = val_metrics["top1"]
                    torch.save(model.state_dict(), save_path)
                    print(
                        f"Nuevo mejor modelo guardado en {save_path} "
                        f"(Val Top-1 = {best_val_top1:.2f}%)")
        else:
            print("No val_loader: sólo se registran métricas de training.")

    return history
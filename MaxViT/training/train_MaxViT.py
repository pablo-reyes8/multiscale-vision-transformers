import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Any
from training.ema import ModelEMA
import time

from training.train_one_epoch import * 
from training.chekpoints import *


def train_model(
    model: nn.Module,
    train_loader,
    epochs: int,
    val_loader=None,
    device: str = "cuda",
    lr: float = 5e-4,
    weight_decay: float = 0.05,
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    warmup_ratio: float = 0.05,
    min_lr: float = 0.0,
    label_smoothing: float = 0.1,
    print_every: int = 100,
    save_path: str = "best_model.pt",
    last_path: str = "last_model.pt",
    resume_path: str | None = None,
    # EMA
    use_ema: bool = False,
    ema_decay: float = 0.999,
    ema_device: str | None = None,
    ema_warmup_epochs: int = 5,

    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mix_prob: float = 1.0,
    num_classes: int = 100,
    channels_last: bool = False,

    early_stop_on_val_drop: bool = True,
    early_stop_window: int = 3,
    early_stop_metric: str = "top1",   # "top1" o "loss"
    early_stop_drop_top1_points: float = 7.0,
    early_stop_rise_loss: float = 0.5,
    early_stop_require_monotonic: bool = True):

    model.to(device)

    # Optimizer
    param_groups = build_param_groups_no_wd(model, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)

    # Scheduler warmup + cosine (step-based)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = WarmupCosineLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=min_lr)

    # AMP scaler
    scaler = None
    if use_amp and autocast_dtype.lower() in ("fp16", "float16"):
        scaler = make_grad_scaler(device=device, enabled=True)

    # EMA (opcional)
    ema = ModelEMA(model, decay=ema_decay, device=ema_device) if use_ema else None

    start_epoch = 0
    best_val_top1 = -float("inf")

    if resume_path is not None:
        ckpt = load_checkpoint(
            resume_path, model,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            ema=ema,  # si ema=None, load_checkpoint debe ignorarlo
            map_location=device)

        start_epoch = int(ckpt.get("epoch", 0))
        best_val_top1 = float(ckpt.get("best_top1", best_val_top1))
        print(f"Resumed from {resume_path} at epoch {start_epoch} | best_top1 {best_val_top1:.2f}%")

    history = {
        "train_loss": [], "train_top1": [], "train_top3": [], "train_top5": [],
        "val_loss": [], "val_top1": [], "val_top3": [], "val_top5": [],}

    def _should_early_stop():
        if not early_stop_on_val_drop or val_loader is None:
            return False, ""

        w = int(early_stop_window)
        if w <= 1:
            return False, ""

        if early_stop_metric.lower() == "top1":
            if len(history["val_top1"]) < w:
                return False, ""
            recent = history["val_top1"][-w:]
            drop = recent[0] - recent[-1]
            monotonic = True if not early_stop_require_monotonic else all(
                recent[i] >= recent[i + 1] for i in range(len(recent) - 1))

            if monotonic and drop >= float(early_stop_drop_top1_points):
                msg = (f"Early-stop: val_top1 cayó {drop:.2f} puntos en las últimas {w} épocas "
                       f"({recent[0]:.2f}% -> {recent[-1]:.2f}%).")
                return True, msg
            return False, ""

        elif early_stop_metric.lower() == "loss":
            if len(history["val_loss"]) < w:
                return False, ""
            recent = history["val_loss"][-w:]
            rise = recent[-1] - recent[0]
            monotonic = True if not early_stop_require_monotonic else all(
                recent[i] <= recent[i + 1] for i in range(len(recent) - 1))

            if monotonic and rise >= float(early_stop_rise_loss):
                msg = (f"Early-stop: val_loss subió {rise:.4f} en las últimas {w} épocas "
                       f"({recent[0]:.4f} -> {recent[-1]:.4f}).")
                return True, msg
            return False, ""
        else:
            raise ValueError(f"early_stop_metric debe ser 'top1' o 'loss'. Recibí: {early_stop_metric}")

    for epoch in range(start_epoch + 1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        t_epoch = time.time()

        tr_loss, tr_m = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            autocast_dtype=autocast_dtype,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
            label_smoothing=label_smoothing,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mix_prob=mix_prob,
            num_classes=num_classes,
            channels_last=channels_last,
            ema=ema,
            print_every=print_every,)

        history["train_loss"].append(tr_loss)
        history["train_top1"].append(tr_m["top1"])
        history["train_top3"].append(tr_m["top3"])
        history["train_top5"].append(tr_m["top5"])
        print(f"[Train] loss {tr_loss:.4f} | top1 {tr_m['top1']:.2f}% | top3 {tr_m['top3']:.2f}% | top5 {tr_m['top5']:.2f}%")

        save_checkpoint(
            last_path, model, optimizer, scheduler, scaler,
            epoch=epoch, best_top1=best_val_top1,
            extra={
                "autocast_dtype": autocast_dtype,
                "use_amp": use_amp,
                "use_ema": use_ema,
                "ema_decay": ema_decay if use_ema else None},
            ema=ema, )

        if val_loader is not None:
            # Val: normal primero, luego EMA (si aplica)
            if (not use_ema) or (ema is None) or (epoch <= ema_warmup_epochs):
                va_loss, va_m = evaluate_one_epoch(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    use_amp=use_amp,
                    label_smoothing=0.0,)
                tag = "Val"
            else:
                with ema.use_ema_weights(model):
                    va_loss, va_m = evaluate_one_epoch(
                        model=model,
                        dataloader=val_loader,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        use_amp=use_amp,
                        label_smoothing=0.0,)

                tag = "Val-EMA"

            history["val_loss"].append(va_loss)
            history["val_top1"].append(va_m["top1"])
            history["val_top3"].append(va_m["top3"])
            history["val_top5"].append(va_m["top5"])
            print(f"[{tag}] loss {va_loss:.4f} | top1 {va_m['top1']:.2f}% | top3 {va_m['top3']:.2f}% | top5 {va_m['top5']:.2f}%")

            if va_m["top1"] > best_val_top1:
                best_val_top1 = va_m["top1"]
                save_checkpoint(
                    save_path, model, optimizer, scheduler, scaler,
                    epoch=epoch, best_top1=best_val_top1,
                    extra={
                        "autocast_dtype": autocast_dtype,
                        "use_amp": use_amp,
                        "use_ema": use_ema,
                        "ema_decay": ema_decay if use_ema else None},ema=ema,)

                print(f" Best saved to {save_path} (val top1 {best_val_top1:.2f}%)")

            stop, msg = _should_early_stop()
            if stop:
                print(msg)
                print("Stopping training early due to strong validation degradation.")
                break

        dt = time.time() - t_epoch
        print(f"Epoch time: {dt/60:.2f} min")

    # devolver modelo con EMA weights cargados SOLO si EMA estuvo activo
    if use_ema and ema is not None:
        model.load_state_dict(ema.ema.state_dict(), strict=True)

    return history, model
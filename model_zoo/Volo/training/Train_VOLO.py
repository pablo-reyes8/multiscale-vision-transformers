import time
import torch
import torch.nn as nn

from training.chekpoints import * 
from training.ddp_utils import * 
from training.train_one_epoch import * 
from training.metrics import *
from training.warmup import *


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

    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mix_prob: float = 1.0,
    num_classes: int = 100,
    channels_last: bool = False,

    early_stop: bool = True,
    early_stop_metric: str = "top1",
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 0.0,
    early_stop_require_monotonic: bool = False):

    model.to(device)

    # Optimizer
    param_groups = build_param_groups_no_wd(model, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)

    # Scheduler warmup + cosine (step-based)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = WarmupCosineLR(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
    )

    scaler = None
    if use_amp and autocast_dtype.lower() in ("fp16", "float16"):
        scaler = make_grad_scaler(device=device, enabled=True)

    # Resume
    start_epoch = 0
    best_val_top1 = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0

    if resume_path is not None:
        ckpt = load_checkpoint(
            resume_path, model,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            map_location=device,
            strict=True,
        )

        start_epoch = int(ckpt.get("epoch", 0))
        best_val_top1 = float(ckpt.get("best_top1", best_val_top1))
        extra = ckpt.get("extra", {}) or {}
        best_val_loss = float(extra.get("best_val_loss", best_val_loss))
        best_epoch = int(extra.get("best_epoch", best_epoch))

        if is_main_process():
            print(f"Resumed from {resume_path} at epoch {start_epoch} | best_top1 {best_val_top1:.2f}% | best_loss {best_val_loss:.4f}")

    history = {
        "train_loss": [], "train_top1": [], "train_top3": [], "train_top5": [],
        "val_loss": [], "val_top1": [], "val_top3": [], "val_top5": [],
        "lr": []} if is_main_process() else None  # <- solo rank0 guarda history

    # Early stop state (solo rank0 lleva el estado)
    metric = early_stop_metric.lower()
    assert metric in ("top1", "loss")
    patience = int(early_stop_patience)
    mode = "max" if metric == "top1" else "min"
    best_metric = best_val_top1 if metric == "top1" else best_val_loss
    bad_epochs = 0
    last_vals = []

    def _is_improvement(curr: float, best: float) -> bool:
        d = float(early_stop_min_delta)
        return (curr > (best + d)) if mode == "max" else (curr < (best - d))

    def _degradation_monotonic(vals: list[float]) -> bool:
        if not early_stop_require_monotonic or len(vals) < 2:
            return True
        if mode == "max":
            return all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
        else:
            return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

    for epoch in range(start_epoch + 1, epochs + 1):
        if is_main_process():
            print(f"\n=== Epoch {epoch}/{epochs} ===")
        t_epoch = time.time()

        # DDP: reshuffle correcto por epoch
        if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if val_loader is not None and hasattr(val_loader, "sampler") and isinstance(val_loader.sampler, DistributedSampler):
            val_loader.sampler.set_epoch(epoch)

        # --- Train ---
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
            print_every=print_every,)

        if is_main_process():
            history["train_loss"].append(tr_loss)
            history["train_top1"].append(tr_m["top1"])
            history["train_top3"].append(tr_m["top3"])
            history["train_top5"].append(tr_m["top5"])
            history["lr"].append(optimizer.param_groups[0]["lr"])

            print(f"[Train] loss {tr_loss:.4f} | top1 {tr_m['top1']:.2f}% | top3 {tr_m['top3']:.2f}% | top5 {tr_m['top5']:.2f}% | lr {optimizer.param_groups[0]['lr']:.2e}")

            # ✅ guardar "last" SOLO en rank0
            save_checkpoint(
                last_path, model, optimizer, scheduler, scaler,
                epoch=epoch, best_top1=best_val_top1,
                extra={
                    "autocast_dtype": autocast_dtype,
                    "use_amp": use_amp,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "early_stop_metric": metric,
                    "early_stop_patience": patience,
                    "early_stop_min_delta": float(early_stop_min_delta),
                },
            )

        stop_now = False

        # --- Val ---
        if val_loader is not None:
            va_loss, va_m = evaluate_one_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                autocast_dtype=autocast_dtype,
                use_amp=use_amp,
                label_smoothing=0.0,
                channels_last=channels_last)

            if is_main_process():
                history["val_loss"].append(va_loss)
                history["val_top1"].append(va_m["top1"])
                history["val_top3"].append(va_m["top3"])
                history["val_top5"].append(va_m["top5"])

                print(f"[Val]   loss {va_loss:.4f} | top1 {va_m['top1']:.2f}% | top3 {va_m['top3']:.2f}% | top5 {va_m['top5']:.2f}%")

                # Best saved por top1
                if va_m["top1"] > best_val_top1:
                    best_val_top1 = va_m["top1"]
                    if va_loss < best_val_loss:
                        best_val_loss = va_loss
                        best_epoch = epoch

                    save_checkpoint(
                        save_path, model, optimizer, scheduler, scaler,
                        epoch=epoch, best_top1=best_val_top1,
                        extra={
                            "autocast_dtype": autocast_dtype,
                            "use_amp": use_amp,
                            "best_val_loss": best_val_loss,
                            "best_epoch": best_epoch,
                        },
                    )
                    print(f"Best saved to {save_path} (val top1 {best_val_top1:.2f}%)")

                # Early stop (solo rank0 decide)
                if early_stop:
                    curr_metric = va_m["top1"] if metric == "top1" else va_loss

                    last_vals.append(float(curr_metric))
                    if len(last_vals) > patience:
                        last_vals = last_vals[-patience:]

                    if _is_improvement(curr_metric, best_metric):
                        best_metric = float(curr_metric)
                        bad_epochs = 0
                    else:
                        bad_epochs += 1

                    if bad_epochs >= patience and _degradation_monotonic(last_vals):
                        print(f"Early-stop: no improvement on val_{metric} for {patience} epochs.")
                        stop_now = True

        # DDP: sincroniza el “stop” a todos los ranks
        stop_now = ddp_broadcast_bool(stop_now, device=device)
        if stop_now:
            break

        if is_main_process():
            dt = time.time() - t_epoch
            print(f"Epoch time: {dt/60:.2f} min")

    # return: history solo en rank0; en otros ranks devuelve None
    return history, (model.module if hasattr(model, "module") else model)
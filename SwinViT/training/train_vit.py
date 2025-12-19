import torch.nn as nn
import torch.nn.functional as F
import torch
import time

from training.training_metrics import * 
from training.cosine_lr import * 
from training.autocast import *
from training.one_epoch import *
from training.chekpoints import *

def train_swinvit(
    model: nn.Module,
    train_loader,
    val_loader=None,
    epochs: int = 50,
    device: str = "cuda",
    lr: float = 5e-4,
    weight_decay: float = 0.05,
    autocast_dtype: str = "fp16",   # para A100 usual: "bf16"
    use_amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    warmup_ratio: float = 0.05,     # 5% de steps
    min_lr: float = 0.0,
    label_smoothing: float = 0.1,
    print_every: int = 100,
    save_path: str = "best_swinvit_cifar100.pt",
    resume_path: str | None = None,):

    model.to(device)
    # Optimizer (pro: param groups sin WD en norm/bias)
    param_groups = build_param_groups_no_wd(model, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)

    # Scheduler warmup + cosine por step
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = WarmupCosineLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=min_lr)

    # AMP scaler: SOLO FP16
    scaler = None
    if use_amp and autocast_dtype.lower() in ("fp16", "float16"):
        scaler = make_grad_scaler(device=device, enabled=True)

    start_epoch = 0
    best_val_top1 = -float("inf")

    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0))
        best_val_top1 = float(ckpt.get("best_top1", best_val_top1))
        print(f"Resumed from {resume_path} at epoch {start_epoch} | best_top1 {best_val_top1:.2f}%")

    history = {
        "train_loss": [], "train_top1": [], "train_top3": [], "train_top5": [],
        "val_loss": [], "val_top1": [], "val_top3": [], "val_top5": [],}

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
            print_every=print_every,)

        history["train_loss"].append(tr_loss)
        history["train_top1"].append(tr_m["top1"])
        history["train_top3"].append(tr_m["top3"])
        history["train_top5"].append(tr_m["top5"])

        print(f"[Train] loss {tr_loss:.4f} | top1 {tr_m['top1']:.2f}% | top3 {tr_m['top3']:.2f}% | top5 {tr_m['top5']:.2f}%")

        if val_loader is not None:
            va_loss, va_m = evaluate_one_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                autocast_dtype=autocast_dtype,
                use_amp=use_amp,
                label_smoothing=label_smoothing,)

            history["val_loss"].append(va_loss)
            history["val_top1"].append(va_m["top1"])
            history["val_top3"].append(va_m["top3"])
            history["val_top5"].append(va_m["top5"])

            print(f"[Val]   loss {va_loss:.4f} | top1 {va_m['top1']:.2f}% | top3 {va_m['top3']:.2f}% | top5 {va_m['top5']:.2f}%")

            if va_m["top1"] > best_val_top1:
                best_val_top1 = va_m["top1"]
                save_checkpoint(
                    save_path, model, optimizer, scheduler, scaler,
                    epoch=epoch, best_top1=best_val_top1,
                    extra={"autocast_dtype": autocast_dtype, "use_amp": use_amp},)
                print(f" Best saved to {save_path} (val top1 {best_val_top1:.2f}%)")

        dt = time.time() - t_epoch
        print(f"Epoch time: {dt/60:.2f} min")

    return history
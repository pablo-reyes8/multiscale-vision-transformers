#!/usr/bin/env python
import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.data_utils import describe_loader
from data.load_cifrar100 import get_cifar100_dataloaders
from model.MaxViT import MaxViT
from model_configurations import (
    maxvit_cifar100_tiny,
    maxvit_cifar100_small,
    maxvit_cifar100_base,
)
from training.amp import seed_everything
from training.train_MaxViT import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train MaxViT on CIFAR-100")

    parser.add_argument("--variant", choices=["tiny", "small", "base"], default="tiny")
    parser.add_argument("--stem-type", choices=["A", "B"], default="A")
    parser.add_argument("--drop-path-rate", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--mix-prob", type=float, default=1.0)

    parser.add_argument("--ra-num-ops", type=int, default=2)
    parser.add_argument("--ra-magnitude", type=int, default=7)
    parser.add_argument("--random-erasing-p", type=float, default=0.25)

    parser.add_argument("--amp", dest="use_amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", default=None)
    parser.add_argument("--autocast-dtype", type=str, default="fp16")

    parser.add_argument("--use-ema", action="store_true", default=False)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-warmup-epochs", type=int, default=5)

    parser.add_argument("--channels-last", action="store_true", default=False)
    parser.add_argument("--print-every", type=int, default=100)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-name", type=str, default="best_model.pt")
    parser.add_argument("--last-name", type=str, default="last_model.pt")
    parser.add_argument("--describe", action="store_true", default=False)

    return parser.parse_args()


def build_config(variant: str, stem_type: str, drop_path_rate: float):
    variants = {
        "tiny": maxvit_cifar100_tiny,
        "small": maxvit_cifar100_small,
        "base": maxvit_cifar100_base,
    }
    return variants[variant](stem_type=stem_type, drop_path_rate=drop_path_rate)


def main():
    args = parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"

    if args.use_amp is None:
        args.use_amp = device.startswith("cuda")

    grad_clip_norm = args.grad_clip_norm
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        grad_clip_norm = None

    if args.seed is not None:
        seed_everything(args.seed)

    cfg = build_config(args.variant, args.stem_type, args.drop_path_rate)

    train_loader, val_loader, test_loader = get_cifar100_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        val_split=args.val_split,
        pin_memory=args.pin_memory,
        ra_num_ops=args.ra_num_ops,
        ra_magnitude=args.ra_magnitude,
        random_erasing_p=args.random_erasing_p,
    )

    if args.describe:
        describe_loader(train_loader, name="train")
        if val_loader is not None:
            describe_loader(val_loader, name="val")
        describe_loader(test_loader, name="test")

    model = MaxViT(cfg)

    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / f"maxvit_{args.variant}"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / args.save_name
    last_path = output_dir / args.last_name

    if (args.mixup_alpha > 0.0 or args.cutmix_alpha > 0.0) and args.label_smoothing > 0.0:
        print("Note: label_smoothing may be redundant when using mixup/cutmix.")

    train_model(
        model=model,
        train_loader=train_loader,
        epochs=args.epochs,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        autocast_dtype=args.autocast_dtype,
        use_amp=args.use_amp,
        grad_clip_norm=grad_clip_norm,
        warmup_ratio=args.warmup_ratio,
        min_lr=args.min_lr,
        label_smoothing=args.label_smoothing,
        print_every=args.print_every,
        save_path=str(save_path),
        last_path=str(last_path),
        resume_path=args.resume,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        ema_warmup_epochs=args.ema_warmup_epochs,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mix_prob=args.mix_prob,
        num_classes=cfg.num_classes,
        channels_last=args.channels_last,
    )


if __name__ == "__main__":
    main()

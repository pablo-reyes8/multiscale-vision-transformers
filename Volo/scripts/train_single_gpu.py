import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.VOLO import VOLOClassifier
from data.load_data_ddp import get_cifar100_datasets
from training.Train_VOLO import train_model


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    return tuple(int(v) for v in items)


def _select_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        return "cpu"
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VOLO single-GPU training.")

    # data
    parser.add_argument("--data-dir", default="./data/cifar100")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)

    # model
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--hierarchical", action="store_true")
    parser.add_argument("--downsample-kind", choices=["map", "token"], default="map")
    parser.add_argument("--pooling", choices=["mean", "cls", "cli"], default="cls")
    parser.add_argument("--embed-dim", type=int, default=320)
    parser.add_argument("--outlooker-depth", type=int, default=5)
    parser.add_argument("--outlooker-heads", type=int, default=10)
    parser.add_argument("--transformer-depth", type=int, default=10)
    parser.add_argument("--transformer-heads", type=int, default=10)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--attn-dropout", type=float, default=0.05)
    parser.add_argument("--drop-path-rate", type=float, default=0.20)
    parser.add_argument("--cls-attn-depth", type=int, default=2)
    parser.add_argument("--use-pos-embed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-cls-pos", action=argparse.BooleanOptionalAction, default=True)

    # hierarchical config
    parser.add_argument("--dims", default="192,256,384")
    parser.add_argument("--outlooker-depths", default="2,2,0")
    parser.add_argument("--outlooker-heads-list", default="6,8,12")
    parser.add_argument("--transformer-depths", default="0,2,2")
    parser.add_argument("--transformer-heads-list", default="6,8,12")

    # training
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--autocast-dtype", default="fp16")
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--save-path", default="best_model.pt")
    parser.add_argument("--last-path", default="last_model.pt")
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--mix-prob", type=float, default=1.0)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stop-metric", choices=["top1", "loss"], default="top1")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-require-monotonic", action=argparse.BooleanOptionalAction, default=False)

    return parser


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = build_parser().parse_args()

    device = _select_device(args.device)

    train_ds, val_ds, _ = get_cifar100_datasets(
        data_dir=args.data_dir,
        val_split=args.val_split,
        img_size=args.img_size,
        seed=7,
        ddp_safe_download=False,)

    persistent_workers = args.persistent_workers and args.num_workers > 0
    train_kwargs = {}
    if args.num_workers > 0:
        train_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=persistent_workers,
        **train_kwargs,)

    val_loader = None
    if val_ds is not None:
        val_kwargs = {}
        if args.num_workers > 0:
            val_kwargs["prefetch_factor"] = args.prefetch_factor
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=persistent_workers,
            **val_kwargs,)

    if args.hierarchical:
        dims = _parse_int_tuple(args.dims)
        outlooker_depths = _parse_int_tuple(args.outlooker_depths)
        outlooker_heads_list = _parse_int_tuple(args.outlooker_heads_list)
        transformer_depths = _parse_int_tuple(args.transformer_depths)
        transformer_heads_list = _parse_int_tuple(args.transformer_heads_list)
    else:
        dims = outlooker_depths = outlooker_heads_list = transformer_depths = transformer_heads_list = None

    model = VOLOClassifier(
        num_classes=args.num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        hierarchical=args.hierarchical,
        downsample_kind=args.downsample_kind,
        embed_dim=args.embed_dim,
        outlooker_depth=args.outlooker_depth,
        outlooker_heads=args.outlooker_heads,
        transformer_depth=args.transformer_depth,
        transformer_heads=args.transformer_heads,
        dims=dims or (192, 256, 384),
        outlooker_depths=outlooker_depths or (2, 2, 0),
        outlooker_heads_list=outlooker_heads_list or (6, 8, 12),
        transformer_depths=transformer_depths or (0, 2, 2),
        transformer_heads_list=transformer_heads_list or (6, 8, 12),
        kernel_size=args.kernel_size,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        drop_path_rate=args.drop_path_rate,
        pooling=args.pooling,
        cls_attn_depth=args.cls_attn_depth,
        use_pos_embed=args.use_pos_embed,
        use_cls_pos=args.use_cls_pos,).to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        device=str(device),
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_amp=args.use_amp,
        autocast_dtype=args.autocast_dtype,
        grad_clip_norm=args.grad_clip_norm,
        warmup_ratio=args.warmup_ratio,
        min_lr=args.min_lr,
        label_smoothing=args.label_smoothing,
        print_every=args.print_every,
        save_path=args.save_path,
        last_path=args.last_path,
        resume_path=args.resume_path,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mix_prob=args.mix_prob,
        num_classes=args.num_classes,
        channels_last=args.channels_last,
        early_stop=args.early_stop,
        early_stop_metric=args.early_stop_metric,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_require_monotonic=args.early_stop_require_monotonic,)


if __name__ == "__main__":
    main()

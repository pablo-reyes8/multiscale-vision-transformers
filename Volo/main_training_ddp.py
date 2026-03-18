import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.VOLO import VOLOClassifier
from data.dataset_zoo import available_dataset_names, get_classification_datasets, get_dataset_info
from training.Train_VOLO import train_model


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    return tuple(int(v) for v in items)


def _maybe_set_threads(num_threads: int | None, num_interop_threads: int | None) -> None:
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    if num_interop_threads is not None:
        torch.set_num_interop_threads(num_interop_threads)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VOLO DDP training (torchrun compatible).")

    # data
    parser.add_argument("--dataset", default="cifar100", choices=available_dataset_names())
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--ddp-safe-download", action=argparse.BooleanOptionalAction, default=True)

    # model
    parser.add_argument("--num-classes", type=int, default=None)
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

    # ddp/runtime
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--find-unused-parameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--num-interop-threads", type=int, default=1)
    return parser


def setup_ddp(backend: str):
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def main(args: argparse.Namespace | None = None):
    if args is None:
        args = build_parser().parse_args()

    _maybe_set_threads(args.num_threads, args.num_interop_threads)

    local_rank = setup_ddp(args.backend)
    device = torch.device(f"cuda:{local_rank}")

    dataset_info = get_dataset_info(args.dataset)
    num_classes = args.num_classes if args.num_classes is not None else dataset_info["num_classes"]

    train_ds, val_ds, _ = get_classification_datasets(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        val_split=args.val_split,
        ra_num_ops=2,
        ra_magnitude=7,
        random_erasing_p=0.25,
        img_size=args.img_size,
        seed=7,
        ddp_safe_download=args.ddp_safe_download,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    persistent_workers = args.persistent_workers and args.num_workers > 0
    train_kwargs = {}
    if args.num_workers > 0:
        train_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=persistent_workers,
        **train_kwargs,
    )

    val_loader = None
    if val_ds is not None:
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
        val_kwargs = {}
        if args.num_workers > 0:
            val_kwargs["prefetch_factor"] = args.prefetch_factor
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=persistent_workers,
            **val_kwargs,
        )

    if args.hierarchical:
        dims = _parse_int_tuple(args.dims)
        outlooker_depths = _parse_int_tuple(args.outlooker_depths)
        outlooker_heads_list = _parse_int_tuple(args.outlooker_heads_list)
        transformer_depths = _parse_int_tuple(args.transformer_depths)
        transformer_heads_list = _parse_int_tuple(args.transformer_heads_list)
    else:
        dims = outlooker_depths = outlooker_heads_list = transformer_depths = transformer_heads_list = None

    model = VOLOClassifier(
        num_classes=num_classes,
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
        use_cls_pos=args.use_cls_pos,
    ).to(device)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=args.find_unused_parameters,
    )

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
        num_classes=num_classes,
        channels_last=args.channels_last,
        early_stop=args.early_stop,
        early_stop_metric=args.early_stop_metric,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_require_monotonic=args.early_stop_require_monotonic,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

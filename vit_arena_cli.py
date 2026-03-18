import argparse
import json

from vit_arena import (
    AUGMENT_PRESETS,
    ARENA_DEFAULTS,
    build_dry_run_report,
    compare_models,
    describe_models,
    describe_tips,
    resolve_model_names,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified comparison arena for the ViT-family subprojects in this repository."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["vit", "hierarchical_vit", "swin"],
        help="Models to compare. Use 'all' to run every preset in vit_arena_presets.py.",
    )
    parser.add_argument(
        "--augment",
        choices=sorted(AUGMENT_PRESETS.keys()),
        default=ARENA_DEFAULTS["augment"],
        help="Shared train/eval preprocessing preset for every requested model.",
    )
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory used to download/store CIFAR-100.")
    parser.add_argument("--output-dir", type=str, default=None, help="Where the arena stores checkpoints and summaries.")
    parser.add_argument("--img-size", type=int, default=ARENA_DEFAULTS["img_size"], help="Input image size.")
    parser.add_argument("--num-classes", type=int, default=ARENA_DEFAULTS["num_classes"], help="Number of classes.")
    parser.add_argument("--batch-size", type=int, default=ARENA_DEFAULTS["batch_size"], help="Training batch size.")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=ARENA_DEFAULTS["eval_batch_size"],
        help="Evaluation batch size for val/test.",
    )
    parser.add_argument("--epochs", type=int, default=ARENA_DEFAULTS["epochs"], help="Epochs per model.")
    parser.add_argument("--lr", type=float, default=ARENA_DEFAULTS["lr"], help="Shared learning rate.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=ARENA_DEFAULTS["weight_decay"],
        help="Shared weight decay.",
    )
    parser.add_argument("--val-split", type=float, default=ARENA_DEFAULTS["val_split"], help="Validation split.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default=None, help="Device identifier, e.g. cpu or cuda.")
    parser.add_argument("--autocast-dtype", type=str, default="fp16", help="AMP dtype (fp16 or bf16).")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm. Use 0 to disable.")
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=ARENA_DEFAULTS["warmup_ratio"],
        help="Warmup ratio for cosine scheduling.",
    )
    parser.add_argument("--min-lr", type=float, default=ARENA_DEFAULTS["min_lr"], help="Minimum cosine LR.")
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=ARENA_DEFAULTS["label_smoothing"],
        help="Shared label smoothing value.",
    )
    parser.add_argument("--print-every", type=int, default=100, help="Log every N training steps.")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional cap for train batches per epoch, useful for smoke runs.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Optional cap for val/test batches, useful for smoke runs.",
    )
    parser.add_argument("--seed", type=int, default=ARENA_DEFAULTS["seed"], help="Global seed.")

    parser.add_argument("--list-models", action="store_true", help="Print available model presets and exit.")
    parser.add_argument("--show-tips", action="store_true", help="Print preset tips and exit.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the requested models and print parameter counts without touching the dataset.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="When used with --dry-run, print the report as JSON instead of a readable text block.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_models:
        print(describe_models())
        return

    if args.show_tips:
        print(describe_tips())
        return

    model_names = resolve_model_names(args.models)

    if args.dry_run:
        report = build_dry_run_report(
            model_names=model_names,
            img_size=args.img_size,
            num_classes=args.num_classes,
        )
        if args.json:
            print(json.dumps(report, indent=2))
            return

        for row in report:
            print(
                f"{row['model']}: "
                f"family={row['family']} | "
                f"params={row['params']:,} | "
                f"trainable={row['trainable_params']:,}"
            )
        return

    summary = compare_models(
        model_names=model_names,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        augment=args.augment,
        img_size=args.img_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        num_workers=args.num_workers,
        device=args.device,
        autocast_dtype=args.autocast_dtype,
        use_amp=not args.no_amp,
        grad_clip_norm=(None if args.grad_clip_norm == 0 else args.grad_clip_norm),
        warmup_ratio=args.warmup_ratio,
        min_lr=args.min_lr,
        label_smoothing=args.label_smoothing,
        print_every=args.print_every,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        seed=args.seed,
    )

    print("\nComparison summary")
    print(summary["table"])
    print(f"\nArtifacts saved to: {summary['output_dir']}")


if __name__ == "__main__":
    main()

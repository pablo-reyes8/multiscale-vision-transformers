"""
Command-line utilities to train and evaluate the Swin Transformer.
"""

import argparse
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.cifrar100 import get_cifar100_dataloaders
from model.swin_vision_transformer import SwinTransformer
from training.one_epoch import evaluate_one_epoch
from training.train_vit import train_swinvit
from validation.show_predictions import show_predictions_grid

from validation.top_accuracy_classes import (
    collect_predictions,
    per_class_accuracy,
    print_best_and_worst_classes)

from validation.tsne_classes import collect_features, tsne_plot
from validation.url_predict import predict_from_url, show_url_prediction_pair
from validation.utils import resolve_class_names


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument("--img-size", type=int, default=32, help="Input image size used for the Swin config.")
    parser.add_argument("--patch-size", type=int, default=4, help="Patch size for the stem conv.")
    parser.add_argument("--num-classes", type=int, default=100, help="Number of classification targets.")
    parser.add_argument("--embed-dim", type=int, default=96, help="Base embedding dimension.")
    parser.add_argument("--depths", nargs="+", type=int, help="Blocks per stage (4 values).")
    parser.add_argument("--num-heads", nargs="+", type=int, help="Attention heads per stage (4 values).")
    parser.add_argument("--window-size", type=int, default=7, help="Window size for W-MSA/SW-MSA.")
    parser.add_argument("--mlp-ratio", type=float, default=4.0, help="Hidden expansion for MLPs.")
    parser.add_argument("--drop-rate", type=float, default=0.0, help="Dropout after patch embedding.")
    parser.add_argument("--attn-dropout", type=float, default=0.0, help="Dropout inside attention.")
    parser.add_argument("--proj-dropout", type=float, default=0.0, help="Dropout after attention projection.")
    parser.add_argument("--mlp-dropout", type=float, default=0.0, help="Dropout inside MLPs.")
    parser.add_argument("--drop-path-rate", type=float, default=0.1, help="Stochastic depth rate.")
    parser.add_argument("--no-rel-pos-bias", action="store_true", help="Disable relative position bias.")


def build_model(args) -> SwinTransformer:
    depths = tuple(args.depths) if args.depths else (2, 2, 6, 2)
    num_heads = tuple(args.num_heads) if args.num_heads else (3, 6, 12, 24)

    return SwinTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.drop_rate,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        mlp_dropout=args.mlp_dropout,
        drop_path_rate=args.drop_path_rate,
        use_rel_pos_bias=not args.no_rel_pos_bias,)


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)


def run_train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader = get_cifar100_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        val_split=args.val_split,
        pin_memory=device.startswith("cuda"),)

    model = build_model(args)

    ckpt_path = Path(args.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    history = train_swinvit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        autocast_dtype=args.autocast_dtype,
        use_amp=not args.no_amp,
        grad_clip_norm=args.grad_clip_norm,
        warmup_ratio=args.warmup_ratio,
        min_lr=args.min_lr,
        label_smoothing=args.label_smoothing,
        print_every=args.print_every,
        save_path=str(ckpt_path),
        resume_path=args.resume,)

    if args.evaluate_test:
        test_loss, test_metrics = evaluate_one_epoch(
            model=model,
            dataloader=test_loader,
            device=device,
            autocast_dtype=args.autocast_dtype,
            use_amp=not args.no_amp,
            label_smoothing=args.label_smoothing,)

        print(
            f"[Test] Loss: {test_loss:.4f} | "
            f"Top-1: {test_metrics['top1']:.2f}% | "
            f"Top-3: {test_metrics['top3']:.2f}% | "
            f"Top-5: {test_metrics['top5']:.2f}%")

    if args.save_final:
        final_path = Path(args.save_final)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final_path)
        print(f"Final model checkpoint saved to {final_path}")

    return history


def run_eval(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    _, _, test_loader = get_cifar100_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        val_split=0.0,
        pin_memory=device.startswith("cuda"),)

    model = build_model(args)
    _load_checkpoint(model, args.checkpoint, device=device)

    test_loss, test_metrics = evaluate_one_epoch(
        model=model,
        dataloader=test_loader,
        device=device,
        autocast_dtype=args.autocast_dtype,
        use_amp=not args.no_amp,
        label_smoothing=args.label_smoothing,)

    print(
        f"[Eval] Loss: {test_loss:.4f} | "
        f"Top-1: {test_metrics['top1']:.2f}% | "
        f"Top-3: {test_metrics['top3']:.2f}% | "
        f"Top-5: {test_metrics['top5']:.2f}%")

    return test_loss, test_metrics


def run_validate(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    def _get_test_loader():
        _, _, test_loader = get_cifar100_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            num_workers=args.num_workers,
            val_split=0.0,
            pin_memory=device.startswith("cuda"),)
        return test_loader

    model = build_model(args)
    _load_checkpoint(model, args.checkpoint, device=device)
    model.to(device)

    task = args.task

    if task == "test":
        test_loader = _get_test_loader()
        test_loss, test_metrics = evaluate_one_epoch(
            model=model,
            dataloader=test_loader,
            device=device,
            autocast_dtype=args.autocast_dtype,
            use_amp=not args.no_amp,
            label_smoothing=args.label_smoothing,)
        print(
            f"[Test] Loss: {test_loss:.4f} | "
            f"Top-1: {test_metrics['top1']:.2f}% | "
            f"Top-3: {test_metrics['top3']:.2f}% | "
            f"Top-5: {test_metrics['top5']:.2f}%")
        return test_loss, test_metrics

    if task in ("grid", "misclassified"):
        test_loader = _get_test_loader()
        class_names = resolve_class_names(test_loader, data_dir=args.data_dir)
        show_predictions_grid(
            model=model,
            dataloader=test_loader,
            class_names=class_names,
            device=device,
            autocast_dtype=args.autocast_dtype,
            use_amp=not args.no_amp,
            n=args.num_images,
            only_misclassified=(task == "misclassified" or args.only_misclassified),)
        return None

    if task == "per-class":
        test_loader = _get_test_loader()
        class_names = resolve_class_names(test_loader, data_dir=args.data_dir)
        _, targets, preds = collect_predictions(
            model=model,
            dataloader=test_loader,
            device=device,
            autocast_dtype=args.autocast_dtype,
            use_amp=not args.no_amp)
        acc, counts = per_class_accuracy(preds, targets, num_classes=len(class_names))
        print_best_and_worst_classes(acc, counts, class_names, k=args.topk_classes)
        return acc, counts

    if task == "tsne":
        test_loader = _get_test_loader()
        class_names = resolve_class_names(test_loader, data_dir=args.data_dir)
        features, labels = collect_features(
            model=model,
            dataloader=test_loader,
            device=device,
            max_samples=args.tsne_max_samples,
            autocast_dtype=args.autocast_dtype,
            use_amp=not args.no_amp)
        tsne_plot(
            features=features,
            labels=labels,
            class_names=class_names,
            num_points=args.tsne_max_samples,
            perplexity=args.tsne_perplexity,
            n_iter=args.tsne_iter,
            random_state=args.seed,
            subset_classes=args.tsne_subset_classes)
        return None

    if task == "url":
        if not args.url:
            raise ValueError("url task requires --url")
        class_names = resolve_class_names(data_dir=args.data_dir)
        topk_idxs, topk_vals, img_pil, x_tensor = predict_from_url(
            model=model,
            url=args.url,
            class_names=class_names,
            device=device,
            autocast_dtype=args.autocast_dtype,
            use_amp=not args.no_amp,
            topk=args.url_topk)
        if args.show:
            show_url_prediction_pair(
                img_pil=img_pil,
                x_tensor_norm=x_tensor,
                topk_idxs=topk_idxs,
                topk_vals=topk_vals,
                class_names=class_names)
        return None

    raise ValueError(f"Unknown validation task: {task}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or evaluate SwinViT on CIFAR-100.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train SwinTransformer.")
    train_parser.add_argument("--data-dir", type=str, default="./data", help="Where CIFAR-100 is stored/downloaded.")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    train_parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    train_parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay for AdamW.")
    train_parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of training data used for validation.")
    train_parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    train_parser.add_argument("--device", type=str, default=None, help="Device identifier (cuda or cpu).")
    train_parser.add_argument("--autocast-dtype", type=str, default="fp16", help="AMP dtype (fp16 or bf16).")
    train_parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    train_parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Max grad norm (set 0 to disable).")
    train_parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio for cosine LR.")
    train_parser.add_argument("--min-lr", type=float, default=0.0, help="Minimum LR after cosine decay.")
    train_parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing for CE.")
    train_parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best_swinvit_cifar100.pt", help="Path to save the best checkpoint.")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from a checkpoint path.")
    train_parser.add_argument("--save-final", type=str, help="Optional path to save final weights.")
    train_parser.add_argument("--evaluate-test", action="store_true", help="Run a test pass after training.")
    train_parser.add_argument("--print-every", type=int, default=100, help="Logging frequency during training.")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    add_model_args(train_parser)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint on CIFAR-100 test set.")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file to load.")
    eval_parser.add_argument("--data-dir", type=str, default="./data", help="Where CIFAR-100 is stored/downloaded.")
    eval_parser.add_argument("--batch-size", type=int, default=256, help="Batch size for evaluation.")
    eval_parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    eval_parser.add_argument("--device", type=str, default=None, help="Device identifier (cuda or cpu).")
    eval_parser.add_argument("--autocast-dtype", type=str, default="fp16", help="AMP dtype (fp16 or bf16).")
    eval_parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision during eval.")
    eval_parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for CE.")
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    add_model_args(eval_parser)

    validate_parser = subparsers.add_parser("validate", help="Run validation utilities for a trained model.")
    validate_parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=("test", "grid", "misclassified", "per-class", "tsne", "url"),
        help="Validation task to run.")
    validate_parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file to load.")
    validate_parser.add_argument("--data-dir", type=str, default="./data", help="Where CIFAR-100 is stored/downloaded.")
    validate_parser.add_argument("--batch-size", type=int, default=256, help="Batch size for evaluation.")
    validate_parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    validate_parser.add_argument("--device", type=str, default=None, help="Device identifier (cuda or cpu).")
    validate_parser.add_argument("--autocast-dtype", type=str, default="fp16", help="AMP dtype (fp16 or bf16).")
    validate_parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    validate_parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for test.")
    validate_parser.add_argument("--num-images", type=int, default=8, help="Number of images for grid display.")
    validate_parser.add_argument("--only-misclassified", action="store_true", help="Show only mistakes in grid task.")
    validate_parser.add_argument("--topk-classes", type=int, default=10, help="Top/bottom classes to print.")
    validate_parser.add_argument("--tsne-max-samples", type=int, default=2000, help="Max samples for t-SNE.")
    validate_parser.add_argument("--tsne-perplexity", type=int, default=30, help="t-SNE perplexity.")
    validate_parser.add_argument("--tsne-iter", type=int, default=1000, help="t-SNE iterations.")
    validate_parser.add_argument("--tsne-subset-classes", nargs="+", type=int, default=None, help="Subset class ids.")
    validate_parser.add_argument("--url", type=str, default=None, help="Image URL for the url task.")
    validate_parser.add_argument("--url-topk", type=int, default=5, help="Top-k predictions for url task.")
    validate_parser.add_argument("--show", action="store_true", help="Show prediction figure for url task.")
    validate_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    add_model_args(validate_parser)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "validate":
        run_validate(args)
    else:
        parser.error("Unknown command.")


if __name__ == "__main__":
    main()

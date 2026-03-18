"""
Command-line utilities to train and evaluate the original Vision Transformer.
"""

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.cifrar100 import get_cifar100_dataloaders
from model.vision_transformer import VisionTransformer
from training.one_epoch import evaluate_one_epoch
from training.train_vit import train_vit


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument("--img-size", type=int, default=32, help="Input image size.")
    parser.add_argument("--patch-size", type=int, default=4, help="Patch size.")
    parser.add_argument("--num-classes", type=int, default=100, help="Number of classification targets.")
    parser.add_argument("--embed-dim", type=int, default=192, help="Token embedding dimension.")
    parser.add_argument("--depth", type=int, default=6, help="Number of transformer encoder blocks.")
    parser.add_argument("--num-heads", type=int, default=3, help="Attention heads per block.")
    parser.add_argument("--mlp-ratio", type=float, default=4.0, help="Hidden expansion for feed-forward layers.")
    parser.add_argument("--drop-rate", type=float, default=0.0, help="Dropout after positional embeddings.")
    parser.add_argument("--attn-dropout", type=float, default=0.0, help="Dropout on attention probabilities.")
    parser.add_argument("--proj-dropout", type=float, default=0.0, help="Dropout after attention projection.")
    parser.add_argument("--mlp-dropout", type=float, default=0.1, help="Dropout inside the MLP.")
    parser.add_argument("--drop-path-rate", type=float, default=0.1, help="Stochastic depth rate.")
    parser.add_argument("--patch-norm", action="store_true", help="Apply LayerNorm after patch embedding.")
    parser.add_argument("--no-qkv-bias", action="store_true", help="Disable bias in QKV projections.")


def build_model(args) -> VisionTransformer:
    return VisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=not args.no_qkv_bias,
        patch_norm=args.patch_norm,
        drop_rate=args.drop_rate,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        mlp_dropout=args.mlp_dropout,
        drop_path_rate=args.drop_path_rate,
    )


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
        pin_memory=device.startswith("cuda"),
    )

    model = build_model(args)

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = train_vit(
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
        save_path=str(checkpoint_path),
        resume_path=args.resume,
    )

    if args.evaluate_test:
        test_loss, test_metrics = evaluate_one_epoch(
            model=model,
            dataloader=test_loader,
            device=device,
            autocast_dtype=args.autocast_dtype,
            use_amp=not args.no_amp,
            label_smoothing=args.label_smoothing,
        )

        print(
            f"[Test] Loss: {test_loss:.4f} | "
            f"Top-1: {test_metrics['top1']:.2f}% | "
            f"Top-3: {test_metrics['top3']:.2f}% | "
            f"Top-5: {test_metrics['top5']:.2f}%"
        )

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
        pin_memory=device.startswith("cuda"),
    )

    model = build_model(args)
    _load_checkpoint(model, args.checkpoint, device=device)

    test_loss, test_metrics = evaluate_one_epoch(
        model=model,
        dataloader=test_loader,
        device=device,
        autocast_dtype=args.autocast_dtype,
        use_amp=not args.no_amp,
        label_smoothing=args.label_smoothing,
    )

    print(
        f"[Eval] Loss: {test_loss:.4f} | "
        f"Top-1: {test_metrics['top1']:.2f}% | "
        f"Top-3: {test_metrics['top3']:.2f}% | "
        f"Top-5: {test_metrics['top5']:.2f}%"
    )

    return test_loss, test_metrics


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate ViT on CIFAR-100.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train VisionTransformer.")
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
    train_parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best_vit_cifar100.pt", help="Path to save the best checkpoint.")
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
    eval_parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for evaluation.")
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    add_model_args(eval_parser)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "eval":
        run_eval(args)
    else:
        parser.error("Unknown command.")


if __name__ == "__main__":
    main()

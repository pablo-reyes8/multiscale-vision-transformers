"""
Command-line utilities to train and evaluate the Hierarchical ViT model.
"""

import argparse
import sys
from pathlib import Path

import torch

# Ensure local imports work when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_cifrar import get_cifar100_dataloaders
from model.hierarchical_vit import HierarchicalViT
from training.train_one_epoch import evaluate_one_epoch
from training.train_vit import train_vit


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument("--img-size", type=int, default=32, help="Input image size.")
    parser.add_argument("--patch-size", type=int, default=4, help="Patch size for the stem conv.")
    parser.add_argument("--num-classes", type=int, default=100, help="Number of classification targets.")
    parser.add_argument("--embed-dims", nargs="+", type=int, help="Embedding dimensions per stage.")
    parser.add_argument("--depths", nargs="+", type=int, help="Number of transformer blocks per stage.")
    parser.add_argument("--num-heads", nargs="+", type=int, help="Number of attention heads per stage.")
    parser.add_argument("--mlp-ratio", type=float, default=4.0, help="Hidden expansion for feed-forward layers.")
    parser.add_argument("--attn-dropout", type=float, default=0.0, help="Dropout inside attention.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout after attention/MLP.")


def build_model(args) -> HierarchicalViT:
    embed_dims = tuple(args.embed_dims) if args.embed_dims else (192, 384, 576)
    depths = tuple(args.depths) if args.depths else (2, 2, 4)
    num_heads = tuple(args.num_heads) if args.num_heads else (3, 6, 9)

    return HierarchicalViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=args.num_classes,
        embed_dims=embed_dims,
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=args.mlp_ratio,
        attn_dropout=args.attn_dropout,
        dropout=args.dropout,)


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

    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        save_path = str(ckpt_path)
    else:
        save_path = None

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
        use_grad_scaler=not args.no_amp,
        save_best=save_path is not None,
        save_path=save_path,
        scheduler=None,
        print_every=args.print_every,)

    if args.evaluate_test:
        test_loss, test_metrics = evaluate_one_epoch(
            model=model,
            dataloader=test_loader,
            device=device,
            autocast_dtype=args.autocast_dtype,
            use_amp=not args.no_amp,)

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
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    test_loss, test_metrics = evaluate_one_epoch(
        model=model,
        dataloader=test_loader,
        device=device,
        autocast_dtype=args.autocast_dtype,
        use_amp=not args.no_amp,)

    print(
        f"[Eval] Loss: {test_loss:.4f} | "
        f"Top-1: {test_metrics['top1']:.2f}% | "
        f"Top-3: {test_metrics['top3']:.2f}% | "
        f"Top-5: {test_metrics['top5']:.2f}%")

    return test_loss, test_metrics


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or evaluate the Hierarchical ViT on CIFAR-100.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train HierarchicalViT.")
    train_parser.add_argument("--data-dir", type=str, default="./data", help="Where CIFAR-100 is stored/downloaded.")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    train_parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    train_parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of training data used for validation.")
    train_parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    train_parser.add_argument("--device", type=str, default=None, help="Device identifier (cuda or cpu).")
    train_parser.add_argument("--autocast-dtype", type=str, default="fp16", help="AMP dtype (fp16 or bf16).")
    train_parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    train_parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best_hierarchical_vit.pth", help="Path to save the best validation checkpoint.")
    train_parser.add_argument("--save-final", type=str, help="Optional path to always save the final weights.")
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

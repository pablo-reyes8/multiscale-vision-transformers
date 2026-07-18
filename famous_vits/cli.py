"""Command-line interface for the unified ViT library."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from .config import load_pipeline_config
from .data import available_dataset_names, get_classification_dataloaders, get_dataset_info
from .factory import list_models, model_info
from .orchestrator import ViTOrchestrator
from .pipeline import run_pipeline

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("The value must be a JSON object.")
    return parsed


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=list_models(), required=True)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--in-chans", type=int, default=3)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--model-kwargs", type=_json_object, default={}, metavar="JSON")
    parser.add_argument("--device", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="famous-vits", description="Unified local ViT model library"
    )
    parser.add_argument("--version", action="version", version="famous-vits 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List registered model presets")
    info_parser = subparsers.add_parser("info", help="Show model metadata")
    info_parser.add_argument("model", choices=list_models())

    train_parser = subparsers.add_parser("train", help="Train on a built-in or synthetic dataset")
    _add_model_args(train_parser)
    train_parser.add_argument("--dataset", choices=available_dataset_names(), default="cifar100")
    train_parser.add_argument("--data-dir", default="./data")
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--eval-batch-size", type=int, default=256)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--val-split", type=float, default=0.1)
    train_parser.add_argument(
        "--optimizer", choices=("adam", "adamw", "sgd", "rmsprop"), default="adamw"
    )
    train_parser.add_argument("--lr", type=float, default=5e-4)
    train_parser.add_argument("--optimizer-kwargs", type=_json_object, default={}, metavar="JSON")
    train_parser.add_argument("--output", type=Path, default=None)
    train_parser.add_argument("--smoke-test", action="store_true")

    infer_parser = subparsers.add_parser("infer", help="Run image or synthetic inference")
    infer_parser.add_argument("--checkpoint", type=Path, required=True)
    infer_parser.add_argument("--input", type=Path)
    infer_parser.add_argument("--top-k", type=int, default=5)
    infer_parser.add_argument("--device", default=None)
    infer_parser.add_argument("--output", type=Path)
    infer_parser.add_argument("--smoke-test", action="store_true")

    analyze_parser = subparsers.add_parser(
        "analyze", help="Evaluate, calibrate and build a classification report"
    )
    analyze_parser.add_argument("--checkpoint", type=Path, required=True)
    analyze_parser.add_argument("--dataset", choices=available_dataset_names(), default="cifar100")
    analyze_parser.add_argument("--data-dir", default="./data")
    analyze_parser.add_argument("--batch-size", type=int, default=256)
    analyze_parser.add_argument("--num-workers", type=int, default=2)
    analyze_parser.add_argument("--device", default=None)
    analyze_parser.add_argument("--num-bins", type=int, default=15)
    analyze_parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))

    run_parser = subparsers.add_parser("run", help="Run a train, arena or analysis YAML pipeline")
    run_parser.add_argument("--config", type=Path, required=True)

    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Validate a YAML pipeline and print its resolved defaults",
    )
    validate_parser.add_argument("--config", type=Path, required=True)
    return parser


def _synthetic_loaders(num_classes: int, in_chans: int, img_size: int, batch_size: int):
    sample_count = 4
    images = torch.randn(sample_count, in_chans, img_size, img_size)
    targets = torch.arange(sample_count) % num_classes
    loader = DataLoader(
        TensorDataset(images, targets),
        batch_size=min(batch_size, sample_count),
    )
    return loader, loader


def train_command(args: argparse.Namespace) -> int:
    if args.smoke_test:
        num_classes = args.num_classes or 2
        train_loader, val_loader = _synthetic_loaders(
            num_classes, args.in_chans, args.img_size, args.batch_size
        )
        class_names = [str(index) for index in range(num_classes)]
    else:
        if args.in_chans != 3:
            raise ValueError(
                "Built-in image datasets emit RGB tensors; use --in-chans 3 or a custom loader through the Python API."
            )
        info = get_dataset_info(args.dataset)
        num_classes = args.num_classes or info["num_classes"]
        if num_classes != info["num_classes"]:
            raise ValueError("--num-classes must match the selected built-in dataset.")
        train_loader, val_loader, _ = get_classification_dataloaders(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            data_dir=args.data_dir,
            num_workers=args.num_workers,
            val_split=args.val_split,
            img_size=args.img_size,
        )
        class_names = list(getattr(train_loader.dataset, "classes", [])) or None

    orchestrator = ViTOrchestrator(
        args.model,
        num_classes=num_classes,
        in_chans=args.in_chans,
        img_size=args.img_size,
        optimizer=args.optimizer,
        lr=args.lr,
        optimizer_kwargs=args.optimizer_kwargs,
        device=args.device,
        **args.model_kwargs,
    )
    orchestrator.fit(train_loader, epochs=args.epochs, val_loader=val_loader)
    orchestrator.configuration.update(
        dataset="synthetic" if args.smoke_test else args.dataset,
        class_names=class_names,
    )
    output = args.output or Path("outputs") / f"{args.model}.pt"
    orchestrator.save(output)
    print(json.dumps({"checkpoint": str(output), **orchestrator.summary()}, indent=2))
    return 0


def _input_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(path)


def _image_transform(img_size: int, in_chans: int):
    from torchvision import transforms

    operations: list[Any] = [transforms.Resize((img_size, img_size))]
    if in_chans == 1:
        operations.append(transforms.Grayscale(num_output_channels=1))
    operations.extend(
        [transforms.ToTensor(), transforms.Normalize([0.5] * in_chans, [0.5] * in_chans)]
    )
    return transforms.Compose(operations)


def infer_command(args: argparse.Namespace) -> int:
    orchestrator = ViTOrchestrator.from_checkpoint(args.checkpoint, device=args.device)
    summary = orchestrator.summary()
    img_size_value = summary["input_size"]
    img_size = int(img_size_value if isinstance(img_size_value, int) else img_size_value[0])
    in_chans = int(summary["input_channels"])
    if args.smoke_test:
        images = torch.randn(1, in_chans, img_size, img_size)
        paths = [Path("synthetic")]
    else:
        if args.input is None:
            raise ValueError("--input is required unless --smoke-test is used.")
        if in_chans not in {1, 3}:
            raise ValueError(
                "Image-file inference supports 1 or 3 channels; use the Python API for multispectral tensors."
            )
        paths = _input_paths(args.input)
        transform = _image_transform(img_size, in_chans)
        images = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths])

    values, indices = orchestrator.predict_topk(images, k=args.top_k)
    class_names = orchestrator.configuration.get("class_names")
    result = []
    for path, sample_values, sample_indices in zip(
        paths, values.cpu(), indices.cpu(), strict=False
    ):
        predictions = [
            {
                "class_id": int(index),
                "class_name": class_names[int(index)] if class_names else None,
                "probability": float(value),
            }
            for value, index in zip(sample_values, sample_indices, strict=False)
        ]
        result.append({"input": str(path), "predictions": predictions})
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


def analyze_command(args: argparse.Namespace) -> int:
    orchestrator = ViTOrchestrator.from_checkpoint(args.checkpoint, device=args.device)
    config = orchestrator.configuration
    _, _, test_loader = get_classification_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        val_split=0.0,
        img_size=int(config["img_size"]),
    )
    class_names = list(getattr(test_loader.dataset, "classes", [])) or None
    analysis = orchestrator.analyze(test_loader, class_names=class_names, num_bins=args.num_bins)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(analysis["confusion_matrix"], args.output_dir / "confusion_matrix.pt")
    payload = {"report": analysis["report"], "calibration": analysis["calibration"]}
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    orchestrator.plot_confusion_matrix(
        test_loader,
        class_names=class_names,
        save_path=args.output_dir / "confusion_matrix.png",
    )
    orchestrator.plot_calibration(
        test_loader,
        num_bins=args.num_bins,
        save_path=args.output_dir / "calibration.png",
    )
    print(json.dumps({"output_dir": str(args.output_dir), **analysis["report"]}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        for name in list_models():
            info = model_info(name)
            print(f"{name:22} {info.family:26} {info.description}")
        return 0
    if args.command == "info":
        print(json.dumps(model_info(args.model).__dict__, indent=2))
        return 0
    if args.command == "train":
        return train_command(args)
    if args.command == "infer":
        return infer_command(args)
    if args.command == "run":
        print(json.dumps(run_pipeline(args.config), indent=2))
        return 0
    if args.command == "validate-config":
        print(json.dumps(load_pipeline_config(args.config), indent=2))
        return 0
    return analyze_command(args)

#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from data.load_cifrar100 import get_cifar100_dataloaders
from model.MaxViT import MaxViT
from model_configurations import (
    maxvit_cifar100_tiny,
    maxvit_cifar100_small,
    maxvit_cifar100_base,
)
from inference.evaluate_loader import evaluate_classifier
from inference.confussion_matrix import confusion_matrix, plot_confusion_matrix, top_confusions
from inference.calibration_stats import calibration_stats, plot_reliability
from inference.grad_cam import GradCAM, find_last_conv2d, overlay_cam
from inference.oclusion import occlusion_sensitivity, plot_occlusion_heatmap
from inference.history_visual import show_predictions_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Run MaxViT inference and analysis on CIFAR-100")

    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file to load.")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights if present in checkpoint.")

    parser.add_argument("--variant", choices=["tiny", "small", "base"], default="tiny")
    parser.add_argument("--stem-type", choices=["A", "B"], default="A")
    parser.add_argument("--drop-path-rate", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--amp", dest="use_amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", default=None)
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    parser.add_argument(
        "--analysis",
        nargs="+",
        default=["eval"],
        choices=["eval", "confusion", "calibration", "gradcam", "occlusion", "predictions"],
        help="Which analyses to run.",
    )

    parser.add_argument("--topk", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index for Grad-CAM/occlusion.")
    parser.add_argument("--class-idx", type=int, default=None, help="Target class for Grad-CAM/occlusion.")
    parser.add_argument("--max-classes", type=int, default=30, help="Classes shown in confusion matrix plot.")
    parser.add_argument("--no-show", action="store_true", help="Disable matplotlib visualizations.")

    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory to save outputs.")

    return parser.parse_args()


def build_config(variant: str, stem_type: str, drop_path_rate: float):
    variants = {
        "tiny": maxvit_cifar100_tiny,
        "small": maxvit_cifar100_small,
        "base": maxvit_cifar100_base,
    }
    return variants[variant](stem_type=stem_type, drop_path_rate=drop_path_rate)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str, use_ema: bool = False):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        if use_ema and state.get("model_ema") is not None:
            ema_state = state["model_ema"]
            if isinstance(ema_state, dict) and "model" in ema_state:
                state = ema_state["model"]
            else:
                state = ema_state
        elif "model" in state:
            state = state["model"]
    model.load_state_dict(state, strict=True)


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _get_sample(dataset, index: int):
    if index < 0 or index >= len(dataset):
        raise IndexError(f"sample-index {index} is out of range (0..{len(dataset)-1}).")
    x, y = dataset[index]
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x, y


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_amp is None:
        args.use_amp = device.startswith("cuda")

    cfg = build_config(args.variant, args.stem_type, args.drop_path_rate)
    model = MaxViT(cfg)

    load_checkpoint(model, args.checkpoint, device=device, use_ema=args.use_ema)
    model.to(device)

    _, _, test_loader = get_cifar100_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        val_split=0.0,
        pin_memory=device.startswith("cuda"),
    )

    output_dir = Path(args.output_dir) if args.output_dir else None

    if "eval" in args.analysis:
        metrics = evaluate_classifier(
            model=model,
            loader=test_loader,
            device=device,
            amp=args.use_amp,
            amp_dtype=args.amp_dtype,
            num_classes=cfg.num_classes,
            topk=tuple(args.topk),
        )
        metrics_fmt = " | ".join(
            [f"top{k}: {metrics[f'top{k}']:.2f}%" for k in sorted(args.topk)]
        )
        print(f"[Eval] loss {metrics['loss']:.4f} | {metrics_fmt}")
        if output_dir is not None:
            _save_json(output_dir / "eval_metrics.json", metrics)

    if "confusion" in args.analysis:
        cm = confusion_matrix(model, test_loader, num_classes=cfg.num_classes, device=device)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / "confusion_matrix.npy", cm)
        top = top_confusions(cm, k=20)
        print("Top confusions (count, true, pred):")
        for count, true_idx, pred_idx in top:
            print(f"  {count:4d} | {true_idx:3d} -> {pred_idx:3d}")
        if not args.no_show:
            plot_confusion_matrix(cm, class_names=None, normalize=True, max_classes=args.max_classes)

    if "calibration" in args.analysis:
        calib = calibration_stats(model, test_loader, device=device)
        print(f"Calibration ECE: {calib['ece']:.4f}")
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            npz_path = output_dir / "calibration_stats.npz"
            np.savez(
                npz_path,
                bins=calib["bins"],
                acc=calib["acc"],
                avg_conf=calib["avg_conf"],
                counts=calib["counts"],
                ece=calib["ece"],
            )
        if not args.no_show:
            plot_reliability(calib)

    if "predictions" in args.analysis:
        if not args.no_show:
            show_predictions_grid(model, test_loader, class_names=None, n=16, device=device)

    if "gradcam" in args.analysis:
        x, y = _get_sample(test_loader.dataset, args.sample_index)
        x = x.to(device)
        target_layer = find_last_conv2d(model)
        cam = GradCAM(model, target_layer)
        try:
            cam_map, class_idx = cam(x, class_idx=args.class_idx)
        finally:
            cam.close()
        print(f"Grad-CAM target class: {class_idx} (gt: {int(y)})")
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(cam_map.cpu(), output_dir / "gradcam.pt")
        if not args.no_show:
            overlay_cam(x[0].detach().cpu(), cam_map[0])

    if "occlusion" in args.analysis:
        x, y = _get_sample(test_loader.dataset, args.sample_index)
        base_score, class_idx, heat = occlusion_sensitivity(
            model,
            x,
            target_class=args.class_idx,
            device=device,
        )
        print(f"Occlusion target class: {class_idx} (gt: {int(y)}) | base prob {base_score:.4f}")
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(torch.tensor(heat), output_dir / "occlusion_heatmap.pt")
        if not args.no_show:
            plot_occlusion_heatmap(x[0].detach().cpu(), heat)


if __name__ == "__main__":
    main()

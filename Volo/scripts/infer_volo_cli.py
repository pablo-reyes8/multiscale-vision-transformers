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

from data.load_data_ddp import get_cifar100_datasets
from model.VOLO import VOLOClassifier
from inference.evaluate_loader import evaluate_classifier
from inference.confussion_matrix import confusion_matrix, plot_confusion_matrix, top_confusions
from inference.calibration_stats import calibration_stats, plot_reliability
from inference.grad_cam import GradCAM, find_last_conv2d, overlay_cam
from inference.oclusion import occlusion_sensitivity, plot_occlusion_heatmap
from inference.history_visual import show_predictions_grid


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    return tuple(int(v) for v in items)


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _get_sample(dataset, index: int):
    if index < 0 or index >= len(dataset):
        raise IndexError(f"sample-index {index} is out of range (0..{len(dataset) - 1}).")
    x, y = dataset[index]
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x, y


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run VOLO inference and analysis on CIFAR-100")

    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file to load.")
    parser.add_argument("--device", type=str, default=None)

    # data
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="./data/cifar100")
    parser.add_argument("--img-size", type=int, default=32)

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

    parser.add_argument("--dims", default="192,256,384")
    parser.add_argument("--outlooker-depths", default="2,2,0")
    parser.add_argument("--outlooker-heads-list", default="6,8,12")
    parser.add_argument("--transformer-depths", default="0,2,2")
    parser.add_argument("--transformer-heads-list", default="6,8,12")

    # inference settings
    parser.add_argument("--amp", dest="use_amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", default=None)
    parser.add_argument("--amp-dtype", type=str, default="fp16", choices=["bf16", "fp16"])
    parser.add_argument(
        "--analysis",
        nargs="+",
        default=["eval", "confusion", "calibration", "gradcam", "occlusion", "predictions"],
        choices=["eval", "confusion", "calibration", "gradcam", "occlusion", "predictions"],
        help="Which analyses to run.",
    )
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--class-idx", type=int, default=None)
    parser.add_argument("--max-classes", type=int, default=30)
    parser.add_argument("--no-show", action="store_true")

    parser.add_argument("--output-dir", type=str, default="./outputs/volo_inference")

    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_amp is None:
        args.use_amp = device.startswith("cuda")

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
        use_cls_pos=args.use_cls_pos,
    )

    _load_checkpoint(model, args.checkpoint, device=device)
    model.to(device)

    _, _, test_ds = get_cifar100_datasets(
        data_dir=args.data_dir,
        val_split=0.0,
        img_size=args.img_size,
        ddp_safe_download=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "eval" in args.analysis:
        metrics = evaluate_classifier(
            model=model,
            loader=test_loader,
            device=device,
            amp=args.use_amp,
            amp_dtype=args.amp_dtype,
            num_classes=args.num_classes,
            topk=tuple(args.topk),
        )
        metrics_fmt = " | ".join([f"top{k}: {metrics[f'top{k}']:.2f}%" for k in sorted(args.topk)])
        print(f"[Eval] loss {metrics['loss']:.4f} | {metrics_fmt}")
        _save_json(output_dir / "eval_metrics.json", metrics)

    if "confusion" in args.analysis:
        cm = confusion_matrix(model, test_loader, num_classes=args.num_classes, device=device)
        np.save(output_dir / "confusion_matrix.npy", cm)
        top = top_confusions(cm, k=20)
        print("Top confusions (count, true, pred):")
        for count, true_idx, pred_idx in top:
            print(f"  {count:4d} | {true_idx:3d} -> {pred_idx:3d}")
        plot_confusion_matrix(
            cm,
            class_names=None,
            normalize=True,
            max_classes=args.max_classes,
            save_path=output_dir / "confusion_matrix.png",
            show=not args.no_show,
        )

    if "calibration" in args.analysis:
        calib = calibration_stats(model, test_loader, device=device)
        print(f"Calibration ECE: {calib['ece']:.4f}")
        np.savez(
            output_dir / "calibration_stats.npz",
            bins=calib["bins"],
            acc=calib["acc"],
            avg_conf=calib["avg_conf"],
            counts=calib["counts"],
            ece=calib["ece"],
        )
        plot_reliability(calib, save_prefix=str(output_dir / "calibration"), show=not args.no_show)

    if "predictions" in args.analysis:
        show_predictions_grid(
            model,
            test_loader,
            class_names=None,
            n=16,
            device=device,
            save_path=output_dir / "predictions_grid.png",
            show=not args.no_show,
        )

    if "gradcam" in args.analysis:
        x, y = _get_sample(test_ds, args.sample_index)
        x = x.to(device)
        target_layer = find_last_conv2d(model)
        cam = GradCAM(model, target_layer)
        try:
            cam_map, class_idx = cam(x, class_idx=args.class_idx)
        finally:
            cam.close()
        print(f"Grad-CAM target class: {class_idx} (gt: {int(y)})")
        torch.save(cam_map.cpu(), output_dir / "gradcam.pt")
        overlay_cam(
            x[0].detach().cpu(),
            cam_map[0],
            save_path=output_dir / "gradcam_overlay.png",
            show=not args.no_show,
        )

    if "occlusion" in args.analysis:
        x, y = _get_sample(test_ds, args.sample_index)
        base_score, class_idx, heat = occlusion_sensitivity(
            model,
            x,
            target_class=args.class_idx,
            device=device,
        )
        print(f"Occlusion target class: {class_idx} (gt: {int(y)}) | base prob {base_score:.4f}")
        torch.save(torch.tensor(heat), output_dir / "occlusion_heatmap.pt")
        plot_occlusion_heatmap(
            x[0].detach().cpu(),
            heat,
            save_path=output_dir / "occlusion_heatmap.png",
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()

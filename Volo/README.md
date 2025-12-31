# VOLO (Vision Outlooker)

This subproject implements VOLO for CIFAR-100, focused on the Outlook Attention family (local dynamic aggregation) and a light global transformer stack. It provides both a flat VOLO variant (Outlooker + Transformer blocks) and a hierarchical variant with pyramid downsampling, plus training and inference CLIs and analysis utilities.

## Architecture
- **Patch embedding:** `PatchEmbeddingConv` converts images into a grid of patches via strided Conv2d, with optional padding for non-divisible sizes.
- **Outlook attention:** `OutlookAttention` performs dynamic local aggregation over kxk neighborhoods. It operates on channel-last maps `[B, H, W, C]` for convenience with spatial windows.
- **Outlooker blocks:** `OutlookerBlock` stacks layer norm, Outlook attention, and MLP with residual connections (pre-norm).
- **Flat VOLO:** A single local stage (`VOLOStage`) followed by global `TransformerBlock`s over tokens, then pooling (mean / cls / cli).
- **Hierarchical VOLO:** `VOLOPyramid` stacks local and global stages with downsampling between levels (map or token pooling) and mean pooling at the end.
- **Head:** LayerNorm over the final features and a linear classifier.

## Repository Structure
- `model/`: VOLO building blocks (`VOLO.py`, outlook, attention, embeddings, pyramid).
- `data/`: CIFAR-100 loaders with DDP-safe download and deterministic splits.
- `training/`: Training loop, metrics, AMP helpers, mixup/cutmix, and checkpointing.
- `inference/`: Evaluation and analysis utilities (confusion matrix, calibration, Grad-CAM, occlusion, prediction grids).
- `scripts/`: Training and inference CLIs.
- `test/`: Pytest suite for model components, data, and training utilities.

## Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r Volo/requirements.txt
   ```
   Use PyTorch wheels that match your hardware/driver (see https://pytorch.org).

## Training from the CLI
Single GPU (default config matches the DDP script defaults):
```bash
python Volo/scripts/train_single_gpu.py \
  --data-dir ./data/cifar100 \
  --epochs 130 \
  --batch-size 256
```

DDP (torchrun):
```bash
torchrun --nproc_per_node=2 Volo/main_training_ddp.py \
  --data-dir ./data/cifar100 \
  --epochs 130 \
  --batch-size 256
```

Useful flags:
- `--pooling mean|cls|cli` selects the flat VOLO pooling strategy.
- `--hierarchical` enables the pyramid backbone and `--downsample-kind map|token`.
- `--mixup-alpha`, `--cutmix-alpha`, `--mix-prob` enable mixup/cutmix.
- `--use-amp/--no-amp` and `--autocast-dtype fp16|bf16` control mixed precision.
- `--resume-path` resumes from a checkpoint produced by `training/chekpoints.py`.

## Inference and Analysis CLI
Run evaluation and analysis on a trained checkpoint:
```bash
python Volo/scripts/infer_volo_cli.py \
  --checkpoint /path/to/best_model.pt \
  --analysis eval confusion calibration gradcam occlusion predictions
```

Outputs are saved by default to `./outputs/volo_inference`:
- `eval_metrics.json`
- `confusion_matrix.npy` and `confusion_matrix.png`
- `calibration_stats.npz` and reliability plots
- `gradcam.pt` and `gradcam_overlay.png`
- `occlusion_heatmap.pt` and `occlusion_heatmap.png`
- `predictions_grid.png`

Use `--no-show` in headless environments and `--output-dir` to redirect results.

## Docker
Build the VOLO image from the repository root:
```bash
docker build -f Volo/Dockerfile -t volo .
```

Run inference inside the container (mount checkpoints/data as needed):
```bash
docker run --rm -it \
  --gpus all \
  -v $PWD/data:/app/data \
  -v $PWD/outputs:/app/outputs \
  -v $PWD/checkpoints:/app/checkpoints \
  volo \
  python scripts/infer_volo_cli.py --checkpoint /app/checkpoints/best_model.pt
```

## Testing
Run the unit test suite:
```bash
pytest Volo/test
```

## References
- Yuan et al., "VOLO: Vision Outlooker for Visual Recognition," 2021.
- Vaswani et al., "Attention is All You Need," NeurIPS 2017.

## License
Released under the MIT License. See `LICENSE` at the repository root for details.

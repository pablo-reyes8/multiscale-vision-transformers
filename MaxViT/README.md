# MaxViT (Multi-Axis Vision Transformer)

MaxViT combines local windowed attention and global grid attention inside every block, paired with MBConv-style convolutions. This design balances locality and long-range context while keeping compute manageable for small images like CIFAR-100. This subproject provides a clean, modular implementation of MaxViT along with training and inference CLIs and analysis utilities.

## Architecture
- **Stem:** Light convolutional stem (A or B) to map inputs into the initial feature space.
- **MBConv + MaxViT blocks:** Each block uses MBConv-style depthwise/separable convolutions for local mixing, followed by two attention passes: windowed attention (local) and grid attention (global).
- **Hierarchical stages:** The model downsamples between stages (conv or pooling), expanding channel capacity while shrinking spatial size.
- **Head:** Global average pooling and a linear classifier.

## Repository Structure
- `model/`: MaxViT building blocks (stem, MBConv, attention, blocks, stages).
- `model_configurations.py`: Predefined CIFAR-100 presets (tiny/small/base).
- `data/`: CIFAR-100 dataloaders and dataset utilities.
- `training/`: Training loop, scheduler, EMA, AMP, and mixup/cutmix.
- `inference/`: Evaluation and analysis utilities (confusion matrix, calibration, Grad-CAM, occlusion, etc.).
- `scripts/`: CLI entrypoints for training and inference.
- `tests/`: Pytest suite for model components and training/data utilities.
- `config/`: Reference YAML configs for model presets.

## Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Use PyTorch wheels that match your hardware/driver (see https://pytorch.org).

## Training from the CLI
Run training on CIFAR-100:
```bash
 scripts/train_maxvit_cli.py \
  --variant tiny \
  --data-dir ./data \
  --epochs 50 \
  --val-split 0.1 \
  --batch-size 128
```

Useful flags:
- `--variant tiny|small|base` selects a preset from `model_configurations.py`.
- `--stem-type A|B`, `--drop-path-rate` customize the backbone.
- `--mixup-alpha`, `--cutmix-alpha`, `--mix-prob` enable mixup/cutmix.
- `--use-ema`, `--ema-decay`, `--ema-warmup-epochs` enable EMA weights.
- `--autocast-dtype fp16|bf16` and `--no-amp` control mixed precision.

## Inference and Analysis CLI
Run evaluation on CIFAR-100 test set:
```bash
python scripts/infer_maxvit_cli.py \
  --checkpoint experiments/maxvit_tiny/best_model.pt \
  --analysis eval
```

Run multiple analyses (confusion, calibration, Grad-CAM):
```bash
python scripts/infer_maxvit_cli.py \
  --checkpoint experiments/maxvit_tiny/best_model.pt \
  --analysis eval confusion calibration gradcam
```

Notes:
- Use `--use-ema` to evaluate EMA weights from a training checkpoint.
- Use `--no-show` to disable matplotlib windows in headless environments.
- Use `--output-dir` to save metrics and analysis outputs.

## Docker
Build an image from this directory:
```bash
docker build -t maxvit .
```

Train inside the container (mount data/checkpoints as needed):
```bash
docker run --rm -it \
  --gpus all \
  -v $PWD/data:/app/data \
  -v $PWD/experiments:/app/experiments \
  maxvit \
  python scripts/train_maxvit_cli.py --data-dir /app/data --output-dir /app/experiments/maxvit_tiny
```


## References
- Tu et al., "MaxViT: Multi-Axis Vision Transformer," ECCV 2022.
- Tan and Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML 2019 (MBConv).
- Vaswani et al., "Attention is All You Need," NeurIPS 2017.

## License
Released under the MIT License. See `LICENSE` at the repository root for details.

# Hierarchical Vision Transformer (HierarchicalViT)

A hierarchical Vision Transformer tailored for CIFAR-100 that blends convolutional patch embedding with progressive pooling (PiT-lite style) to reduce token count while deepening representations. The model uses lightweight transformer blocks per stage, depthwise + pointwise pooling between stages, and a simple classifier head on globally averaged tokens.

## Architecture
- **Conv patch stem:** `PatchEmbeddingConv` slices the image into non-overlapping patches via a strided convolution and normalizes the resulting tokens.
- **Hierarchical transformer stages:** Each stage applies a stack of pre-norm transformer encoder blocks with multi-head self-attention and GELU MLPs.
- **Pooling between stages:** `PoolingLayer` performs depthwise 3x3 stride-2 pooling followed by pointwise projection to shrink spatial resolution and expand channel capacity.
- **Classifier head:** LayerNorm followed by a linear layer on the mean of the final-stage tokens.

## Repository Structure
- `model/`: Core building blocks (`hierarchical_vit.py`, attention, patch embedding, pooling).
- `data/`: CIFAR-100 dataloaders and visualization helpers.
- `training/`: Training loop, evaluation utilities, and AMP helpers.
- `validation/`: Lightweight evaluation helpers and visualization scripts.
- `scripts/`: CLI entrypoints for training/evaluation (`python -m scripts.main ...`).
- `test/`: Pytest-based unit tests for model and attention components.
- `experiments/`: Example outputs and visualizations.
- `notebooks/`: Exploration and experimentation notebooks.

## Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Use the PyTorch wheels that match your hardware/driver (see https://pytorch.org for alternatives).

## Training from the CLI
Run a full train/val loop (defaults target CIFAR-100):
```bash
python -m scripts.main train \
  --data-dir ./data \
  --epochs 20 \
  --val-split 0.1 \
  --checkpoint-path checkpoints/best_hvit.pth \
  --evaluate-test
```

Evaluate an existing checkpoint:
```bash
python -m scripts.main eval \
  --checkpoint checkpoints/best_hvit.pth \
  --data-dir ./data
```

Useful flags:
- `--embed-dims`, `--depths`, `--num-heads` let you override the default 3-stage configuration.
- `--autocast-dtype fp16|bf16` and `--no-amp` control mixed precision behavior.
- `--save-final` stores the last-epoch weights in addition to the best validation checkpoint.

## Docker
Build an image:
```bash
docker build -t hierarchical-vit .
```

Train inside the container (mount data/checkpoints as needed):
```bash
docker run --rm -it \
  --gpus all \
  -v $PWD/data:/app/data \
  -v $PWD/checkpoints:/app/checkpoints \
  hierarchical-vit \
  python -m scripts.main train --data-dir /app/data --checkpoint-path /app/checkpoints/best_hvit.pth
```

## Testing
Run the unit test suite:
```bash
pytest
```

## References
- Alexey Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” ICLR 2021 (Vision Transformer).
- Byeongho Heo et al., “Rethinking Spatial Dimensions of Vision Transformers,” ICCV 2021 (PiT).
- Ashish Vaswani et al., “Attention is All You Need,” NeurIPS 2017 (Transformer).

## License
Released under the MIT License. See `LICENSE` for details.

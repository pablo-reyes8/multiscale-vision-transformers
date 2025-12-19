# Swin Vision Transformer (SwinViT)

This subproject implements a Swin Transformer backbone tailored for CIFAR-100, including window-based self-attention, shifted windows, and hierarchical patch merging. The codebase includes training utilities, scripts for command-line execution, and pytest coverage for key model components.

## Architecture
- **Patch embedding:** `PatchEmbeddingConv` converts images into a patch grid using a strided convolution.
- **Swin stages:** Each stage stacks Swin Transformer blocks with W-MSA and SW-MSA, alternating shifts.
- **Patch merging:** `PatchMerging` reduces spatial resolution while increasing channel capacity between stages.
- **Classifier head:** LayerNorm followed by a linear layer over the masked global average of the final feature map.

## Repository Structure
- `model/`: Swin modules (attention, blocks, window utilities, patch merging).
- `data/`: CIFAR-100 dataloaders and visualization helpers.
- `training/`: Training loop, evaluation utilities, AMP helpers, schedulers.
- `scripts/`: CLI entrypoints (`python -m scripts.main ...`).
- `test/`: Pytest-based unit tests for the Swin model and components.
- `notebooks/`: Exploratory notebooks and experiments.

## Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Use PyTorch wheels that match your hardware/driver (see https://pytorch.org).

## Training from the CLI
Train with CIFAR-100 defaults:
```bash
python -m scripts.main train \
  --data-dir ./data \
  --epochs 20 \
  --val-split 0.1 \
  --checkpoint-path checkpoints/best_swinvit.pth \
  --evaluate-test
```

Evaluate an existing checkpoint:
```bash
python -m scripts.main eval \
  --checkpoint checkpoints/best_swinvit.pth \
  --data-dir ./data
```

Useful flags:
- `--depths` and `--num-heads` override the 4-stage Swin layout.
- `--window-size` sets the attention window size.
- `--autocast-dtype fp16|bf16` and `--no-amp` control mixed precision behavior.

## Docker
Build an image:
```bash
docker build -t swinvit .
```

Train inside the container (mount data/checkpoints as needed):
```bash
docker run --rm -it \
  --gpus all \
  -v $PWD/data:/app/data \
  -v $PWD/checkpoints:/app/checkpoints \
  swinvit \
  python -m scripts.main train --data-dir /app/data --checkpoint-path /app/checkpoints/best_swinvit.pth
```

## Testing
Run the unit test suite:
```bash
pytest
```

## Validation Utilities
Run evaluation and analysis helpers from the CLI:
```bash
python -m scripts.main validate --task test --checkpoint checkpoints/best_swinvit.pth
python -m scripts.main validate --task grid --checkpoint checkpoints/best_swinvit.pth --num-images 8
python -m scripts.main validate --task misclassified --checkpoint checkpoints/best_swinvit.pth --num-images 8
python -m scripts.main validate --task per-class --checkpoint checkpoints/best_swinvit.pth --topk-classes 10
python -m scripts.main validate --task tsne --checkpoint checkpoints/best_swinvit.pth --tsne-max-samples 1000
python -m scripts.main validate --task url --checkpoint checkpoints/best_swinvit.pth --url https://example.com/image.jpg --show
```

## References
- Ze Liu et al., “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows,” ICCV 2021.
- Ashish Vaswani et al., “Attention is All You Need,” NeurIPS 2017.

## License
Released under the MIT License. See `LICENSE` at the repository root for details.

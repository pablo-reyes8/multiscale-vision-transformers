# Vision Transformer (ViT)

This subproject implements the original Vision Transformer baseline for compact image classification benchmarks: non-overlapping patch embedding, a prepended class token, learnable absolute positional embeddings, and full global self-attention in every encoder block. The goal is to provide a clean baseline that matches the overall training setup used by the other transformer families in this repository.

## Architecture
- **Patch embedding:** `PatchEmbedding` slices the image into non-overlapping patches and projects them into the token dimension.
- **Token sequence:** a learned `class token` is prepended and combined with learnable absolute positional embeddings.
- **Transformer encoder:** stacked pre-norm encoder blocks with global multi-head self-attention and GELU MLPs.
- **Classifier head:** LayerNorm followed by a linear classifier on the final CLS token.

## Repository Structure
- `model/`: ViT modules (`patch_embedding.py`, attention blocks, backbone).
- `data/`: dataset loader helpers plus `dataset_zoo.py` for shared multi-dataset support.
- `training/`: Training loop, warmup+cosine LR scheduler, AMP helpers, checkpoints.
- `validation/`: Lightweight checkpoint-loading evaluation helper.
- `scripts/`: CLI entrypoints (`python -m scripts.main ...`).
- `test/`: Pytest-based tests for patching, blocks, model forward, and one training epoch.
- `config/`: Reference YAML config for the CIFAR-100 setup.

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Use PyTorch wheels that match your hardware/driver if you are not using the default CPU install.

## Training from the CLI
Train the baseline on CIFAR-100:
```bash
python -m scripts.main train \
  --dataset cifar100 \
  --data-dir ./data \
  --epochs 20 \
  --val-split 0.1 \
  --checkpoint-path checkpoints/best_vit_cifar100.pt \
  --evaluate-test
```

Evaluate an existing checkpoint:
```bash
python -m scripts.main eval \
  --checkpoint checkpoints/best_vit_cifar100.pt \
  --dataset cifar100 \
  --data-dir ./data
```

Supported datasets:
- `cifar100`
- `svhn`
- `oxford_pets`
- `food101`
- `tiny_imagenet`

Example with another dataset at 32x32:
```bash
python -m scripts.main train \
  --dataset oxford_pets \
  --img-size 32 \
  --data-dir ./data \
  --epochs 20 \
  --val-split 0.1 \
  --checkpoint-path checkpoints/best_vit_oxford_pets.pt
```

Useful flags:
- `--embed-dim`, `--depth`, `--num-heads` change the ViT width/depth.
- `--drop-path-rate`, `--drop-rate`, `--attn-dropout`, `--mlp-dropout` control regularization.
- `--autocast-dtype fp16|bf16` and `--no-amp` control mixed precision.
- `--resume` restores optimizer/scheduler/scaler state from a training checkpoint.

## Docker
Build an image:
```bash
docker build -t vit-baseline .
```

Train inside the container:
```bash
docker run --rm -it \
  --gpus all \
  -v $PWD/data:/app/data \
  -v $PWD/checkpoints:/app/checkpoints \
  vit-baseline \
  python -m scripts.main train --data-dir /app/data --checkpoint-path /app/checkpoints/best_vit_cifar100.pt
```

## Testing
Run the unit tests:
```bash
pytest
```

## References
- Alexey Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR 2021.
- Ashish Vaswani et al., "Attention is All You Need," NeurIPS 2017.

## License
Released under the MIT License. See `LICENSE` at the repository root for details.

# Vision Transformers Lab

A focused research sandbox for comparing modern Vision Transformer families under a shared training and evaluation setup. The project emphasizes clarity, reproducibility, and side-by-side analysis on CIFAR-100 and related image classification tasks.

## Research Focus
- Compare hierarchical inductive biases (pooling vs. windowed attention) under matched training pipelines.
- Provide a consistent CLI, evaluation suite, and validation utilities across model families.
- Track design tradeoffs (accuracy, stability, compute profile) with minimal implementation ambiguity.

## What’s Here Today
- **HierarchicalViT/**: PiT-style pooling hierarchy, full training CLI, Dockerfile, tests, and validation utilities.
- **SwinViT/**: Shifted window attention with patch merging, training CLI, validation tools, tests, and Dockerfile.
- **Model families (roadmap)**: MaxViT with unified benchmarking utilities.

## Getting Started
1) Create an environment (Python ≥3.10 recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

2) Run a model pipeline (example: HierarchicalViT):

```bash
cd HierarchicalViT
python -m scripts.main train --data-dir ./data --epochs 20 --checkpoint-path ./checkpoints/best_hvit.pth
```
Add `--evaluate-test` to benchmark on the test split after training. Use `python -m scripts.main eval --checkpoint ...` to evaluate saved weights.

Example for SwinViT:
```bash
cd SwinViT
python -m scripts.main train --data-dir ./data --epochs 20 --checkpoint-path ./checkpoints/best_swinvit.pth
```

## Repository Structure
- `HierarchicalViT/` – Complete hierarchical ViT implementation, tests, and Dockerfile.
- `SwinViT/` – Shifted window transformer implementation, validation utilities, tests, and Dockerfile.
- `MaxViT/` (planned) – Blocked local-global attention model with grid/windowed attention.
- `notebooks/` (planned) – Shared exploratory notebooks and visualizations.
- `requirements.txt` – Shared dependencies for all model families.

## Model Families: Key Differences
- **HierarchicalViT**: Uses global attention within each stage and reduces token count via pooling between stages (PiT-style). This emphasizes global context early with explicit spatial downsampling.
- **SwinViT**: Uses local window attention with shifted windows to enable cross-window interaction, and patch merging to downsample. This favors locality and computational efficiency at higher resolutions.
- **MaxViT** (planned): Combines local windowed attention and global grid attention within each block to balance locality and global context.

## Testing
From the repo root, run the available suite:
```bash
pytest HierarchicalViT/test
```
Additional suites:
```bash
pytest SwinViT/test
```

## Roadmap
- Implement MaxViT with unified benchmarking harness.
- Add experiment tracking hooks (e.g., optional WandB/MLflow) and logging utilities.
- Expand evaluation scripts to include FLOPs/params reporting and latency profiling.

## References
- Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” ICLR 2021 (ViT).
- Liu et al., “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows,” ICCV 2021.
- Tu et al., “MaxViT: Multi-Axis Vision Transformer,” ECCV 2022.
- Vaswani et al., “Attention is All You Need,” NeurIPS 2017.

## License
Unless stated otherwise, the project is distributed under the MIT License (see `LICENSE`). Individual submodules may ship with their own notices if required.

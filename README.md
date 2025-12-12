# Vision Transformers Lab

A curated collection of vision transformer experiments focused on clarity, reproducibility, and extensibility. The roadmap includes a custom Hierarchical ViT (already implemented), plus upcoming Swin Transformer and MaxViT variants for side-by-side benchmarking on CIFAR-100 and related image classification tasks.

## What’s Here Today
- **HierarchicalViT/**: Hierarchical Vision Transformer with PiT-style pooling, full training CLI, Dockerfile, tests, and documentation.
- **Notebooks (planned)**: Prototyping and ablations shared across models.
- **Model families (roadmap)**: Swin Transformer and MaxViT implemented with common data and training utilities.

## Getting Started
1) Create an environment (Python ≥3.10 recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

2) Run the existing Hierarchical ViT pipeline:

```bash
cd HierarchicalViT
python -m scripts.main train --data-dir ./data --epochs 20 --checkpoint-path ./checkpoints/best_hvit.pth
```
Add `--evaluate-test` to benchmark on the test split after training. Use `python -m scripts.main eval --checkpoint ...` to evaluate saved weights.

## Repository Structure
- `HierarchicalViT/` – Complete hierarchical ViT implementation, tests, and Dockerfile.
- `SwinTransformer/` (planned) – Shifted window transformer implementation and benchmarks.
- `MaxViT/` (planned) – Blocked local-global attention model with grid/windowed attention.
- `notebooks/` (planned) – Shared exploratory notebooks and visualizations.
- `requirements.txt` – Shared dependencies for all model families.

## Testing
From the repo root, run the available suite:
```bash
pytest HierarchicalViT/test
```
Additional suites will be added alongside new model families.

## Roadmap
- Implement Swin Transformer training stack (data pipeline parity, CLI, tests).
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

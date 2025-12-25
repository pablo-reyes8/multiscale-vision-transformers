# Vision Transformers Lab

A focused research sandbox for comparing modern Vision Transformer families under a shared training, evaluation, and analysis setup. The repo emphasizes clarity, reproducibility, and side-by-side inspection of architectural tradeoffs on CIFAR-100 and related image classification tasks.

## Highlights
- Three complete families: HierarchicalViT, SwinViT, and MaxViT.
- Consistent data pipelines, training loops, and evaluation utilities.
- CLI entrypoints for training, evaluation, and analysis.
- Model-specific Dockerfiles plus a root Dockerfile for the full workspace.
- Pytest coverage for core components and critical training utilities.

## Subprojects
| Folder | Model Family | Key Idea | Status |
| --- | --- | --- | --- |
| `HierarchicalViT/` | PiT-style hierarchical ViT | Token pooling between stages | Complete |
| `SwinViT/` | Swin Transformer | Shifted windows + patch merging | Complete |
| `MaxViT/` | MaxViT | Window + grid attention per block | Complete |

Each subproject has its own `README.md`, `requirements.txt`, scripts, tests, and (where relevant) notebooks and experiments.

## Quickstart
1) Create an environment (Python >=3.10 recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

2) Train a model (examples):
```bash
cd HierarchicalViT
python -m scripts.main train --data-dir ./data --epochs 20 --checkpoint-path ./checkpoints/best_hvit.pth
```

```bash
cd SwinViT
python -m scripts.main train --data-dir ./data --epochs 20 --checkpoint-path ./checkpoints/best_swinvit.pth
```

```bash
cd MaxViT
python scripts/train_maxvit_cli.py --variant tiny --data-dir ./data --epochs 20 --val-split 0.1
```

## Inference and Analysis
MaxViT provides a dedicated inference CLI with analysis utilities (confusion matrix, calibration, Grad-CAM, occlusion, prediction grids):
```bash
cd MaxViT
python scripts/infer_maxvit_cli.py --checkpoint experiments/maxvit_tiny/best_model.pt --analysis eval confusion calibration
```

SwinViT and HierarchicalViT include validation helpers accessible through their CLI modules (see each subproject README for details).

## Repository Structure
- `HierarchicalViT/` – Hierarchical ViT implementation, tests, and Dockerfile.
- `SwinViT/` – Shifted window transformer, validation utilities, tests, and Dockerfile.
- `MaxViT/` – MaxViT model, training + inference CLIs, analysis suite, and tests.
- `requirements.txt` – Shared dependency set for the workspace.
- `Dockerfile` – Root container for the whole repo.

## Model Families: Key Differences
- **HierarchicalViT**: Global attention per stage, explicit pooling between stages (PiT-style). Emphasizes structured downsampling and stable token reduction.
- **SwinViT**: Local attention in windows with shifted windowing for cross-window context; patch merging downsampling. Optimizes for efficiency at higher resolutions.
- **MaxViT**: Combines local window attention and global grid attention within each block, paired with MBConv-style convolutions. Balances locality and global context in every block.

## Results
See `MaxViT/README.md` for the current MaxViT training metrics and visualizations. The MaxViT training notebook reports a best validation Top-1 of 66.68% and Top-5 of 89.92% on CIFAR-100 (tiny variant, 20 epochs).

## Docker
Build the root image:
```bash
docker build -t vit-lab .
```

Or build a subproject directly:
```bash
cd MaxViT
docker build -t maxvit .
```

## Testing
From the repo root:
```bash
pytest HierarchicalViT/test
pytest SwinViT/test
pytest MaxViT/tests
```


## References
- Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” ICLR 2021.
- Liu et al., “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows,” ICCV 2021.
- Tu et al., “MaxViT: Multi-Axis Vision Transformer,” ECCV 2022.
- Vaswani et al., “Attention is All You Need,” NeurIPS 2017.

## License
Unless stated otherwise, the project is distributed under the MIT License (see `LICENSE`). Individual submodules may ship with their own notices if required.

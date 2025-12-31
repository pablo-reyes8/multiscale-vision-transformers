# Vision Transformers Lab


[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/multiscale-vision-transformers)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/multiscale-vision-transformers)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/multiscale-vision-transformers)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/multiscale-vision-transformers)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/multiscale-vision-transformers?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/multiscale-vision-transformers?style=social)

A focused research sandbox for comparing modern Vision Transformer families under a shared training, evaluation, and analysis setup. The repo emphasizes clarity, reproducibility, and side-by-side inspection of architectural tradeoffs on CIFAR-100 and related image classification tasks.
> **CIFAR-100 (single-run snapshot)** — Best Val **Top-1**: **VOLO 67.90%** · **MaxViT 66.68%** · **HViT 51.50%** · **Swin 51.04%**  
> Best Val **Top-5**: **MaxViT 89.92%** · **VOLO 88.78%** · **Swin 79.88%** · **HViT 78.40%**  
> *(HViT and Swin use a closely matched recipe; MaxViT and VOLO include additional augmentation/recipe exploration. See “Benchmark Protocol” + the full table below.)*



## Table of Contents
- [Vision Transformers Lab](#vision-transformers-lab)
  - [Highlights](#highlights)
  - [Subprojects](#subprojects)
  - [Model Families: Key Differences](#model-families-key-differences)
  - [Quickstart](#quickstart)
  - [Inference and Analysis](#inference-and-analysis)
  - [Repository Structure](#repository-structure)
  - [Docker](#docker)
  - [Testing](#testing)
  - [CIFAR-100 results](#cifar-100-results)
    - [What these results suggest (research-oriented takeaways)](#what-these-results-suggest-research-oriented-takeaways)
    - [Next steps to make the comparison benchmark-clean](#next-steps-to-make-the-comparison-benchmark-clean)
  - [References](#references)
  - [License](#license)



## Highlights
- **Four complete families**: HierarchicalViT, SwinViT, MaxViT, and **VOLO (Vision Outlooker)**.
- Consistent CIFAR-100 data pipeline shared across subprojects (loaders, transforms, evaluation utilities).
- CLI entrypoints for training, evaluation, and (where implemented) analysis.
- Model-specific Dockerfiles plus a root Dockerfile for the full workspace.
- Pytest coverage for core components and critical training utilities.
- A repo structure designed for **side-by-side architectural comparison** (shared conventions, logs, reproducible scripts).


## Subprojects

<div align="center">
  
| Folder | Model Family | Key Idea | Status |
| --- | --- | --- | --- |
| [`HierarchicalViT/`](./HierarchicalViT) | PiT-style hierarchical ViT | Token pooling between stages | Complete |
| [`SwinViT/`](./SwinViT) | Swin Transformer | Shifted windows + patch merging | Complete |
| [`MaxViT/`](./MaxViT) | MaxViT | Window + grid attention per block | Complete |
| [`Volo/`](./Volo) | VOLO (Vision Outlooker) | Outlook attention + token refinement | Complete |
</div>

Each subproject has its own `README.md`, `requirements.txt`, scripts, tests, and (where relevant) notebooks and experiments.


## Model Families: Key Differences
- **HierarchicalViT**: Global attention per stage, explicit pooling between stages (PiT-style). Emphasizes structured downsampling and stable token reduction.
- **SwinViT**: Local attention in windows with shifted windowing for cross-window context; patch merging downsampling. Optimizes for efficiency at higher resolutions.
- **MaxViT**: Combines local window attention and global grid attention within each block, paired with MBConv-style convolutions. Balances locality and global context in every block.
- **VOLO (Vision Outlooker)**: Replaces early global self-attention with **outlook attention** to inject fine-grained local context into tokens, then refines with transformer-style processing. Strong inductive bias for recognition via “token enrichment” before global mixing.

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
```
Repository Structure
├── HierarchicalViT/        # Hierarchical ViT implementation + tests + Dockerfile
├── SwinViT/                # Swin Transformer implementation + validation utilities + tests + Dockerfile
├── MaxViT/                 # MaxViT implementation + training/inference CLIs + analysis suite + tests
├── Volo/                   # VOLO implementation + training logs/scripts/tests (see subproject README)
├── training logs/          # Training histories (.txt) for all runs (HViT / Swin / MaxViT / VOLO)
├── requirements.txt        # Shared dependency set for the workspace
└── Dockerfile              # Root container for the whole repository
```


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

## CIFAR-100 results 

The table below reports the **best validation epoch observed** for each model (from my training runs).

### Benchmark Protocol (this snapshot)
These numbers come from **single runs** and are intended as a practical side-by-side snapshot.

- **Dataset**: CIFAR-100 (torchvision), standard train/test.
- **Input resolution**: 32×32.
- **Data pipeline**: kept **highly consistent across families** (same CIFAR-100 loaders and preprocessing; HViT and Swin share the same training setup).
- **Training recipe**:
  - **HViT & Swin**: trained under the **same recipe** (optimizer/schedule/augmentations kept aligned).
  - **MaxViT**: I ran additional experiments with **Mixup/CutMix**, so the training recipe is *close* but not identical.
- **Reproducibility**: raw logs for each run are stored in `training_logs/` (`*.txt`).

> **Interpretation note:** comparisons are strongest between **HViT vs Swin** (matched recipe).  
> MaxViT numbers are still comparable, but reflect some extra augmentation experiments.

<div align="center">

| Model | Best epoch | Val loss | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|---:|---:|
| **Hierarchical ViT (HViT)** | 43/50 | 2.0410 | **51.50%** | 71.20% | 78.40% |
| **Swin Transformer** | 43/50 | 2.4145 | **51.04%** | 71.30% | **79.88%** |
| **MaxViT** | 17/20 | 1.2132 | **66.68%** | **84.58%** | **89.92%** |
| **VOLO (Vision Outlooker)** | 63/130 | 1.3082 | **67.90%** | 83.76% | 88.78% |

</div>

### What these results suggest (research-oriented takeaways)

**1) Top-1 ranking (this snapshot): VOLO ≳ MaxViT ≫ (HViT ≈ Swin).**  
VOLO reaches **67.90% Top-1**, slightly ahead of MaxViT (**66.68%**).  
Both substantially outperform HViT (**51.50%**) and Swin (**51.04%**).  
Interpretation: on **32×32**, strong locality priors (Outlook attention / MBConv + multi-axis attention) appear especially effective.

**2) Top-5 ranking: MaxViT > VOLO > (Swin > HViT).**  
MaxViT leads on Top-5 (**89.92%**) vs VOLO (**88.78%**).  
This can indicate MaxViT spreads probability mass across plausible classes more effectively under this recipe, even when Top-1 is close.

**3) Recipe sensitivity warning (benchmark fairness).**  
HViT vs Swin comparisons are the cleanest (matched recipe).  
VOLO and MaxViT currently reflect more exploration (longer training for VOLO; extra Mixup/CutMix for MaxViT).  
Treat these as **strong hints**, not final conclusions.

**4) Generalization gap & regularization target (HViT).**  
HViT shows a large train–val gap in this setup, suggesting clear headroom via stronger regularization, augmentation, and/or early stopping.


--- 

### Next steps to make the comparison benchmark-clean
- For **HViT**, focus on closing the generalization gap: tune drop-path, strengthen aug, and consider EMA + early stopping.
- For **Swin**, modest recipe tuning often helps (WD/LR schedule, aug strength). It’s already close in Top-1 and strong in Top-5.
- For **MaxViT**, test robustness: try smaller variants / fewer params, or evaluate transfer (CIFAR-10, Tiny-ImageNet) to see whether the advantage persists.



## References
- Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” ICLR 2021.
- Liu et al., “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows,” ICCV 2021.
- Tu et al., “MaxViT: Multi-Axis Vision Transformer,” ECCV 2022.
- Vaswani et al., “Attention is All You Need,” NeurIPS 2017.
- Yuan et al., “VOLO: Vision Outlooker for Visual Recognition,” arXiv:2106.13112, 2021.


## License
Unless stated otherwise, the project is distributed under the MIT License (see `LICENSE`). Individual submodules may ship with their own notices if required.

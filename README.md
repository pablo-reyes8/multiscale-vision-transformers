# Vision Transformers Lab


[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/multiscale-vision-transformers)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/multiscale-vision-transformers)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/multiscale-vision-transformers)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/multiscale-vision-transformers)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/multiscale-vision-transformers?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/multiscale-vision-transformers?style=social)

A focused research sandbox for comparing modern Vision Transformer families under a shared training, evaluation, and analysis setup. Experiments are designed to run on a single NVIDIA T4 (practical, budget-aware ViT variants rather than massive SOTA-scale models), while the codebase remains scalable by design—configs, training loops, and model definitions can be expanded to larger sizes and longer schedules as compute allows.

> **CIFAR-100 (single-run snapshot)** — Best Val **Top-1**: **VOLO 67.90%** · **MaxViT 66.68%** · **HViT 51.50%** · **Swin 51.04%**  
> Best Val **Top-5**: **MaxViT 89.92%** · **VOLO 88.78%** · **Swin 79.88%** · **HViT 78.40%**  
> *(HViT and Swin use a closely matched recipe; MaxViT and VOLO include additional augmentation/recipe exploration. See “Benchmark Protocol” + the full table below.)*



## Table of Contents
- [Vision Transformers Lab](#vision-transformers-lab)
  - [Highlights](#highlights)
  - [Subprojects](#subprojects)
  - [Model Families: Key Differences](#model-families-key-differences)
  - [Quickstart](#quickstart)
  - [Comparison Arena](#comparison-arena)
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
- **Five model families**: original ViT, HierarchicalViT, SwinViT, MaxViT, and **VOLO (Vision Outlooker)**.
- Consistent data pipeline shared across subprojects (same loader API, same transform conventions, same evaluation entrypoints).
- A root-level comparison arena with one shared split, one shared training loop, and selectable augmentation presets for direct cross-architecture experiments.
- Subproject CLIs now support multiple easy-to-fetch academic datasets via `--dataset`, including `cifar100`, `svhn`, `oxford_pets`, `food101`, and `tiny_imagenet`.
- CLI entrypoints for training, evaluation, and (where implemented) analysis.
- Model-specific Dockerfiles plus a root Dockerfile for the full workspace.
- Pytest coverage for core components and critical training utilities.
- A repo structure designed for **side-by-side architectural comparison** (shared conventions, logs, reproducible scripts).


## Subprojects

<div align="center">
  
| Folder | Model Family | Key Idea | Status |
| --- | --- | --- | --- |
| [`ViT/`](./ViT) | Original Vision Transformer | Flat patch sequence + CLS token + global attention | Complete |
| [`HierarchicalViT/`](./HierarchicalViT) | PiT-style hierarchical ViT | Token pooling between stages | Complete |
| [`SwinViT/`](./SwinViT) | Swin Transformer | Shifted windows + patch merging | Complete |
| [`MaxViT/`](./MaxViT) | MaxViT | Window + grid attention per block | Complete |
| [`Volo/`](./Volo) | VOLO (Vision Outlooker) | Outlook attention + token refinement | Complete |
</div>

Each subproject has its own `README.md`, `requirements.txt`, scripts, tests, and (where relevant) notebooks and experiments.


## Model Families: Key Differences
- **ViT**: The original baseline with a flat patch sequence, learnable absolute positional embeddings, a CLS token, and global attention in every block.
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
cd ViT
python -m scripts.main train --dataset cifar100 --data-dir ./data --epochs 20 --checkpoint-path ./checkpoints/best_vit_cifar100.pt
```

```bash
cd HierarchicalViT
python -m scripts.main train --dataset oxford_pets --img-size 32 --data-dir ./data --epochs 20 --checkpoint-path ./checkpoints/best_hvit_oxford_pets.pth
```

```bash
cd SwinViT
python -m scripts.main train --dataset tiny_imagenet --img-size 32 --data-dir ./data --epochs 20 --checkpoint-path ./checkpoints/best_swinvit_tiny_imagenet.pth
```

```bash
cd MaxViT
python scripts/train_maxvit_cli.py --variant tiny --dataset food101 --img-size 32 --data-dir ./data --epochs 20 --val-split 0.1
```

```bash
cd Volo
python scripts/train_single_gpu.py --dataset svhn --img-size 32 --data-dir ./data --epochs 20 --batch-size 256
```

3) Inspect the shared comparison arena:
```bash
python3 vit_arena_cli.py --list-models
python3 vit_arena_cli.py --show-tips
python3 vit_arena_cli.py --dry-run --models vit hierarchical_vit swin maxvit_tiny volo
```

## Comparison Arena
At the repository root there is now a reproducible arena for comparing multiple architectures under the same CIFAR-100 split, the same optimizer/scheduler recipe, and the same preprocessing choice:

- `vit_arena.py`: shared experiment engine (data split, training loop, evaluation, summaries).
- `vit_arena_cli.py`: CLI entrypoint to launch comparisons or dry-run model construction.
- `vit_arena_presets.py`: default model presets plus compatibility tips and recommended constraints.

Available augmentation presets:
- `matched`: `RandomCrop + HorizontalFlip + RandAugment + CIFAR-100 normalization`
- `basic`: `RandomCrop + HorizontalFlip + CIFAR-100 normalization`
- `none`: no augmentation, only CIFAR-100 normalization
- `raw`: no augmentation, only normalization to approximately `[-1, 1]`

Example: compare several models with the same recipe
```bash
python3 vit_arena_cli.py \
  --models vit hierarchical_vit swin maxvit_tiny volo \
  --augment matched \
  --epochs 20 \
  --batch-size 128 \
  --output-dir arena_runs/cifar100_matched
```

Example: compare with a raw pipeline (no augment, only `[-1, 1]` normalization)
```bash
python3 vit_arena_cli.py \
  --models vit hierarchical_vit swin \
  --augment raw \
  --epochs 10 \
  --output-dir arena_runs/cifar100_raw
```

The arena writes a manifest, resolved model configs, per-model histories, checkpoints, and CSV/JSON summaries to the output directory so the comparison remains reproducible.

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
├── ViT/                    # Original Vision Transformer baseline + tests + Dockerfile
├── HierarchicalViT/        # Hierarchical ViT implementation + tests + Dockerfile
├── SwinViT/                # Swin Transformer implementation + validation utilities + tests + Dockerfile
├── MaxViT/                 # MaxViT implementation + training/inference CLIs + analysis suite + tests
├── Volo/                   # VOLO implementation + training logs/scripts/tests (see subproject README)
├── vit_arena.py            # Shared arena to train/evaluate multiple architectures under one recipe
├── vit_arena_cli.py        # CLI for cross-model comparison runs and dry-runs
├── vit_arena_presets.py    # Default presets plus compatibility/config tips
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
pytest ViT/test
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
```bibtex
@inproceedings{dosovitskiy2021vit,
  author    = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
  title     = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}

@inproceedings{liu2021swin,
  author    = {Ze Liu and Yutong Lin and Yue Cao and Han Hu and Yixuan Wei and Zheng Zhang and Stephen Lin and Baining Guo},
  title     = {Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021}
}

@inproceedings{tu2022maxvit,
  author    = {Zhengzhong Tu and Hossein Talebi and Han Zhang and Feng Yang and Peyman Milanfar and Alan Bovik and Yinxiao Li},
  title     = {MaxViT: Multi-Axis Vision Transformer},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}

@inproceedings{vaswani2017attention,
  author    = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and {\L}ukasz Kaiser and Illia Polosukhin},
  title     = {Attention Is All You Need},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2017}
}

@article{yuan2021volo,
  author  = {Li Yuan and Yuchen Chen and Tao Wang and Weihao Yu and Yujun Shi and Zihang Jiang and Francis E. H. Tay and Jiashi Feng and Shuicheng Yan},
  title   = {VOLO: Vision Outlooker for Visual Recognition},
  journal = {arXiv preprint arXiv:2106.13112},
  year    = {2021}
}
```

## License

This project is licensed under the **MIT License**.  Individual submodules may ship with their own notices if required. I recomend to reference the original papers.
See the [LICENSE](LICENSE) file for the full legal text.


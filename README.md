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

## CIFAR-100 results (single-run snapshot)

The table below reports the **best validation epoch observed** for each model (based on my training logs).  
⚠️ These runs are **not perfectly apples-to-apples** (different training budgets/recipes and different compute per epoch), so treat this as a *comparative snapshot* rather than a definitive benchmark.

| Model | Best epoch | Val loss | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|---:|---:|
| **Hierarchical ViT (HViT)** | 43/50 | 2.0410 | **51.50%** | 71.20% | 78.40% |
| **Swin Transformer** | 43/50 | 2.4145 | **51.04%** | 71.30% | **79.88%** |
| **MaxViT** | 17/20 | 1.2132 | **66.68%** | **84.58%** | **89.92%** |

### What these results suggest (research-oriented takeaways)

**1) Accuracy ranking (Top-1): MaxViT ≫ (HViT ≈ Swin).**  
MaxViT reaches **66.68% Top-1**, outperforming HViT (**51.50%**) by **+15.18 pp**, and Swin (**51.04%**) by **+15.64 pp**.  
A reasonable interpretation is inductive bias: **MaxViT’s hybrid design** (strong locality + multi-axis attention) seems to be a better match for **32×32** imagery and CIFAR-100 supervision under this training setup.

**2) HViT shows a large train–val gap (potential overfit / recipe mismatch).**  
At the best HViT epoch, training Top-1 is ~**77%** while validation Top-1 is **51.5%** (≈ **25 pp gap**).  
This points to clear headroom in generalization—likely addressable with a more aggressive regularization/augmentation recipe (e.g., stronger drop-path, RandAugment, Mixup/CutMix, EMA), or by stopping earlier around the best validation window.

**3) Swin vs HViT: similar Top-1, but Swin is better on Top-5.**  
HViT slightly edges Swin on Top-1 (**51.50 vs 51.04**), while Swin improves Top-5 (**79.88 vs 78.40**).  
This may indicate that Swin produces better-ranked alternative hypotheses (more probability mass on plausible classes), even when Top-1 remains comparable.

**4) Efficiency trade-off (as trained here).**  
With the current pipeline, Swin trains at roughly **~530 img/s** and **~1.5 min/epoch**, whereas MaxViT runs at about **~370 img/s** and **~2.2 min/epoch**.  
In practice: **MaxViT is markedly more accurate in this setting**, but it is also **heavier per epoch**.

### Next steps to make the comparison benchmark-clean
- For **HViT**, focus on closing the generalization gap: tune drop-path, strengthen aug, and consider EMA + early stopping.
- For **Swin**, modest recipe tuning often helps (WD/LR schedule, aug strength). It’s already close in Top-1 and strong in Top-5.
- For **MaxViT**, test robustness: try smaller variants / fewer params, or evaluate transfer (CIFAR-10, Tiny-ImageNet) to see whether the advantage persists.


## References
- Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” ICLR 2021.
- Liu et al., “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows,” ICCV 2021.
- Tu et al., “MaxViT: Multi-Axis Vision Transformer,” ECCV 2022.
- Vaswani et al., “Attention is All You Need,” NeurIPS 2017.

## License
Unless stated otherwise, the project is distributed under the MIT License (see `LICENSE`). Individual submodules may ship with their own notices if required.

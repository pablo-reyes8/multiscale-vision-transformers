# Famous Vision Transformers

[![CI](https://github.com/pablo-reyes8/multiscale-vision-transformers/actions/workflows/ci.yml/badge.svg)](https://github.com/pablo-reyes8/multiscale-vision-transformers/actions/workflows/ci.yml)
[![Docker](https://github.com/pablo-reyes8/multiscale-vision-transformers/actions/workflows/docker.yml/badge.svg)](https://github.com/pablo-reyes8/multiscale-vision-transformers/actions/workflows/docker.yml)
[![CodeQL](https://github.com/pablo-reyes8/multiscale-vision-transformers/actions/workflows/codeql.yml/badge.svg)](https://github.com/pablo-reyes8/multiscale-vision-transformers/actions/workflows/codeql.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A from-scratch PyTorch library and research model zoo for compact Vision
Transformers. The project exposes one stable API across ViT, HierarchicalViT,
SwinViT, MaxViT and VOLO while preserving their architecture-specific research
implementations and comparison workflows.

The default presets target practical CIFAR-sized experiments, but every public
constructor exposes `num_classes`, `in_chans` and `img_size` for other
classification datasets and multispectral tensors.

## Highlights

- Eight registered presets across five transformer families.
- `create_model(...)` API inspired by `timm`.
- High-level training, evaluation, checkpointing and explainability facade.
- Versioned YAML pipelines for training, model comparisons and analysis.
- Dataset registry for CIFAR-100, SVHN, Oxford-IIIT Pet, Food-101 and Tiny ImageNet.
- Reproducible arena with shared splits, augmentation and optimization settings.
- Installable CLIs, multi-stage Docker image, CI, CodeQL and Dependabot.

## Installation

```bash
git clone https://github.com/pablo-reyes8/multiscale-vision-transformers.git
cd multiscale-vision-transformers
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows, activate the environment with `.venv\Scripts\activate`.
Use `python -m pip install -e ".[research]"` for the legacy notebooks and experiment tools.

## Python API

```python
import torch
from famous_vits import create_model, list_models

print(list_models())

model = create_model(
    "swin",
    num_classes=37,
    in_chans=1,
    img_size=64,
)

logits = model(torch.randn(2, 1, 64, 64))
print(logits.shape)  # (2, 37)
```

Available presets:

```text
vit                   hierarchical_vit
swin                  maxvit_tiny
maxvit_small          maxvit_base
volo                  volo_hierarchical
```

Local pretrained weights are explicit and never downloaded silently:

```python
model = create_model(
    "vit",
    num_classes=100,
    checkpoint_path="outputs/vit_cifar100.pt",
)
```

## High-level orchestration

```python
from famous_vits import ViTOrchestrator

vit = ViTOrchestrator(
    "vit",
    num_classes=100,
    optimizer="adamw",
    lr=5e-4,
    device="cuda",
)

vit.fit(train_loader, epochs=20, val_loader=val_loader)
metrics = vit.evaluate(test_loader)
probabilities = vit.predict(images)
features = vit.extract_features(images)
analysis = vit.analyze(test_loader, class_names=class_names)
vit.save("outputs/vit.pt")
```

The analysis API includes top-k metrics, classification reports, confusion
matrices, calibration/ECE, prediction plots, Grad-CAM and occlusion sensitivity.

## YAML pipelines

YAML is the recommended interface for reproducible runs:

```bash
famous-vits validate-config --config configs/train_cifar100.yaml
famous-vits run --config configs/train_cifar100.yaml
```

Minimal example:

```yaml
version: 1
task: train

model:
  name: vit
  num_classes: 100
  in_chans: 3
  img_size: 32

data:
  dataset: cifar100
  augment: matched

training:
  epochs: 20
  optimizer: adamw
  lr: 0.0005

output:
  checkpoint: outputs/vit_cifar100.pt
```

Included examples cover training, the comparison arena and a CPU smoke test:
[`configs/`](configs/). Environment variables such as `${DATA_DIR}` are expanded
when configurations are loaded.

## CLI

```bash
famous-vits list
famous-vits info maxvit_tiny
famous-vits train --model vit --dataset cifar100 --epochs 20
famous-vits infer --checkpoint outputs/vit.pt --input image.png
famous-vits analyze --checkpoint outputs/vit.pt --dataset cifar100

vit-arena --models vit hierarchical_vit swin --dataset cifar100 --epochs 20
```

The historical `python vit_arena_cli.py ...` command remains available as a
compatibility shim.

## Docker

The container is built in two stages and runs as a non-root user:

```bash
docker build -t famous-vits:local .
docker run --rm famous-vits:local list
docker run --rm famous-vits:local run --config configs/smoke_test.yaml
```

Mount persistent inputs and outputs when training:

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/app/data" \
  -v "$PWD/outputs:/app/outputs" \
  famous-vits:local run --config configs/train_cifar100.yaml
```

## Repository structure

```text
famous_vits/          stable library, CLI, YAML runner and arena
model_zoo/            architecture-specific source, notebooks and legacy tests
configs/              versioned train/arena/smoke YAML examples
tests/                public API and integration tests
artifacts/            small historical benchmark evidence
docs/                 extended library documentation
.github/              CI, security, dependency and release automation
```

## Development

```bash
make install
make lint
make test
make build
make smoke
```

Pre-commit hooks are available through `pre-commit install`. Pull requests run
Ruff, tests on Python 3.10 and 3.12, YAML smoke tests and distribution builds.
Docker, dependency review and CodeQL have separate workflows.

## Benchmark snapshot

These are historical single-run CIFAR-100 results and should not be interpreted
as a controlled leaderboard. Full logs are kept in
[`artifacts/training_logs/`](artifacts/training_logs/).

| Model | Best validation Top-1 | Best validation Top-5 |
| --- | ---: | ---: |
| VOLO | 67.90% | 88.78% |
| MaxViT | 66.68% | 89.92% |
| HierarchicalViT | 51.50% | 78.40% |
| SwinViT | 51.04% | 79.88% |

## Project policies

- Contributions: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security reports: [SECURITY.md](SECURITY.md)
- Code of Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Support: [SUPPORT.md](SUPPORT.md)
- Releases: [CHANGELOG.md](CHANGELOG.md) and [RELEASING.md](RELEASING.md)
- Academic citation: [CITATION.cff](CITATION.cff)
- Detailed API guide: [docs/library.md](docs/library.md)

## License

Released under the [MIT License](LICENSE).

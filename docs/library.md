# Unified ViT library

The repository now has two distinct roles:

- `famous_vits` is the reusable model library. It owns model discovery, construction,
  training, checkpointing, prediction, evaluation and analysis.
- `famous_vits.arena` is the comparison application. It consumes the public library API
  and applies one recipe to multiple registered presets.

The original `ViT/`, `HierarchicalViT/`, `SwinViT/`, `MaxViT/` and `Volo/` directories
remain the source implementations and keep their historical notebooks and scripts working.
Root modules such as `vit_arena.py` and `shared_dataset_zoo.py` are compatibility shims.

## Python API

Model creation follows the familiar `timm.create_model` style:

```python
from famous_vits import create_model, list_models, model_info

print(list_models())
print(model_info("maxvit_tiny"))

model = create_model(
    "swin",
    num_classes=37,
    in_chans=1,
    img_size=64,
)
```

All presets accept `num_classes`, `in_chans` and `img_size` at the same entrypoint.
Architecture-specific preset values can be overridden with keyword arguments:

```python
small_vit = create_model(
    "vit",
    num_classes=10,
    in_chans=4,
    img_size=32,
    embed_dim=96,
    depth=4,
    num_heads=3,
)
```

Local weights are explicit; the package never downloads a checkpoint silently:

```python
model = create_model(
    "vit",
    num_classes=100,
    checkpoint_path="outputs/vit.pt",
)
```

## High-level orchestrator

```python
from famous_vits import ViTOrchestrator

vit = ViTOrchestrator(
    "volo",
    num_classes=100,
    img_size=32,
    optimizer="adamw",
    lr=5e-4,
)

vit.fit(train_loader, epochs=20, val_loader=val_loader)
metrics = vit.evaluate(test_loader)
probabilities = vit.predict(images)
features = vit.extract_features(images)

analysis = vit.analyze(test_loader, class_names=class_names)
vit.plot_confusion_matrix(test_loader, class_names=class_names, save_path="confusion.png")
vit.plot_calibration(test_loader, save_path="calibration.png")
cam, classes = vit.grad_cam(images)
occlusion, classes = vit.occlusion_sensitivity(images)

vit.save("outputs/volo.pt")
restored = ViTOrchestrator.from_checkpoint("outputs/volo.pt")
```

The analysis facade covers the common utilities previously available only in individual
MaxViT or VOLO inference folders: top-k evaluation, confusion matrices, per-class reports,
calibration/ECE, prediction plots, Grad-CAM and occlusion sensitivity.

## CLI

After `pip install -e .`:

```bash
famous-vits list
famous-vits info swin

famous-vits train \
  --model vit \
  --dataset cifar100 \
  --epochs 20 \
  --output outputs/vit_cifar100.pt

famous-vits infer \
  --checkpoint outputs/vit_cifar100.pt \
  --input examples/image.png

famous-vits analyze \
  --checkpoint outputs/vit_cifar100.pt \
  --dataset cifar100 \
  --output-dir outputs/vit_analysis
```

For a fast installation check without downloading data:

```bash
famous-vits train \
  --model vit \
  --num-classes 2 \
  --batch-size 2 \
  --epochs 1 \
  --smoke-test \
  --model-kwargs '{"embed_dim": 24, "depth": 1, "num_heads": 3}'
```

## Arena

The arena is intentionally separate from model selection:

```bash
vit-arena --list-models
vit-arena --dry-run --models vit swin maxvit_tiny
vit-arena \
  --dataset svhn \
  --models vit hierarchical_vit swin \
  --augment matched \
  --epochs 20 \
  --output-dir arena_runs/svhn
```

The old `python3 vit_arena_cli.py ...` command remains supported.


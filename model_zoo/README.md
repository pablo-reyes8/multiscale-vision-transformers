# Model zoo

This directory contains the original, from-scratch architecture projects used by
`famous_vits`:

| Directory | Family | Main implementation |
| --- | --- | --- |
| `ViT/` | Vision Transformer | `model/vision_transformer.py` |
| `HierarchicalViT/` | PiT-style hierarchical ViT | `model/hierarchical_vit.py` |
| `SwinViT/` | Swin Transformer | `model/swin_vision_transformer.py` |
| `MaxViT/` | MaxViT | `model/MaxViT.py` |
| `Volo/` | VOLO | `model/VOLO.py` |

These folders preserve architecture-specific research code, notebooks and tests.
Application code should normally import models through the stable public API:

```python
from famous_vits import create_model

model = create_model("swin", num_classes=100, in_chans=3, img_size=32)
```

The comparison benchmark lives separately in `famous_vits/arena/`.


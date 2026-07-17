import pytest
import torch

from famous_vits import create_model, list_models, model_info

TINY_CONFIGS = {
    "vit": {"embed_dim": 24, "depth": 1, "num_heads": 3},
    "hierarchical_vit": {
        "embed_dims": (24, 48),
        "depths": (1, 1),
        "num_heads": (3, 6),
    },
    "swin": {
        "embed_dim": 24,
        "depths": (1, 1, 1, 1),
        "num_heads": (3, 3, 6, 12),
        "window_size": 2,
    },
    "maxvit_tiny": {
        "stem_out_ch": 16,
        "dims": (16, 32),
        "depths": (1, 1),
        "heads": (2, 4),
        "window_size": 4,
        "grid_size": 4,
    },
    "volo": {
        "embed_dim": 24,
        "outlooker_depth": 1,
        "outlooker_heads": 3,
        "transformer_depth": 1,
        "transformer_heads": 3,
        "pooling": "mean",
    },
}


def test_registry_exposes_every_preset():
    assert set(list_models()) == {
        "vit",
        "hierarchical_vit",
        "swin",
        "maxvit_tiny",
        "maxvit_small",
        "maxvit_base",
        "volo",
        "volo_hierarchical",
    }
    assert model_info("hvit").name == "hierarchical_vit"
    assert model_info("maxvit").name == "maxvit_tiny"


@pytest.mark.parametrize("name", TINY_CONFIGS)
@pytest.mark.parametrize("in_chans", [1, 4])
def test_every_family_supports_general_input_channels(name, in_chans):
    model = create_model(
        name,
        num_classes=7,
        in_chans=in_chans,
        img_size=32,
        **TINY_CONFIGS[name],
    )
    first_conv = next(module for module in model.modules() if isinstance(module, torch.nn.Conv2d))
    assert first_conv.in_channels == in_chans
    with torch.no_grad():
        assert model(torch.randn(2, in_chans, 32, 32)).shape == (2, 7)


def test_pretrained_requires_explicit_local_checkpoint():
    with pytest.raises(ValueError, match="checkpoint_path"):
        create_model("vit", pretrained=True)

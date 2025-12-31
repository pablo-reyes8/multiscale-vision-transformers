import torch

from model.VOLO import VOLOClassifier


def test_volo_classifier_flat_mean():
    torch.manual_seed(0)
    model = VOLOClassifier(
        num_classes=10,
        img_size=32,
        patch_size=4,
        hierarchical=False,
        embed_dim=64,
        outlooker_depth=1,
        outlooker_heads=4,
        transformer_depth=1,
        transformer_heads=4,
        pooling="mean")

    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)


def test_volo_classifier_flat_cli():
    torch.manual_seed(0)
    model = VOLOClassifier(
        num_classes=10,
        img_size=32,
        patch_size=4,
        hierarchical=False,
        embed_dim=64,
        outlooker_depth=1,
        outlooker_heads=4,
        transformer_depth=1,
        transformer_heads=4,
        pooling="cli",
        cls_attn_depth=1)

    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)


def test_volo_classifier_hierarchical_mean():
    torch.manual_seed(0)
    model = VOLOClassifier(
        num_classes=10,
        img_size=32,
        patch_size=4,
        hierarchical=True,
        downsample_kind="map",
        dims=(64, 96),
        outlooker_depths=(1, 0),
        outlooker_heads_list=(4, 4),
        transformer_depths=(0, 1),
        transformer_heads_list=(4, 4),
        pooling="mean")

    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)

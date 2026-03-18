import torch

from model.vision_transformer import VisionTransformer


def test_vit_forward_produces_logits():
    torch.manual_seed(0)
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=64,
        depth=3,
        num_heads=4,
        mlp_ratio=2.0,
        drop_path_rate=0.0,
    )

    images = torch.randn(2, 3, 32, 32)
    logits = model(images)

    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def test_vit_forward_features_returns_cls_and_patch_tokens():
    torch.manual_seed(0)
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        drop_path_rate=0.0,
    )

    images = torch.randn(2, 3, 16, 16)
    cls, patch_tokens, grid = model.forward_features(images)

    assert cls.shape == (2, 64)
    assert patch_tokens.shape == (2, 16, 64)
    assert grid == (4, 4)
    assert torch.isfinite(cls).all()
    assert torch.isfinite(patch_tokens).all()

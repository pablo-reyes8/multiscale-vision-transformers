import torch

from model.hierarchical_vit import HierarchicalViT


def test_hierarchical_vit_forward_produces_logits():
    torch.manual_seed(0)
    model = HierarchicalViT(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dims=(32, 64),
        depths=(1, 1),
        num_heads=(4, 8),
        mlp_ratio=2.0,
        attn_dropout=0.0,
        dropout=0.0)

    images = torch.randn(2, 3, 32, 32)
    logits = model(images)

    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()

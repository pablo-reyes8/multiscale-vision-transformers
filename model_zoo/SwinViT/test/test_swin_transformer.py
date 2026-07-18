import torch

from model.swin_vision_transformer import SwinTransformer


def test_swin_transformer_forward_handles_small_inputs():
    torch.manual_seed(0)
    model = SwinTransformer(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=32,
        depths=(1, 1, 1, 1),
        num_heads=(2, 4, 8, 8),
        window_size=4,
        drop_path_rate=0.0)

    images = torch.randn(2, 3, 16, 16)
    logits = model(images)

    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()

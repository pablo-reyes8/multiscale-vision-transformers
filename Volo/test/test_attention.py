import torch

from model.attention import TransformerBlock


def test_transformer_block_forward_backward():
    torch.manual_seed(0)
    B, N, C = 2, 64, 192
    x = torch.randn(B, N, C, requires_grad=True)

    blk = TransformerBlock(
        dim=C,
        num_heads=6,
        mlp_ratio=4.0,
        attn_dropout=0.0,
        dropout=0.1,
        drop_path=0.0,
    )

    y = blk(x)
    assert y.shape == x.shape

    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

import torch

from model.MaxViT_block import MaxViTBlock, MaxViTBlockConfig


def test_maxvit_block_shape():
    torch.manual_seed(0)
    b, c, h, w = 2, 64, 32, 32
    x = torch.randn(b, c, h, w)

    block = MaxViTBlock(
        MaxViTBlockConfig(
            dim=c,
            num_heads=8,
            window_size=4,
            grid_size=4,
            drop_path=0.1,
        )
    )

    y = block(x)
    assert y.shape == x.shape


def test_maxvit_block_backward():
    torch.manual_seed(0)
    b, c, h, w = 2, 32, 16, 16
    x = torch.randn(b, c, h, w, requires_grad=True)

    block = MaxViTBlock(
        MaxViTBlockConfig(
            dim=c,
            num_heads=4,
            window_size=4,
            grid_size=4,
            drop_path=0.0,
        )
    )

    y = block(x).sum()
    y.backward()
    assert x.grad is not None

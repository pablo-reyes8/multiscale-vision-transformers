import torch

from model.outlook import OutlookAttention, OutlookerBlock


def test_outlook_attention_stride1():
    torch.manual_seed(0)
    B, H, W, C = 2, 16, 16, 192
    x_map = torch.randn(B, H, W, C, requires_grad=True)

    oa = OutlookAttention(
        dim=C,
        num_heads=6,
        kernel_size=3,
        stride=1,
        attn_drop=0.0,
        proj_drop=0.0,
    )

    y = oa(x_map)
    assert y.shape == x_map.shape

    y.mean().backward()
    assert x_map.grad is not None
    assert torch.isfinite(x_map.grad).all()


def test_outlook_attention_stride2():
    torch.manual_seed(0)
    B, H, W, C = 2, 16, 16, 192
    x_map = torch.randn(B, H, W, C, requires_grad=True)

    oa = OutlookAttention(
        dim=C,
        num_heads=6,
        kernel_size=3,
        stride=2,
        attn_drop=0.0,
        proj_drop=0.0,
    )

    y = oa(x_map)
    assert y.shape == (B, H // 2, W // 2, C)

    y.mean().backward()
    assert x_map.grad is not None
    assert torch.isfinite(x_map.grad).all()


def test_outlooker_block():
    torch.manual_seed(0)
    B, H, W, C = 2, 16, 16, 192
    x_map = torch.randn(B, H, W, C, requires_grad=True)

    blk = OutlookerBlock(
        dim=C,
        num_heads=6,
        kernel_size=3,
        stride=1,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        mlp_drop=0.0,
    )

    y = blk(x_map)
    assert y.shape == x_map.shape

    y.mean().backward()
    assert x_map.grad is not None
    assert torch.isfinite(x_map.grad).all()

import pytest
import torch

from model.volo_stage import VOLOStage


def test_volo_stage_forward():
    torch.manual_seed(0)
    B, H, W, C = 2, 16, 16, 192
    x = torch.randn(B, H, W, C, requires_grad=True)

    stage = VOLOStage(
        dim=C,
        depth=3,
        num_heads=6,
        kernel_size=3,
        stride=1,
        drop_path=[0.0, 0.05, 0.1],
    )

    y = stage(x)
    assert y.shape == x.shape

    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_volo_stage_drop_path_length_check():
    with pytest.raises(AssertionError):
        VOLOStage(dim=64, depth=2, num_heads=4, drop_path=[0.1])

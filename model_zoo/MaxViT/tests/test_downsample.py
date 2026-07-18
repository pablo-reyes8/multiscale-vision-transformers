import torch

from model.downsample import Downsample, DownsampleConfig


def test_downsample_conv_shape():
    x = torch.randn(2, 64, 32, 32)
    y = Downsample(64, 128, DownsampleConfig(kind="conv"))(x)
    assert y.shape == (2, 128, 16, 16)


def test_downsample_pool_shape():
    x = torch.randn(2, 64, 32, 32)
    y = Downsample(64, 128, DownsampleConfig(kind="pool"))(x)
    assert y.shape == (2, 128, 16, 16)

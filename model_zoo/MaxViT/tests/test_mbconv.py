import torch

from model.MBConv import MBConv, MBConvConfig


def test_mbconv_stride1_residual_shape():
    x = torch.randn(2, 64, 32, 32)
    m = MBConv(64, 64, stride=1, cfg=MBConvConfig(expand_ratio=4.0, se_ratio=0.25, drop_path=0.0))
    y = m(x)
    assert y.shape == (2, 64, 32, 32)


def test_mbconv_stride2_downsample_shape():
    x = torch.randn(2, 64, 32, 32)
    m = MBConv(64, 128, stride=2, cfg=MBConvConfig(expand_ratio=4.0, se_ratio=0.25))
    y = m(x)
    assert y.shape == (2, 128, 16, 16)


def test_mbconv_no_expand():
    x = torch.randn(2, 64, 32, 32)
    m = MBConv(64, 64, stride=1, cfg=MBConvConfig(expand_ratio=1.0, se_ratio=0.25))
    y = m(x)
    assert y.shape == (2, 64, 32, 32)

import torch

from model.max_vit_stem import MaxViTStem, StemConfig


def test_stem_a_shape():
    x = torch.randn(2, 3, 32, 32)
    stem = MaxViTStem(StemConfig(stem_type="A", out_ch=64))
    y = stem(x)
    assert y.shape == (2, 64, 32, 32)


def test_stem_b_shape():
    x = torch.randn(2, 3, 32, 32)
    stem = MaxViTStem(StemConfig(stem_type="B", out_ch=64))
    y = stem(x)
    assert y.shape == (2, 64, 32, 32)

import torch

from model.attention import MultiHeadSelfAttention, AttentionConfig
from model.local_attention import LocalAttention2D, LocalAttention2DConfig


def test_mhsa_shape():
    torch.manual_seed(0)
    b, n, c = 8, 16, 64
    x = torch.randn(b, n, c)
    attn = MultiHeadSelfAttention(AttentionConfig(dim=c, num_heads=8))
    y = attn(x)
    assert y.shape == x.shape


def test_mhsa_backward():
    torch.manual_seed(0)
    b, n, c = 4, 49, 96
    x = torch.randn(b, n, c, requires_grad=True)
    attn = MultiHeadSelfAttention(AttentionConfig(dim=c, num_heads=8, attn_drop=0.1, proj_drop=0.1))
    y = attn(x).sum()
    y.backward()
    assert x.grad is not None


def test_local_attention_window_shape():
    torch.manual_seed(0)
    b, h, w, c = 2, 32, 32, 64
    x = torch.randn(b, h, w, c)
    m = LocalAttention2D(LocalAttention2DConfig(mode="window", dim=c, num_heads=8, window_size=4))
    y = m(x)
    assert y.shape == x.shape


def test_local_attention_grid_shape():
    torch.manual_seed(0)
    b, h, w, c = 2, 32, 32, 64
    x = torch.randn(b, h, w, c)
    m = LocalAttention2D(LocalAttention2DConfig(mode="grid", dim=c, num_heads=8, grid_size=4))
    y = m(x)
    assert y.shape == x.shape


def test_local_attention_backward():
    torch.manual_seed(0)
    b, h, w, c = 2, 16, 16, 32
    x = torch.randn(b, h, w, c, requires_grad=True)
    m = LocalAttention2D(LocalAttention2DConfig(mode="window", dim=c, num_heads=4, window_size=4))
    y = m(x).sum()
    y.backward()
    assert x.grad is not None

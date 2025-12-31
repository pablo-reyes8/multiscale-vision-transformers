import pytest
import torch

from model.embeddings import PatchEmbeddingConv, PosEmbed2D


def test_patch_embedding_conv_divisible():
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 64, 64
    x = torch.randn(B, C, H, W)

    pe = PatchEmbeddingConv(
        patch_size=4,
        in_chans=3,
        embed_dim=192,
        norm_layer=torch.nn.LayerNorm,
        pad_if_needed=True,
        return_tokens=True,
    )

    x_map, (Hp, Wp), x_tok, (pad_h, pad_w) = pe(x)

    assert x_map.shape == (B, H // 4, W // 4, 192)
    assert x_tok.shape == (B, (H // 4) * (W // 4), 192)
    assert (pad_h, pad_w) == (0, 0)
    assert (Hp, Wp) == (H // 4, W // 4)


def test_patch_embedding_conv_non_divisible():
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 65, 63
    x = torch.randn(B, C, H, W)

    pe = PatchEmbeddingConv(
        patch_size=4,
        in_chans=3,
        embed_dim=192,
        norm_layer=torch.nn.LayerNorm,
        pad_if_needed=True,
        return_tokens=True,
    )

    x_map, (Hp, Wp), x_tok, (pad_h, pad_w) = pe(x)

    assert (H + pad_h) % 4 == 0
    assert (W + pad_w) % 4 == 0
    assert x_map.shape == (B, Hp, Wp, 192)
    assert x_tok.shape == (B, Hp * Wp, 192)


def test_patch_embedding_conv_no_pad_raises():
    x = torch.randn(1, 3, 31, 31)
    pe = PatchEmbeddingConv(
        patch_size=4,
        in_chans=3,
        embed_dim=32,
        norm_layer=None,
        pad_if_needed=False,
        return_tokens=True,
    )

    with pytest.raises(AssertionError):
        pe(x)


def test_patch_embedding_return_tokens_false():
    x = torch.randn(1, 3, 32, 32)
    pe = PatchEmbeddingConv(
        patch_size=4,
        in_chans=3,
        embed_dim=32,
        norm_layer=None,
        pad_if_needed=True,
        return_tokens=False,
    )

    x_map, (Hp, Wp), (pad_h, pad_w) = pe(x)
    assert x_map.shape == (1, Hp, Wp, 32)
    assert (pad_h, pad_w) == (0, 0)


def test_pos_embed2d_interpolation():
    torch.manual_seed(0)
    pos = PosEmbed2D(4, 4, 8)

    x = torch.zeros(2, 16, 8)
    out = pos(x, (4, 4))
    assert out.shape == x.shape
    assert not torch.allclose(out, x)

    x_big = torch.zeros(2, 64, 8)
    out_big = pos(x_big, (8, 8))
    assert out_big.shape == x_big.shape

import torch
import torch.nn as nn

from model.patch_embedding import PatchEmbeddingConv
from model.transformer_pooling import PoolingLayer


def test_patch_embedding_conv_outputs_expected_shapes():
    torch.manual_seed(0)
    embedder = PatchEmbeddingConv(
        img_size=32,
        patch_size=4,
        in_chans=3,
        embed_dim=16,
        norm_layer=nn.LayerNorm)

    images = torch.randn(2, 3, 32, 32)
    tokens, grid = embedder(images)

    assert tokens.shape == (2, 64, 16)
    assert grid == (8, 8)


def test_pooling_layer_downsamples_grid_and_channels():
    torch.manual_seed(0)
    pooling = PoolingLayer(
        dim_in=16,
        dim_out=24,
        kernel_size=3,
        stride=2,
        norm_layer=nn.LayerNorm)

    grid = (8, 8)
    tokens = torch.randn(2, grid[0] * grid[1], 16)

    pooled, new_grid = pooling(tokens, grid)

    assert new_grid == (4, 4)
    assert pooled.shape == (2, 16, 24)

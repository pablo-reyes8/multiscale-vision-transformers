import torch
import torch.nn as nn

from model.Vit_embeddings import PatchEmbeddingConv


def test_patch_embedding_conv_outputs_expected_shapes():
    torch.manual_seed(0)
    embedder = PatchEmbeddingConv(
        patch_size=4,
        in_chans=3,
        embed_dim=16,
        norm_layer=nn.LayerNorm,
        return_tokens=True)

    images = torch.randn(2, 3, 32, 32)
    x_map, grid, tokens, pad_hw = embedder(images)

    assert x_map.shape == (2, 8, 8, 16)
    assert grid == (8, 8)
    assert tokens.shape == (2, 64, 16)
    assert pad_hw == (0, 0)

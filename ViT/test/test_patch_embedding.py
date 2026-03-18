import torch

from model.patch_embedding import PatchEmbedding


def test_patch_embedding_outputs_expected_shapes():
    torch.manual_seed(0)
    embedder = PatchEmbedding(
        img_size=32,
        patch_size=4,
        in_chans=3,
        embed_dim=64,
    )

    images = torch.randn(2, 3, 32, 32)
    tokens, grid = embedder(images)

    assert tokens.shape == (2, 64, 64)
    assert grid == (8, 8)

import torch

from model.patch_merging import PatchMerging


def test_patch_merging_downsamples_and_expands_channels():
    torch.manual_seed(0)
    merge = PatchMerging(dim=8, out_dim=16)
    x = torch.randn(1, 5, 6, 8)

    y = merge(x)

    assert y.shape == (1, 3, 3, 16)

import torch

from model.pooling_volo_blocks import VOLOPyramid


def test_volo_pyramid_map_downsample():
    torch.manual_seed(0)
    B, H, W = 2, 16, 16
    x_map = torch.randn(B, H, W, 64)

    pyr = VOLOPyramid(
        dims=(64, 96),
        outlooker_depths=(1, 0),
        outlooker_heads=(4, 4),
        transformer_depths=(0, 1),
        transformer_heads=(4, 4),
        downsample_kind="map",
        drop_path_rate=0.1,
    )

    x_tok, (Hf, Wf) = pyr(x_map)
    assert x_tok.shape[0] == B
    assert x_tok.shape[2] == 96
    assert Hf * Wf == x_tok.shape[1]


def test_volo_pyramid_token_downsample():
    torch.manual_seed(0)
    B, H, W = 2, 16, 16
    x_map = torch.randn(B, H, W, 64)

    pyr = VOLOPyramid(
        dims=(64, 96),
        outlooker_depths=(1, 0),
        outlooker_heads=(4, 4),
        transformer_depths=(0, 1),
        transformer_heads=(4, 4),
        downsample_kind="token",
        drop_path_rate=0.1,
    )

    x_tok, (Hf, Wf) = pyr(x_map)
    assert x_tok.shape[0] == B
    assert x_tok.shape[2] == 96
    assert Hf * Wf == x_tok.shape[1]

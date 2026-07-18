import torch

from model.window_partition import prepare_windows, restore_from_windows


def test_prepare_and_restore_windows_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 6, 4)

    windows, meta = prepare_windows(x, window_size=4, shift_size=2, pad_value=0.0)
    x_rec = restore_from_windows(windows, meta, C=4)

    assert x_rec.shape == x.shape
    assert torch.allclose(x_rec, x, atol=1e-6)

import torch
import pytest

from model.window_partition import window_partition, window_unpartition
from model.grid_partition import grid_partition, grid_unpartition


def test_window_roundtrip_exact():
    torch.manual_seed(0)
    b, h, w, c = 2, 32, 32, 64
    ws = 4
    x = torch.randn(b, h, w, c)

    windows = window_partition(x, ws)
    x_rec = window_unpartition(windows, ws, H=h, W=w, B=b)

    assert x_rec.shape == x.shape
    assert torch.equal(x_rec, x)


def test_grid_roundtrip_exact():
    torch.manual_seed(0)
    b, h, w, c = 2, 32, 32, 64
    g = 4
    x = torch.randn(b, h, w, c)

    grids, meta = grid_partition(x, g)
    x_rec = grid_unpartition(grids, meta)

    assert x_rec.shape == x.shape
    assert torch.equal(x_rec, x)


def test_window_invalid_divisibility_raises():
    x = torch.randn(1, 30, 32, 8)
    with pytest.raises(ValueError):
        window_partition(x, 4)


def test_grid_invalid_divisibility_raises():
    x = torch.randn(1, 32, 30, 8)
    with pytest.raises(ValueError):
        grid_partition(x, 4)

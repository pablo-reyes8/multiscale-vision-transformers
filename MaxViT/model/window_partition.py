import torch

def window_partition(x: torch.Tensor, window_size: int):
    """
    Partition windows (non-overlapping).

    Args:
        x: Tensor [B, H, W, C] (BHWC)
        window_size: int (ws)

    Returns:
        windows: [B * nW, ws, ws, C]
        where nW = (H/ws) * (W/ws)
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x.ndim==4 (BHWC). Got shape {tuple(x.shape)}")
    B, H, W, C = x.shape
    ws = window_size
    if ws <= 0:
        raise ValueError("window_size must be > 0")
    if (H % ws) != 0 or (W % ws) != 0:
        raise ValueError(f"H and W must be divisible by window_size. Got H={H}, W={W}, ws={ws}")

    x = x.view(B, H // ws, ws, W // ws, ws, C)
    # [B, H/ws, W/ws, ws, ws, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(B * (H // ws) * (W // ws), ws, ws, C)
    return windows

def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    H: int,
    W: int,
    B: int,):
    """
    Reverse window_partition.

    Args:
        windows: [B*nW, ws, ws, C]
        window_size: ws
        H, W: target spatial sizes
        B: batch size

    Returns:
        x: [B, H, W, C]
    """
    if windows.ndim != 4:
        raise ValueError(f"Expected windows.ndim==4. Got shape {tuple(windows.shape)}")
    ws = window_size
    if ws <= 0:
        raise ValueError("window_size must be > 0")
    if (H % ws) != 0 or (W % ws) != 0:
        raise ValueError(f"H and W must be divisible by ws. Got H={H}, W={W}, ws={ws}")

    _, ws_h, ws_w, C = windows.shape
    if ws_h != ws or ws_w != ws:
        raise ValueError(f"windows have ws=({ws_h},{ws_w}) but window_size={ws}")

    nW = (H // ws) * (W // ws)
    if windows.shape[0] != B * nW:
        raise ValueError(f"windows.shape[0] must be B*nW = {B*nW}. Got {windows.shape[0]}")

    x = windows.view(B, H // ws, W // ws, ws, ws, C)
    # invert permute
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, C)
    return x

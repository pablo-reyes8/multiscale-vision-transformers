import torch

def grid_partition(x: torch.Tensor, grid_size: int):
    """
    Grid partition (multi-axis / "grid attention" view).

    It groups tokens by their (i,j) offsets modulo g, producing g*g groups.
    Each group is a downsampled view with stride g.

    Args:
        x: [B, H, W, C] (BHWC)
        grid_size: g

    Returns:
        grids: [B*g*g, Hg, Wg, C] where Hg=H/g, Wg=W/g
        meta: tuple(B, H, W, C, g)
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x.ndim==4 (BHWC). Got shape {tuple(x.shape)}")
    B, H, W, C = x.shape
    g = grid_size
    if g <= 0:
        raise ValueError("grid_size must be > 0")
    if (H % g) != 0 or (W % g) != 0:
        raise ValueError(f"H and W must be divisible by grid_size. Got H={H}, W={W}, g={g}")

    Hg, Wg = H // g, W // g
    # [B, Hg, g, Wg, g, C]
    x = x.view(B, Hg, g, Wg, g, C)
    # bring offsets (g,g) into batch: [B, g, g, Hg, Wg, C] -> [B*g*g, Hg, Wg, C]
    grids = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * g * g, Hg, Wg, C)

    meta = (B, H, W, C, g)
    return grids, meta

def grid_unpartition(grids: torch.Tensor, meta) -> torch.Tensor:
    """
    Reverse grid_partition.

    Args:
        grids: [B*g*g, Hg, Wg, C]
        meta: (B, H, W, C, g) returned by grid_partition

    Returns:
        x: [B, H, W, C]
    """
    if grids.ndim != 4:
        raise ValueError(f"Expected grids.ndim==4. Got shape {tuple(grids.shape)}")
    B, H, W, C, g = meta
    if (H % g) != 0 or (W % g) != 0:
        raise ValueError("Invalid meta: H and W must be divisible by g")

    Hg, Wg = H // g, W // g
    if grids.shape[0] != B * g * g:
        raise ValueError(f"grids.shape[0] must be B*g*g = {B*g*g}. Got {grids.shape[0]}")
    if grids.shape[1] != Hg or grids.shape[2] != Wg or grids.shape[3] != C:
        raise ValueError(
            f"grids shape mismatch. Expected (*,{Hg},{Wg},{C}) got {tuple(grids.shape)}")

    x = grids.view(B, g, g, Hg, Wg, C)
    # invert permute used in grid_partition
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, H, W, C)
    return x
from outlook import *


class VOLOStage(nn.Module):
    """
    Un stage VOLO basado en OutlookerBlocks.

    Mantiene el formato channel-last:
      Input:  [B, H, W, C]
      Output: [B, H, W, C]  (si stride=1)
    Si quisieras un stage que haga downsample, usa stride>1 en los bloques
    (pero en CIFAR te recomiendo stride=1 en el stage inicial).
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        kernel_size: int = 3,
        stride: int = 1,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        mlp_drop: float = 0.0,):

        super().__init__()

        if isinstance(drop_path, float):
            dpr = [drop_path] * depth
        else:
            assert len(drop_path) == depth, "drop_path list must have length=depth"
            dpr = drop_path

        self.blocks = nn.ModuleList([
            OutlookerBlock(
                dim=dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                stride=stride,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=dpr[i],
                mlp_drop=mlp_drop,) for i in range(depth)])

    def forward(self, x_map: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x_map = blk(x_map)
        return x_map
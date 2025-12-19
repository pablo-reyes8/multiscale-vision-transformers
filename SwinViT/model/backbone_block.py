import torch 
from model.swin_block import *

class BasicLayer(nn.Module):
    """
    Un stage de Swin:
      - depth bloques SwinTransformerBlock
      - alterna shift: 0, ws//2, 0, ws//2, ...
      - downsample opcional al final (PatchMerging) excepto en el último stage
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        drop_path_rates: list[float] | None = None,
        downsample: nn.Module | None = None,
        use_rel_pos_bias: bool = True):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.downsample = downsample

        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        assert len(drop_path_rates) == depth, "drop_path_rates debe tener longitud = depth"

        blocks = []
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else (window_size // 2)
            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    attn_dropout=attn_dropout,
                    proj_dropout=proj_dropout,
                    mlp_dropout=mlp_dropout,
                    drop_path=drop_path_rates[i],
                    use_rel_pos_bias=use_rel_pos_bias,))
            
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor):
        """
        x: [B, H, W, C]
        """
        for blk in self.blocks:
            x = blk(x)

        x_down = None
        if self.downsample is not None:
            x_down = self.downsample(x)
        return x, x_down
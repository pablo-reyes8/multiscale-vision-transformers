import torch.nn.functional as F
import torch
import torch.nn as nn

class PoolingLayer(nn.Module):
    """
    Pooling jerárquico para ViT:

    - Toma tokens [B, N, D_in] + grid_size (H, W)
    - Los reinterpreta como feature map [B, D_in, H, W]
    - Aplica:
        depthwise conv (3x3, stride=2, padding=1)
        pointwise conv (1x1) para cambiar D_in -> D_out
    - Devuelve:
        tokens [B, N_out, D_out] y nuevo grid_size (H_out, W_out)
    """

    def __init__(self,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 3,
        stride: int = 2,
        norm_layer: type[nn.Module] | None = nn.LayerNorm):

        super().__init__()
        padding = kernel_size // 2

        # Depthwise conv: cada canal se filtra por separado
        self.depthwise_conv = nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=dim_in)

        # Pointwise conv: mezcla canales y cambia dim
        self.pointwise_conv = nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=1,
            stride=1,
            padding=0)

        self.norm = norm_layer(dim_out) if norm_layer is not None else None

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.stride = stride

    def forward(self, x: torch.Tensor, grid_size: tuple[int, int]):
        """
        Args:
            x: tokens [B, N, D_in]
            grid_size: (H, W) tal que H*W = N

        Returns:
            x_out: tokens [B, N_out, D_out]
            new_grid: (H_out, W_out)
        """
        B, N, D_in = x.shape
        H, W = grid_size

        assert D_in == self.dim_in, f"dim_in {D_in} != {self.dim_in}"
        assert H * W == N, f"H*W={H*W} no coincide con N={N}"

        # [B, N, D_in] -> [B, D_in, H, W]
        x = x.view(B, H, W, D_in).permute(0, 3, 1, 2)

        # Depthwise + pointwise
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        B, D_out, H_out, W_out = x.shape
        N_out = H_out * W_out

        # Volver a tokens: [B, D_out, H_out, W_out] -> [B, N_out, D_out]
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        new_grid = (H_out, W_out)
        return x, new_grid
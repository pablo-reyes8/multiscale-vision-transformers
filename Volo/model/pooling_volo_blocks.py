import torch
import torch.nn as nn
import torch.nn.functional as F

from model.droppath import *
from model.attention import *
from model.volo_stage import *

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


class MapDownsample(nn.Module):
    """
    Downsample para mapas channel-last: [B, H, W, C_in] -> [B, H/2, W/2, C_out]
    usando conv2d stride=2 en formato channel-first internamente.
    """
    def __init__(self, dim_in: int, dim_out: int, kernel_size: int = 3, norm_layer=nn.LayerNorm):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=2, padding=pad, bias=True)
        self.norm = norm_layer(dim_out) if norm_layer is not None else None

    def forward(self, x_map: torch.Tensor):
        # x_map: [B, H, W, C_in]
        B, H, W, C = x_map.shape
        x = x_map.permute(0, 3, 1, 2).contiguous()     # [B, C, H, W]
        x = self.conv(x)                               # [B, C_out, H2, W2]
        x_map = x.permute(0, 2, 3, 1).contiguous()     # [B, H2, W2, C_out]
        if self.norm is not None:
            x_map = self.norm(x_map)
        return x_map
    
###############################################################



def map_to_tokens(x_map: torch.Tensor) -> torch.Tensor:
    B, H, W, C = x_map.shape
    return x_map.view(B, H * W, C)

def tokens_to_map(x_tok: torch.Tensor, H: int, W: int) -> torch.Tensor:
    B, N, C = x_tok.shape
    assert N == H * W
    return x_tok.view(B, H, W, C)

class VOLOPyramid(nn.Module):
    """
    Backbone jerárquico para VOLO (sin classifier head aún).
    - Local: VOLOStage (Outlooker)
    - Global: TransformerStack (opcional)
    - Downsample: map-space (recomendado) o token-space (PoolingLayer tuyo)
    """
    def __init__(
        self,
        dims: tuple[int, ...],                 # ej (192, 256, 384)
        outlooker_depths: tuple[int, ...],     # ej (4, 2, 0)  (0 si no hay outlooker en ese nivel)
        outlooker_heads: tuple[int, ...],      # ej (6, 8, 12)
        transformer_depths: tuple[int, ...],   # ej (0, 4, 6)
        transformer_heads: tuple[int, ...],    # ej (6, 8, 12)
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        downsample_kind: str = "map",          # "map" o "token"
        drop_path_rate: float = 0.0):

        super().__init__()
        L = len(dims)

        assert len(outlooker_depths) == L
        assert len(outlooker_heads) == L
        assert len(transformer_depths) == L
        assert len(transformer_heads) == L

        # schedule lineal de droppath a través de todos los bloques (local+global)
        total_blocks = sum(outlooker_depths) + sum(transformer_depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist() if total_blocks > 0 else []
        dp_i = 0

        self.levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.downsample_kind = downsample_kind

        for i in range(L):
            dim = dims[i]

            # Local stage (Outlooker)
            local = None
            if outlooker_depths[i] > 0:
                local_dpr = dpr[dp_i: dp_i + outlooker_depths[i]]
                dp_i += outlooker_depths[i]
                local = VOLOStage(
                    dim=dim,
                    depth=outlooker_depths[i],
                    num_heads=outlooker_heads[i],
                    kernel_size=kernel_size,
                    stride=1,
                    mlp_ratio=mlp_ratio,
                    drop_path=local_dpr)

            # Global stage (Transformer)
            global_ = None
            if transformer_depths[i] > 0:
                glob_dpr = dpr[dp_i: dp_i + transformer_depths[i]]
                dp_i += transformer_depths[i]

                global_ = TransformerStack(
                    dim=dim,
                    depth=transformer_depths[i],
                    num_heads=transformer_heads[i],
                    mlp_ratio=mlp_ratio,
                    attn_dropout=0.0,
                    dropout=0.1,
                    drop_path=glob_dpr,)

            self.levels.append(nn.ModuleDict({"local": local, "global": global_}))

            # Downsample para pasar dim_i -> dim_{i+1} (si no es el último nivel)
            if i < L - 1:
                if downsample_kind == "map":
                    self.downsamples.append(MapDownsample(dim_in=dim, dim_out=dims[i + 1], kernel_size=3))
                elif downsample_kind == "token":
                    # reusar PoolingLayer
                    self.downsamples.append(PoolingLayer(dim_in=dim, dim_out=dims[i + 1], kernel_size=3, stride=2))
                else:
                    raise ValueError(f"downsample_kind must be 'map' or 'token'. Got {downsample_kind}")

        assert dp_i == total_blocks

    def forward(self, x_map: torch.Tensor):
        """
        x_map: [B, H, W, C0]
        returns:
          x_final_tokens: [B, N_last, C_last]
          last_grid: (H_last, W_last)
        """
        B, H, W, C = x_map.shape

        for i, lvl in enumerate(self.levels):
            # local stage en map
            if lvl["local"] is not None:
                x_map = lvl["local"](x_map)

            # global stage en tokens (si existe)
            if lvl["global"] is not None:
                x_tok = map_to_tokens(x_map)
                x_tok = lvl["global"](x_tok)
                x_map = tokens_to_map(x_tok, H, W)

            # downsample (si aplica)
            if i < len(self.downsamples):
                ds = self.downsamples[i]
                if self.downsample_kind == "map":
                    x_map = ds(x_map)
                    H, W = x_map.shape[1], x_map.shape[2]
                else:
                    # token downsample: necesita grid
                    x_tok = map_to_tokens(x_map)
                    x_tok, (H, W) = ds(x_tok, (H, W))
                    x_map = tokens_to_map(x_tok, H, W)

        x_final = map_to_tokens(x_map)
        return x_final, (H, W)

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.droppath import *


class OutlookAttention(nn.Module):
    """
    Outlook Attention (VOLO): agregación local dinámica sobre ventanas.

    Entrada:  x_map [B, H, W, C]  (channel-last)
    Salida:   y_map [B, H, W, C]

    Parámetros:
      - dim: canales C
      - kernel_size: k (vecindario k×k)
      - stride: s (si s>1 hace downsample tipo "outlook pooling"; para CIFAR típicamente s=1)
      - num_heads: h (partimos canales en cabezas, como MHSA)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        kernel_size: int = 3,
        stride: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,):

        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.kernel_size = kernel_size
        self.stride = stride

        # Genera pesos de atención por posición: [B, H, W, heads * k*k]
        self.attn = nn.Linear(dim, num_heads * kernel_size * kernel_size, bias=True)

        # Proyección para values (antes de unfold)
        self.v = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_map: torch.Tensor) -> torch.Tensor:
        """
        x_map: [B, H, W, C]
        """
        B, H, W, C = x_map.shape
        k = self.kernel_size
        s = self.stride
        heads = self.num_heads
        hd = self.head_dim

        # attention weights
        a = self.attn(x_map)
        # si stride>1, la atención se evalúa en posiciones downsampled
        if s > 1:
            # downsample espacialmente (simple avg pool sobre channel-last)
            a = a.permute(0, 3, 1, 2)                       # [B, heads*k*k, H, W]
            a = F.avg_pool2d(a, kernel_size=s, stride=s)    # [B, heads*k*k, Hs, Ws]
            a = a.permute(0, 2, 3, 1).contiguous()          # [B, Hs, Ws, heads*k*k]

        Hs, Ws = a.shape[1], a.shape[2]
        a = a.view(B, Hs * Ws, heads, k * k)
        a = F.softmax(a, dim=-1)
        a = self.attn_drop(a)

        # values map
        v = self.v(x_map)
        v = v.permute(0, 3, 1, 2).contiguous()

        # unfold extrae vecindarios k×k para cada posición (con padding para "same")
        pad = k // 2
        v_unf = F.unfold(v, kernel_size=k, padding=pad, stride=s)
        v_unf = v_unf.view(B, C, k * k, Hs * Ws).permute(0, 3, 1, 2).contiguous()
        v_unf = v_unf.view(B, Hs * Ws, heads, hd, k * k)

        # apply attention: weighted sum over neighborhood
        # a:     [B, Hs*Ws, heads, k*k]
        # v_unf: [B, Hs*Ws, heads, hd, k*k]
        y = (v_unf * a.unsqueeze(3)).sum(dim=-1)
        y = y.reshape(B, Hs * Ws, C)              # concat heads

        # fold back to spatial map
        y_map = y.view(B, Hs, Ws, C)

        y_map = self.proj(y_map)
        y_map = self.proj_drop(y_map)
        return y_map
    


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class OutlookerBlock(nn.Module):
    """
    Bloque VOLO Outlooker:
      x -> LN -> OutlookAttention -> DropPath + residual
        -> LN -> MLP -> DropPath + residual

    Input/Output: [B, H, W, C]
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int = 3,
        stride: int = 1,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        mlp_drop: float = 0.0):

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        self.attn = OutlookAttention(
            dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            stride=stride,
            attn_drop=attn_drop,
            proj_drop=proj_drop,)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)

        self.mlp = MLP(dim=dim, hidden_dim=hidden_dim, drop=mlp_drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x_map: torch.Tensor) -> torch.Tensor:
        """
        x_map: tensor de forma (B, C, H, W) o (B, N, C), según el bloque.
        """

        # Primer sub-bloque: Norm -> Atención -> DropPath -> Residual

        # Normalización del input
        x_norm_1 = self.norm1(x_map)

        # Atención
        attn_out = self.attn(x_norm_1)
        attn_out = self.drop_path1(attn_out)

        # Suma residual
        x_map = x_map + attn_out

        # Segundo sub-bloque: Norm -> MLP -> DropPath -> Residual ---

        x_norm_2 = self.norm2(x_map)

        # MLP
        mlp_out = self.mlp(x_norm_2)
        mlp_out = self.drop_path2(mlp_out)

        # Segunda suma residual
        x_out = x_map + mlp_out

        return x_out
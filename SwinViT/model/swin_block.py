import torch
import torch.nn as nn

from model.window_partition import *
from model.swin_attention import *

class DropPath(nn.Module):
    """Stochastic Depth (por muestra)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape: [B, 1, 1, 1] para [B,H,W,C]
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = torch.floor(rand)
        return x / keep_prob * mask


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Bloque Swin completo (pre-norm) en formato [B, H, W, C].

    Usa:
      - prepare_windows(x, window_size, shift_size) -> (windows_flat, meta)
      - restore_from_windows(windows_flat, meta, C) -> x_rec
      - WindowAttention(dim, window_size, num_heads)(windows_flat, mask)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        drop_path: float = 0.0,
        use_rel_pos_bias: bool = True):

        super().__init__()
        assert 0 <= shift_size < window_size, "shift_size debe estar en [0, window_size)"
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)

        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            use_rel_pos_bias=use_rel_pos_bias,)

        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim=dim, hidden_dim=hidden_dim, dropout=mlp_dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor):
        """
        x: [B, H, W, C]
        """
        B, H, W, C = x.shape
        assert C == self.dim, f"Canales C={C} != dim={self.dim}"
        # Si H/W son menores que window_size, el padding en prepare_windows
        # completa hasta múltiplos de window_size, evitando errores por tamaños pequeños.

        #  Window Attention
        shortcut = x
        x_norm = self.norm1(x)

        windows_flat, meta = prepare_windows(
            x_norm,
            window_size=self.window_size,
            shift_size=self.shift_size,
            pad_value=0.0)

        attn_mask = meta["attn_mask"]  # None o [nW, N, N] con 0/-inf

        out_windows = self.attn(windows_flat, mask=attn_mask)  # [B*nW, N, C]
        x_attn = restore_from_windows(out_windows, meta, C=C)  # [B, H, W, C]

        x = shortcut + self.drop_path1(x_attn)

        # MLP
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x

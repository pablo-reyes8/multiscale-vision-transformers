import torch
import torch.nn as nn
import torch.nn.functional as F

from model.droppath import * 

def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout_p: float = 0.0, training: bool = True):
    """
    q: (B, H, Lq, d)
    k: (B, H, Lk, d)
    v: (B, H, Lk, d)
    mask: broadcastable a (B, H, Lq, Lk)
          - bool: True = BLOQUEAR (poner -inf)
          - float: 1.0 = permitir, 0.0 = bloquear
    """
    scores = torch.matmul(q, k.transpose(-2, -1))
    dk = q.size(-1)
    scores = scores / (dk ** 0.5)

    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask, float("-inf"))
        else:
            scores = scores.masked_fill(mask <= 0, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    if attn_dropout_p > 0.0:
        attn = F.dropout(attn, p=attn_dropout_p, training=training)

    output = torch.matmul(attn, v)
    return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser múltiplo de num_heads"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # "dropout" lo usaremos como dropout de atención (sobre attn)
        self.attn_dropout_p = dropout
        # y también dejamos dropout de salida si quieres (común en ViT)
        self.out_dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_head).transpose(1, 2)

    def _combine_heads(self, x):
        B, H, L, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * D)

    def forward(self, x_q, x_kv, mask=None):
        q = self._split_heads(self.w_q(x_q))
        k = self._split_heads(self.w_k(x_kv))
        v = self._split_heads(self.w_v(x_kv))

        if mask is not None:
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]
            elif mask.dim() == 4:
                pass
            else:
                raise ValueError(f"Máscara con dims no soportadas: {mask.shape}")

            if mask.dtype != torch.bool:
                mask = (mask <= 0)

        attn_out, _ = scaled_dot_product_attention(
            q, k, v,
            mask=mask,
            attn_dropout_p=self.attn_dropout_p,
            training=self.training)

        attn_out = self._combine_heads(attn_out)

        attn_out = self.w_o(attn_out)
        attn_out = self.out_dropout(attn_out)
        return attn_out
    
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Bloque encoder para ViT (pre-norm):
    x -> LN -> MHA -> DropPath -> +residual
       -> LN -> MLP -> DropPath -> +residual
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
        drop_path: float = 0.0):

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(d_model=dim, num_heads=num_heads, dropout=attn_dropout)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, hidden_dim, dropout=dropout)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x), self.norm1(x), mask=None))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
  
    
class TransformerStack(nn.Module):
    """Stack simple de TransformerBlock sobre tokens [B, N, C]."""
    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio=4.0,
                 attn_dropout=0.0, dropout=0.1, drop_path: float | list[float] = 0.0):
        super().__init__()
        if isinstance(drop_path, float):
            dpr = [drop_path] * depth
        else:
            assert len(drop_path) == depth
            dpr = drop_path

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_dropout=attn_dropout,
                dropout=dropout,
                drop_path=dpr[i] if "drop_path" in TransformerBlock.__init__.__code__.co_varnames else 0.0) for i in range(depth)])

    def forward(self, x_tok: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x_tok = blk(x_tok)
        return x_tok
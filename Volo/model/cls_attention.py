import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

from model.droppath import *
from model.attention import *


class ClassAttention(nn.Module):
    """
    Class Attention: sólo el CLS atiende al conjunto [CLS | tokens].
    Inputs:
      cls:    [B, 1, C]
      tokens: [B, N, C]
    Output:
      cls_out: [B, 1, C]
    """
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(d_model=dim, num_heads=num_heads, dropout=attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, cls: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        kv = torch.cat([cls, tokens], dim=1)        # [B, 1+N, C]
        cls_out = self.attn(cls, kv, mask=None) # [B, 1, C] (solo CLS sale actualizado)
        return self.proj_drop(cls_out)


class ClassAttentionBlock(nn.Module):
    """
    Pre-norm (CaiT-style):
      cls -> LN -> ClassAttn(cls, [cls|tokens]) -> +res
          -> LN -> MLP -> +res
    Nota: tokens NO se actualizan.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.0):

        super().__init__()
        self.norm_cls = nn.LayerNorm(dim)
        self.norm_tok = nn.LayerNorm(dim)
        self.ca = ClassAttention(dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, cls: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        # Class attention update (solo CLS)
        cls_norm = self.norm_cls(cls)
        tok_norm = self.norm_tok(tokens)
        cls = cls + self.ca(cls_norm, tok_norm)

        # MLP update (solo CLS)
        cls = cls + self.mlp(self.norm2(cls))
        return cls
    
class CLIPool(nn.Module):
    """
    "CLI" style pooling: mezcla aprendible entre CLS y mean(tokens).
      z = alpha * cls + (1-alpha) * mean
    """
    def __init__(self, init_alpha: float = 0.5):
        super().__init__()
        # parametriza alpha en logits para mantenerlo en (0,1)
        init_alpha = float(init_alpha)
        init_alpha = min(max(init_alpha, 1e-4), 1 - 1e-4)
        logit = log(init_alpha / (1 - init_alpha))
        self.alpha_logit = nn.Parameter(torch.tensor([logit], dtype=torch.float32))

    def forward(self, cls_vec: torch.Tensor, tok_mean: torch.Tensor) -> torch.Tensor:
        """
        cls_vec:  [B, C]
        tok_mean: [B, C]
        """
        alpha = torch.sigmoid(self.alpha_logit)  # scalar in (0,1)
        return alpha * cls_vec + (1.0 - alpha) * tok_mean
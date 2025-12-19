import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
    """
    Window Multi-Head Self-Attention (W-MSA / SW-MSA).

    Entrada:
      x: [B*nW, N, C]  donde N = ws*ws
      mask (opcional): [nW, N, N] con 0 permitido y -inf bloqueado (tu máscara Swin)
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_rel_pos_bias: bool = True):

        super().__init__()
        assert dim % num_heads == 0, "dim debe ser múltiplo de num_heads"

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rel_pos_bias = use_rel_pos_bias

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        if use_rel_pos_bias:
            ws = window_size
            num_rel = (2 * ws - 1) * (2 * ws - 1)
            self.rel_pos_bias_table = nn.Parameter(torch.zeros(num_rel, num_heads))

            # índice relativo [N, N] para mapear pares (i,j) -> entrada en la tabla
            coords_h = torch.arange(ws)
            coords_w = torch.arange(ws)
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            coords_flat = coords.flatten(1)
            rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
            rel_coords = rel_coords.permute(1, 2, 0).contiguous()
            rel_coords[:, :, 0] += ws - 1
            rel_coords[:, :, 1] += ws - 1
            rel_coords[:, :, 0] *= (2 * ws - 1)
            rel_pos_index = rel_coords.sum(-1)  #
            self.register_buffer("rel_pos_index", rel_pos_index, persistent=False)

            nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        x: [BnW, N, C]
        mask: [nW, N, N] float con 0 / -inf
        """
        BnW, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(BnW, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.use_rel_pos_bias:
            rel_bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)]
            rel_bias = rel_bias.view(N, N, self.num_heads).permute(2, 0, 1).contiguous()
            attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, N, N] -> se aplica por ventana compartir en batch
            nW = mask.shape[0]
            assert BnW % nW == 0, "BnW debe ser múltiplo de nW para aplicar máscara Swin."

            B = BnW // nW

            attn = attn.view(B, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(BnW, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(BnW, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

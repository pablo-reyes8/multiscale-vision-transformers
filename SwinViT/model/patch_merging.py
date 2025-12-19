import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchMerging(nn.Module):
    """
    Patch Merging (Swin): [B, H, W, C] -> [B, H/2, W/2, 2C]

    Pasos:
      1) (opcional) pad si H o W es impar
      2) agrupa 2x2 tokens y concatena canales -> 4C
      3) LayerNorm(4C)
      4) Linear(4C -> 2C)
    """

    def __init__(self, dim: int, out_dim: int | None = None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else 2 * dim

        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: [B, H, W, C]
        """
        assert x.dim() == 4, "x debe ser [B, H, W, C]"
        B, H, W, C = x.shape
        assert C == self.dim, f"C={C} != dim={self.dim}"

        # Pad si H o W es impar (Swin lo hace para poder agrupar 2x2)
        pad_h = H % 2
        pad_w = W % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H = H + pad_h
            W = W + pad_w

        # Tomar 2x2:
        # x0 = (even rows, even cols), x1 = (odd rows, even cols), x2 = (even rows, odd cols), x3 = (odd rows, odd cols)
        x0 = x[:, 0::2, 0::2, :]  
        x1 = x[:, 1::2, 0::2, :]  
        x2 = x[:, 0::2, 1::2, :]  
        x3 = x[:, 1::2, 1::2, :]  

        # Concatenar en canal: [B, H/2, W/2, 4C]
        x = torch.cat([x0, x1, x2, x3], dim=-1)

        # Norm + Linear: [B, H/2, W/2, 2C]
        x = self.norm(x)
        x = self.reduction(x)

        return x
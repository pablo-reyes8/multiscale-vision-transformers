import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbeddingConv(nn.Module):
    """
    Patch embedding estilo Swin.

    - Conv2d con kernel=stride=patch_size para convertir imagen -> grilla de patches.
    - Devuelve el mapa 2D en formato canal-al-final: [B, Hp, Wp, D],
      (más cómodo para window partition).
    - Opcionalmente devuelve tokens [B, N, D].
    - Opcional padding automático si H/W no son divisibles por patch_size.
    """

    def __init__(
        self,
        patch_size: int | tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
        norm_layer: type[nn.Module] | None = nn.LayerNorm,
        pad_if_needed: bool = True,
        return_tokens: bool = True):

        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.patch_size = patch_size  # (Ph, Pw)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.pad_if_needed = pad_if_needed
        self.return_tokens = return_tokens

        # [B, C, H, W] -> [B, D, Hp, Wp]
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,)

        # En Swin normalmente LayerNorm sobre la última dimensión
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W]

        Returns:
            x_map:    [B, Hp, Wp, D]
            (Hp, Wp): tamaño espacial en patches
            x_tokens (opcional): [B, N, D]
            pad_hw (opcional): (pad_h, pad_w) aplicados a la imagen
        """
        B, C, H, W = x.shape
        Ph, Pw = self.patch_size

        pad_h = (Ph - (H % Ph)) % Ph
        pad_w = (Pw - (W % Pw)) % Pw

        if (pad_h != 0 or pad_w != 0):
            if not self.pad_if_needed:
                raise AssertionError(
                    f"Image size ({H}x{W}) no es divisible por patch_size {self.patch_size} "
                    f"y pad_if_needed=False.")

            x = F.pad(x, (0, pad_w, 0, pad_h))

        # [B, D, Hp, Wp]
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        # canal al final -> [B, Hp, Wp, D]
        x_map = x.permute(0, 2, 3, 1).contiguous()

        if self.norm is not None:
            x_map = self.norm(x_map)

        if self.return_tokens:
            x_tokens = x_map.view(B, Hp * Wp, self.embed_dim)
            return x_map, (Hp, Wp), x_tokens, (pad_h, pad_w)

        return x_map, (Hp, Wp), (pad_h, pad_w)
    
    
def trunc_normal_(tensor, mean=0., std=1.):
    with torch.no_grad():
        return tensor.normal_(mean=mean, std=std)
    
class PosEmbed2D(nn.Module):
    """
    Positional embedding aprendible para grilla (H, W) en tokens.

    Guarda [1, H*W, C]. Si en forward llega otro (H,W), interpola.
    """
    def __init__(self, H: int, W: int, dim: int):
        super().__init__()
        self.H0 = H
        self.W0 = W
        self.dim = dim
        self.pos = nn.Parameter(torch.zeros(1, H * W, dim))
        trunc_normal_(self.pos, std=0.02)

    def forward(self, x_tok: torch.Tensor, grid: tuple[int, int]):
        """
        x_tok: [B, N, C]
        grid: (H, W)
        """
        B, N, C = x_tok.shape
        H, W = grid
        if (H == self.H0) and (W == self.W0):
            return x_tok + self.pos

        # Interpola pos emb como mapa [1, C, H, W] -> nuevo tamaño
        pos = self.pos.reshape(1, self.H0, self.W0, self.dim).permute(0, 3, 1, 2)  # [1,C,H0,W0]
        pos = nn.functional.interpolate(pos, size=(H, W), mode="bicubic", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, H * W, self.dim)
        return x_tok + pos
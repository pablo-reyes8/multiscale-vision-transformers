import torch
import torch.nn as nn

class PatchEmbeddingConv(nn.Module):
    """
    Conv-stem para ViT jerárquico.

    - Usa una Conv2d con kernel = patch_size y stride = patch_size
      para "cortar" la imagen en parches y proyectarlos a embed_dim.
    - Devuelve tokens [B, N, D] y el tamaño espacial (H_p, W_p).
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 32,
        patch_size: int | tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
        norm_layer: type[nn.Module] | None = None):

        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.img_size = img_size        # (H, W)
        self.patch_size = patch_size    # (Ph, Pw)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Número de patches por dimensión
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],)

        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Proyección conv: de [B, C, H, W] -> [B, D, H_p, W_p]
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,)

        # Normalizacion
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: tensor de imágenes [B, C, H, W]

        Returns:
            tokens: [B, N, D]  (N = H_p * W_p)
            (H_p, W_p): tamaño de la grilla de patches
        """
        B, C, H, W = x.shape
        Ph, Pw = self.patch_size

        assert H % Ph == 0 and W % Pw == 0, \
            f"Image size ({H}x{W}) no es divisible por patch_size {self.patch_size}"

        # Conv2d: [B, C, H, W] -> [B, D, H_p, W_p]
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        # Aplanar dimensión espacial: [B, D, H_p, W_p] -> [B, D, N]
        x = x.flatten(2)
        # Reordenar a tokens: [B, N, D]
        x = x.transpose(1, 2)

        # Normalizacion
        if self.norm is not None:
            x = self.norm(x)

        return x, (Hp, Wp)
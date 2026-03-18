import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 32,
        patch_size: int | tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
        norm_layer: type[nn.Module] | None = None,
    ):
        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor):
        _, _, height, width = x.shape
        patch_h, patch_w = self.patch_size

        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"Image size ({height}x{width}) must be divisible by patch_size {self.patch_size}"
            )

        x = self.proj(x)
        grid_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x, grid_size

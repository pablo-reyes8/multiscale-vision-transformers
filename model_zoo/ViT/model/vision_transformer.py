import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention_blocks import TransformerEncoderBlock
from model.patch_embedding import PatchEmbedding


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 100,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        patch_norm: bool = False,
        drop_rate: float = 0.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        norm_layer = nn.LayerNorm if patch_norm else None
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.num_patches = self.patch_embed.num_patches
        self.base_grid_size = self.patch_embed.grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        drop_path_values = torch.linspace(0, drop_path_rate, depth).tolist() if depth > 0 else []
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                mlp_dropout=mlp_dropout,
                drop_path=drop_path_values[idx],
            )
            for idx in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.0)

    def _interpolate_pos_encoding(self, grid_size: tuple[int, int]) -> torch.Tensor:
        target_h, target_w = grid_size
        base_h, base_w = self.base_grid_size

        if (target_h, target_w) == (base_h, base_w):
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        patch_pos = patch_pos.reshape(1, base_h, base_w, self.embed_dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, target_h * target_w, self.embed_dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward_features(self, x: torch.Tensor):
        x, grid_size = self.patch_embed(x)
        batch_size = x.shape[0]

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self._interpolate_pos_encoding(grid_size)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls = x[:, 0]
        patch_tokens = x[:, 1:]
        return cls, patch_tokens, grid_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls, _, _ = self.forward_features(x)
        return self.head(cls)

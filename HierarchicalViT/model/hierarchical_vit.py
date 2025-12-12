import torch.nn.functional as F
import torch
import torch.nn as nn

from model.attention_blocks import * 
from model.transformer_pooling import *
from model.patch_embedding import *


class HierarchicalViT(nn.Module):
    """
    Vision Transformer jerárquico con pooling estilo PiT-lite,
    pensado para CIFAR (32x32) pero configurable.

    Pipeline:
    - PatchEmbeddingConv -> Stage 1 (L1 bloques)
    - PoolingLayer       -> Stage 2 (L2 bloques)
    - PoolingLayer       -> Stage 3 (L3 bloques)
    - Global average pooling sobre tokens -> Head de clasificación
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 100,
        embed_dims=(192, 384, 576),
        depths=(2, 2, 4),
        num_heads=(3, 6, 9),
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.1):

        super().__init__()

        assert len(embed_dims) == len(depths) == len(num_heads), \
            "embed_dims, depths y num_heads deben tener misma longitud"
        self.num_stages = len(embed_dims)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        # Patch embedding inicial (Stage 0)
        self.patch_embed = PatchEmbeddingConv(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            norm_layer=nn.LayerNorm)

        # Grid sizes por stage (asumiendo stride=2 en cada pooling)
        H0 = img_size // patch_size
        W0 = img_size // patch_size
        stage_grid_sizes = []
        H, W = H0, W0

        for s in range(self.num_stages):
            stage_grid_sizes.append((H, W))
            if s < self.num_stages - 1:
                H //= 2
                W //= 2
        self.stage_grid_sizes = stage_grid_sizes

        # Positional embeddings por stage
        pos_embeds = []
        for s in range(self.num_stages):
            H_s, W_s = self.stage_grid_sizes[s]
            N_s = H_s * W_s
            pe = nn.Parameter(torch.zeros(1, N_s, embed_dims[s]))
            pos_embeds.append(pe)

        self.pos_embeds = nn.ParameterList(pos_embeds)

        #  Stages: listas de bloques Transformer
        stages = []
        for s in range(self.num_stages):
            dim = embed_dims[s]
            heads = num_heads[s]
            depth = depths[s]

            blocks = nn.ModuleList([
                TransformerBlock(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    attn_dropout=attn_dropout,
                    dropout=dropout) for _ in range(depth)])

            stages.append(blocks)

        self.stages = nn.ModuleList(stages)

        # Pooling layers entre stages (num_stages - 1)
        pools = []

        for s in range(self.num_stages - 1):
            dim_in = embed_dims[s]
            dim_out = embed_dims[s + 1]

            pool = PoolingLayer(
                dim_in=dim_in,
                dim_out=dim_out,
                kernel_size=3,
                stride=2,
                norm_layer=nn.LayerNorm,)

            pools.append(pool)

        self.pool_layers = nn.ModuleList(pools)

        # - Head de clasificación
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        for pe in self.pos_embeds:
            nn.init.trunc_normal_(pe, std=0.02)

        nn.init.trunc_normal_(self.head.weight, std=0.02)

        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.)

    def forward_features(self, x: torch.Tensor):
        """
        Pasa la imagen por todos los stages y devuelve el embedding final
        antes de la cabeza de clasificación.
        """
        B = x.size(0)

        #  Stage 0: patch embedding
        x, grid = self.patch_embed(x)
        H, W = grid

        assert (H, W) == self.stage_grid_sizes[0], \
            f"Grid inicial {grid} != esperado {self.stage_grid_sizes[0]}"

        # Stages jerárquicos
        for s in range(self.num_stages):
            # Añadir positional embedding del stage s
            H_s, W_s = self.stage_grid_sizes[s]
            N_s = H_s * W_s

            assert x.shape[1] == N_s, \
                f"Stage {s}: N tokens {x.shape[1]} != N_s esperado {N_s}"

            x = x + self.pos_embeds[s]

            for blk in self.stages[s]:
                x = blk(x)

            # Si no es el último stage, aplicar pooling
            if s < self.num_stages - 1:
                x, grid = self.pool_layers[s](x, grid)
                H, W = grid

        # Al final: x [B, N_final, D_final]
        # Global average pooling sobre tokens
        x = x.mean(dim=1)
        x = self.norm_head(x)
        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        x = self.head(x)
        return x
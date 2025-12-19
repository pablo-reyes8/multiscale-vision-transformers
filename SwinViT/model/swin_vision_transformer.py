import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Vit_embeddings import * 
from model.backbone_block import *
from model.patch_merging import *

class SwinTransformer(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 100,
        embed_dim: int = 96,
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        num_heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        use_rel_pos_bias: bool = True,):
        super().__init__()

        if isinstance(img_size, int):
            img_h, img_w = img_size, img_size
        else:
            img_h, img_w = img_size

        self.patch_embed = PatchEmbeddingConv(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
            pad_if_needed=True,
            return_tokens=False)

        self.pos_drop = nn.Dropout(drop_rate)

        # ---- window_size por stage (clip a la resolución del stage) ----
        Hp = math.ceil(img_h / patch_size)
        Wp = math.ceil(img_w / patch_size)
        stage_res = [(math.ceil(Hp / (2**i)), math.ceil(Wp / (2**i))) for i in range(4)]
        ws = [max(1, min(window_size, h, w)) for (h, w) in stage_res]
        self.window_sizes = tuple(ws)  # para debug
        # ---------------------------------------------------------------

        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        # Stage 1
        self.layer1 = BasicLayer(
            dim=dims[0], depth=depths[0], num_heads=num_heads[0],
            window_size=ws[0], mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout, proj_dropout=proj_dropout, mlp_dropout=mlp_dropout,
            drop_path_rates=dpr[0:depths[0]],
            downsample=PatchMerging(dim=dims[0], out_dim=dims[1]),
            use_rel_pos_bias=use_rel_pos_bias,)

        # Stage 2
        idx = depths[0]
        self.layer2 = BasicLayer(
            dim=dims[1], depth=depths[1], num_heads=num_heads[1],
            window_size=ws[1], mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout, proj_dropout=proj_dropout, mlp_dropout=mlp_dropout,
            drop_path_rates=dpr[idx:idx + depths[1]],
            downsample=PatchMerging(dim=dims[1], out_dim=dims[2]),
            use_rel_pos_bias=use_rel_pos_bias,)

        # Stage 3
        idx += depths[1]
        self.layer3 = BasicLayer(
            dim=dims[2], depth=depths[2], num_heads=num_heads[2],
            window_size=ws[2], mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout, proj_dropout=proj_dropout, mlp_dropout=mlp_dropout,
            drop_path_rates=dpr[idx:idx + depths[2]],
            downsample=PatchMerging(dim=dims[2], out_dim=dims[3]),
            use_rel_pos_bias=use_rel_pos_bias,)

        # Stage 4
        idx += depths[2]
        self.layer4 = BasicLayer(
            dim=dims[3], depth=depths[3], num_heads=num_heads[3],
            window_size=ws[3], mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout, proj_dropout=proj_dropout, mlp_dropout=mlp_dropout,
            drop_path_rates=dpr[idx:idx + depths[3]],
            downsample=None,
            use_rel_pos_bias=use_rel_pos_bias)

        self.norm = nn.LayerNorm(dims[3])
        self.head = nn.Linear(dims[3], num_classes)

    def _build_patch_mask(self, x: torch.Tensor, pad_hw: tuple[int, int]):
        B, _, H, W = x.shape
        pad_h, pad_w = pad_hw
        mask = x.new_ones((B, 1, H, W))
        if pad_h or pad_w:
            mask = F.pad(mask, (0, pad_w, 0, pad_h))

        Ph, Pw = self.patch_embed.patch_size
        mask = F.avg_pool2d(mask, kernel_size=(Ph, Pw), stride=(Ph, Pw))
        return mask

    def _downsample_mask(self, mask: torch.Tensor):
        H, W = mask.shape[-2], mask.shape[-1]
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            mask = F.pad(mask, (0, pad_w, 0, pad_h))
        return F.avg_pool2d(mask, kernel_size=2, stride=2)

    def _masked_avg_pool(self, x: torch.Tensor, mask: torch.Tensor):
        weights = mask.permute(0, 2, 3, 1)
        weighted = x * weights
        denom = weights.sum(dim=(1, 2)).clamp(min=1e-6)
        return weighted.sum(dim=(1, 2)) / denom

    def forward_features(self, x: torch.Tensor):
        x_map, _, pad_hw = self.patch_embed(x)
        mask = self._build_patch_mask(x, pad_hw)
        x_map = self.pos_drop(x_map)

        _, x_map = self.layer1(x_map)
        mask = self._downsample_mask(mask)
        _, x_map = self.layer2(x_map)
        mask = self._downsample_mask(mask)
        _, x_map = self.layer3(x_map)
        mask = self._downsample_mask(mask)
        x4, _ = self.layer4(x_map)

        x4 = self.norm(x4)
        return self._masked_avg_pool(x4, mask)

    def forward(self, x: torch.Tensor):
        return self.head(self.forward_features(x))

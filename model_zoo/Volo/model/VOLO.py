import math

from model.outlook import * 
from model.embeddings import * 
from model.droppath import * 
from model.pooling_volo_blocks import * 
from model.cls_attention import * 
from model.volo_stage import *


class VOLOClassifier(nn.Module):
    """
    VOLO para CIFAR-100 (y similares), con dos modos:
      - flat: OutlookerStage -> TransformerStack (sin downsample)
              pooling: mean | cls | cli (cls via class-attn final)
      - hierarchical: pirámide con downsample (map o token)
              pooling: SOLO mean (por ahora)

    Flujo base:
      x [B,3,H,W]
        -> PatchEmbeddingConv -> x_tok [B, N, C0]
        -> pos emb (opcional)
        -> backbone (flat o pyramid)
        -> pooling
        -> head
    """

    def __init__(
        self,
        num_classes: int = 100,
        img_size: int = 32,
        in_chans: int = 3,
        patch_size: int = 4,

        # mode
        hierarchical: bool = False,
        downsample_kind: str = "map",   # si hierarchical=True: "map" o "token"

        # dims / depths (flat)
        embed_dim: int = 192,
        outlooker_depth: int = 4,
        outlooker_heads: int = 6,
        transformer_depth: int = 6,
        transformer_heads: int = 6,

        # hierarchical configs (si hierarchical=True)
        dims: tuple[int, ...] = (192, 256, 384),
        outlooker_depths: tuple[int, ...] = (2, 2, 0),
        outlooker_heads_list: tuple[int, ...] = (6, 8, 12),
        transformer_depths: tuple[int, ...] = (0, 2, 2),
        transformer_heads_list: tuple[int, ...] = (6, 8, 12),

        # block hyperparams
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,

        # head / pooling
        pooling: str = "mean",          # flat: "mean"|"cls"|"cli" ; hierarchical: "mean"
        use_pos_embed: bool = True,

        # cls refinamiento (flat)
        cls_attn_depth: int = 2,        # # capas ClassAttentionBlock
        cli_init_alpha: float = 0.5,    # init alpha para pooling="cli"
        use_cls_pos: bool = True):

        super().__init__()

        self.hierarchical = hierarchical
        self.use_pos_embed = use_pos_embed

        if self.hierarchical:
            assert pooling == "mean", "Por ahora hierarchical solo soporta pooling='mean'."
        else:
            assert pooling in ["mean", "cls", "cli"], "pooling en flat debe ser 'mean', 'cls' o 'cli'."
        self.pooling = pooling

        # ---- Patch Embedding ----
        C0 = (dims[0] if hierarchical else embed_dim)

        self.patch_embed = PatchEmbeddingConv(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=C0,
            norm_layer=nn.LayerNorm,
            pad_if_needed=True,
            return_tokens=True,)

        Hp0 = math.ceil(img_size / patch_size)
        Wp0 = math.ceil(img_size / patch_size)

        self.pos_embed = PosEmbed2D(Hp0, Wp0, C0) if use_pos_embed else None
        self.pos_drop = nn.Dropout(dropout)

        # ---- Backbone ----
        if not hierarchical:
            total = outlooker_depth + transformer_depth
            dpr = torch.linspace(0, drop_path_rate, total).tolist() if total > 0 else []
            dpr_local = dpr[:outlooker_depth]
            dpr_glob = dpr[outlooker_depth:]

            self.local_stage = VOLOStage(
                dim=embed_dim,
                depth=outlooker_depth,
                num_heads=outlooker_heads,
                kernel_size=kernel_size,
                stride=1,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_dropout,
                proj_drop=dropout,
                drop_path=dpr_local if len(dpr_local) else 0.0,
                mlp_drop=dropout)

            self.global_blocks = nn.ModuleList([
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=transformer_heads,
                    mlp_ratio=mlp_ratio,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    drop_path=(dpr_glob[i] if len(dpr_glob) else 0.0),
                ) for i in range(transformer_depth)])

            # --- CLS  (solo si pooling usa cls/cli) ---
            self.use_cls = (pooling in ["cls", "cli"])
            if self.use_cls:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                trunc_normal_(self.cls_token, std=0.02)

                self.cls_pos = None
                if use_cls_pos:
                    self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
                    trunc_normal_(self.cls_pos, std=0.02)

                self.cls_attn_blocks = nn.ModuleList([
                    ClassAttentionBlock(
                        dim=embed_dim,
                        num_heads=transformer_heads,
                        mlp_ratio=mlp_ratio,
                        attn_dropout=attn_dropout,
                        dropout=dropout,) for _ in range(int(cls_attn_depth))])

                self.cli_pool = CLIPool(init_alpha=cli_init_alpha) if pooling == "cli" else None
            else:
                self.cls_token = None
                self.cls_pos = None
                self.cls_attn_blocks = None
                self.cli_pool = None


            self.norm = nn.LayerNorm(embed_dim)
            self.norm_feat = nn.LayerNorm(embed_dim)

            self.head = nn.Linear(embed_dim, num_classes)

        else:
            self.pyramid = VOLOPyramid(
                dims=dims,
                outlooker_depths=outlooker_depths,
                outlooker_heads=outlooker_heads_list,
                transformer_depths=transformer_depths,
                transformer_heads=transformer_heads_list,
                kernel_size=kernel_size,
                mlp_ratio=mlp_ratio,
                downsample_kind=downsample_kind,
                drop_path_rate=drop_path_rate,)


            self.norm = nn.LayerNorm(dims[-1])
            self.norm_feat = nn.LayerNorm(dims[-1])
            self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x_map, (Hp, Wp), x_tok, _pad = self.patch_embed(x)   # x_tok [B,N,C0]
        B, N, C0 = x_tok.shape

        # Pos emb sobre tokens del grid
        if self.use_pos_embed and (self.pos_embed is not None):
            x_tok = self.pos_embed(x_tok, (Hp, Wp))
        x_tok = self.pos_drop(x_tok)

        if not self.hierarchical:
            # ---- Flat backbone ----
            # Outlooker trabaja en map (sin CLS)
            x_map = x_tok.view(B, Hp, Wp, C0)
            x_map = self.local_stage(x_map)
            x_tok = x_map.view(B, Hp * Wp, C0)  # [B,N,C]

            # Transformer global (tokens sin CLS)
            for blk in self.global_blocks:
                x_tok = blk(x_tok)

            #  Pooling
            if self.pooling == "mean":
                # Normaliza tokens y promedia
                x_tok_n = self.norm(x_tok)           # [B,N,C]
                feat = x_tok_n.mean(dim=1)           # [B,C]
                feat = self.norm_feat(feat)          # [B,C]
                return self.head(feat)

            # CLS refinado con class-attn final (CaiT-style)
            cls = self.cls_token.expand(B, -1, -1)   # [B,1,C]
            if self.cls_pos is not None:
                cls = cls + self.cls_pos

            for cab in self.cls_attn_blocks:
                cls = cab(cls, x_tok)               # [B,1,C]

            cls_vec = cls.squeeze(1)                # [B,C]
            cls_vec = self.norm_feat(cls_vec)

            if self.pooling == "cls":
                feat = cls_vec
                return self.head(feat)

            # pooling == "cli": mezcla CLS con mean(tokens) normalizado
            tok_mean = self.norm(x_tok).mean(dim=1)  # [B,C]
            feat = self.cli_pool(cls_vec, tok_mean)
            feat = self.norm_feat(feat)
            return self.head(feat)

        else:
            # ---- Hierarchical backbone (solo mean) ----
            x_map = x_tok.view(B, Hp, Wp, C0)
            x_last, (Hf, Wf) = self.pyramid(x_map)      # x_last: [B, Nf, C_last]

            x_last = self.norm(x_last)
            feat = x_last.mean(dim=1)
            feat = self.norm_feat(feat)
            return self.head(feat)

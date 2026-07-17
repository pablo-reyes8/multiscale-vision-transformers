ARENA_DEFAULTS = {
    "dataset": "cifar100",
    "img_size": 32,
    "num_classes": 100,
    "epochs": 20,
    "batch_size": 128,
    "eval_batch_size": 256,
    "lr": 5e-4,
    "weight_decay": 0.05,
    "val_split": 0.1,
    "warmup_ratio": 0.05,
    "min_lr": 0.0,
    "label_smoothing": 0.1,
    "augment": "matched",
    "seed": 42,
}


AUGMENT_PRESETS = {
    "matched": "Dataset-aware crop/flip + RandAugment + dataset normalization.",
    "basic": "Dataset-aware crop/flip + dataset normalization.",
    "none": "No train-time augmentation, only dataset normalization.",
    "raw": "No augmentation, only ToTensor() and Normalize(mean=0.5, std=0.5) to map roughly to [-1, 1].",
}


MODEL_PRESETS = {
    "vit": {
        "family": "Original Vision Transformer",
        "builder": "vit",
        "description": "Flat patch sequence with CLS token and global self-attention.",
        "config": {
            "patch_size": 4,
            "embed_dim": 192,
            "depth": 6,
            "num_heads": 3,
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "patch_norm": False,
            "drop_rate": 0.0,
            "attn_dropout": 0.0,
            "proj_dropout": 0.0,
            "mlp_dropout": 0.1,
            "drop_path_rate": 0.1,
        },
    },
    "hierarchical_vit": {
        "family": "HierarchicalViT",
        "builder": "hierarchical_vit",
        "description": "PiT-like hierarchical transformer with token pooling between stages.",
        "config": {
            "patch_size": 4,
            "embed_dims": (192, 384, 576),
            "depths": (2, 2, 4),
            "num_heads": (3, 6, 9),
            "mlp_ratio": 4.0,
            "attn_dropout": 0.0,
            "dropout": 0.1,
        },
    },
    "swin": {
        "family": "Swin Transformer",
        "builder": "swin",
        "description": "Shifted-window transformer with hierarchical patch merging.",
        "config": {
            "patch_size": 4,
            "embed_dim": 96,
            "depths": (2, 2, 6, 2),
            "num_heads": (3, 6, 12, 24),
            "window_size": 7,
            "mlp_ratio": 4.0,
            "drop_rate": 0.0,
            "attn_dropout": 0.0,
            "proj_dropout": 0.0,
            "mlp_dropout": 0.0,
            "drop_path_rate": 0.1,
            "use_rel_pos_bias": True,
        },
    },
    "maxvit_tiny": {
        "family": "MaxViT",
        "builder": "maxvit",
        "description": "Tiny CIFAR-friendly MaxViT preset.",
        "variant": "tiny",
        "config": {
            "stem_type": "A",
            "drop_path_rate": 0.1,
        },
    },
    "maxvit_small": {
        "family": "MaxViT",
        "builder": "maxvit",
        "description": "Small CIFAR-friendly MaxViT preset.",
        "variant": "small",
        "config": {
            "stem_type": "A",
            "drop_path_rate": 0.15,
        },
    },
    "maxvit_base": {
        "family": "MaxViT",
        "builder": "maxvit",
        "description": "Base CIFAR-friendly MaxViT preset.",
        "variant": "base",
        "config": {
            "stem_type": "A",
            "drop_path_rate": 0.2,
        },
    },
    "volo": {
        "family": "VOLO",
        "builder": "volo",
        "description": "Flat VOLO with outlook attention followed by transformer refinement.",
        "config": {
            "patch_size": 4,
            "hierarchical": False,
            "downsample_kind": "map",
            "embed_dim": 320,
            "outlooker_depth": 5,
            "outlooker_heads": 10,
            "transformer_depth": 10,
            "transformer_heads": 10,
            "kernel_size": 3,
            "mlp_ratio": 4.0,
            "dropout": 0.12,
            "attn_dropout": 0.05,
            "drop_path_rate": 0.2,
            "pooling": "cls",
            "use_pos_embed": True,
            "cls_attn_depth": 2,
            "use_cls_pos": True,
        },
    },
    "volo_hierarchical": {
        "family": "VOLO",
        "builder": "volo",
        "description": "Hierarchical VOLO pyramid preset.",
        "config": {
            "patch_size": 4,
            "hierarchical": True,
            "downsample_kind": "map",
            "dims": (192, 256, 384),
            "outlooker_depths": (2, 2, 0),
            "outlooker_heads_list": (6, 8, 12),
            "transformer_depths": (0, 2, 2),
            "transformer_heads_list": (6, 8, 12),
            "kernel_size": 3,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.0,
            "drop_path_rate": 0.1,
            "pooling": "mean",
            "use_pos_embed": True,
            "cls_attn_depth": 0,
            "use_cls_pos": False,
        },
    },
}


GENERAL_TIPS = [
    "For direct comparison, keep --augment, --epochs, --batch-size, --lr, and --weight-decay fixed across models.",
    "Use the same random seed and val split when you want runs to be directly comparable.",
    "The arena uses one shared split and one shared preprocessing recipe for all requested models in the same run.",
    "If you choose augment=raw, the pipeline skips train-time augmentation and normalizes with mean=std=0.5 so inputs land near [-1, 1].",
    "Smaller models are better for quick sweeps; use max_train_batches and max_eval_batches for smoke comparisons before long runs.",
]


MODEL_TIPS = {
    "vit": [
        "embed_dim must be divisible by num_heads.",
        "Smaller patch sizes create more tokens and increase quadratic attention cost.",
    ],
    "hierarchical_vit": [
        "embed_dims, depths, and num_heads must have the same number of stages.",
        "Choose patch_size and number of stages so the token grid can be halved cleanly between stages.",
    ],
    "swin": [
        "embed_dim and each stage width must be divisible by the corresponding num_heads.",
        "window_size should be sensible relative to the patch grid; the implementation clips oversized windows, but matched settings are cleaner.",
        "depths and num_heads are expected to have four stages in this implementation.",
    ],
    "maxvit_tiny": [
        "Each stage dim should be divisible by its attention heads.",
        "window_size and grid_size need to fit the stage resolutions produced by the CIFAR pyramid.",
        "stem_out_ch must match dims[0] in MaxViT configs.",
    ],
    "maxvit_small": [
        "Each stage dim should be divisible by its attention heads.",
        "window_size and grid_size need to fit the stage resolutions produced by the CIFAR pyramid.",
        "stem_out_ch must match dims[0] in MaxViT configs.",
    ],
    "maxvit_base": [
        "Each stage dim should be divisible by its attention heads.",
        "window_size and grid_size need to fit the stage resolutions produced by the CIFAR pyramid.",
        "stem_out_ch must match dims[0] in MaxViT configs.",
    ],
    "volo": [
        "In flat VOLO, embed_dim must be divisible by both outlooker_heads and transformer_heads.",
        "pooling=cls or pooling=cli requires the flat (non-hierarchical) mode.",
    ],
    "volo_hierarchical": [
        "dims, outlooker_depths, outlooker_heads_list, transformer_depths, and transformer_heads_list must align stage by stage.",
        "In hierarchical VOLO, this implementation expects pooling='mean'.",
    ],
}


def available_model_names():
    return list(MODEL_PRESETS.keys())

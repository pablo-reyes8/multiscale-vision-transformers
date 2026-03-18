import torch

from model.attention_blocks import MultiHeadSelfAttention, TransformerEncoderBlock


def test_multi_head_self_attention_preserves_shape():
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(
        dim=64,
        num_heads=4,
        qkv_bias=True,
        attn_dropout=0.0,
        proj_dropout=0.0,
    )

    tokens = torch.randn(2, 17, 64)
    out = attn(tokens)

    assert out.shape == (2, 17, 64)
    assert torch.isfinite(out).all()


def test_transformer_encoder_block_preserves_shape():
    torch.manual_seed(0)
    block = TransformerEncoderBlock(
        dim=64,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        attn_dropout=0.0,
        proj_dropout=0.0,
        mlp_dropout=0.0,
        drop_path=0.0,
    )

    tokens = torch.randn(2, 17, 64)
    out = block(tokens)

    assert out.shape == (2, 17, 64)
    assert torch.isfinite(out).all()

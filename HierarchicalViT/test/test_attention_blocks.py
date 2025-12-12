import torch

from model.attention_blocks import (
    MultiHeadAttention,
    scaled_dot_product_attention,
)


def test_scaled_dot_product_attention_respects_boolean_mask():
    q = torch.tensor([[[1.0, 0.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    v = torch.tensor([[[1.0, 0.5], [5.0, 0.5]]])
    mask = torch.tensor([[[False, True]]])  # block second token

    output, attn = scaled_dot_product_attention(q, k, v, mask)

    expected = torch.tensor([[[1.0, 0.5]]])
    assert torch.allclose(output, expected, atol=1e-4)
    assert attn.shape == (1, 1, 2)
    assert attn[0, 0, 1] == 0  # masked entry should be zeroed after softmax


def test_multihead_attention_supports_mask_shapes():
    torch.manual_seed(0)
    attention = MultiHeadAttention(d_model=8, num_heads=2, dropout=0.0)
    x = torch.randn(1, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32)

    output = attention(x, x, mask=mask)

    assert output.shape == (1, 4, 8)

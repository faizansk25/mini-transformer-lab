import torch

from core.attention import MultiHeadAttention, RotaryPositionEmbedding


def test_attention_is_causal():
    torch.manual_seed(0)
    attention = MultiHeadAttention(16, 4, dropout=0.0, use_rope=False)
    original = torch.randn(1, 5, 16)
    changed = original.clone()
    changed[:, -1] += 100
    first = attention(original)[0]
    second = attention(changed)[0]
    torch.testing.assert_close(first[:, :-1], second[:, :-1])


def test_rope_preserves_vector_norm():
    rope = RotaryPositionEmbedding(8, max_seq_len=4)
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 3, 4, 8)
    rotated_q, rotated_k = rope(q, k)
    torch.testing.assert_close(rotated_q.norm(dim=-1), q.norm(dim=-1))
    torch.testing.assert_close(rotated_k.norm(dim=-1), k.norm(dim=-1))

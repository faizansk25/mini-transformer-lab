import pytest
import torch

from core.config import TransformerConfig
from core.sampling import generate, top_k_filter, top_p_filter
from core.transformer import MiniTransformer


def test_filters_validate_and_preserve_shape():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.isneginf(top_k_filter(logits, 1)[0, :2]).all()
    assert top_p_filter(logits, 0.8).shape == logits.shape
    with pytest.raises(ValueError):
        top_p_filter(logits, 0.0)


def test_greedy_generation_is_deterministic():
    model = MiniTransformer(
        TransformerConfig(
            vocab_size=256,
            d_model=16,
            num_heads=2,
            num_kv_heads=1,
            num_layers=1,
            d_ff=32,
            max_seq_len=8,
            dropout=0.0,
        )
    )
    prompt = torch.tensor([[1, 2, 3]])
    assert torch.equal(
        generate(model, prompt, max_new_tokens=2, temperature=0.0),
        generate(model, prompt, max_new_tokens=2, temperature=0.0),
    )

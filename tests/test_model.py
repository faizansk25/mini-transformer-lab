import torch

from core.config import TransformerConfig
from core.transformer import MiniTransformer


def make_model() -> MiniTransformer:
    return MiniTransformer(
        TransformerConfig(
            vocab_size=256,
            d_model=32,
            num_heads=4,
            num_kv_heads=2,
            num_layers=2,
            d_ff=64,
            max_seq_len=16,
            dropout=0.0,
        )
    )


def test_forward_backward_and_weight_tying():
    model = make_model()
    inputs = torch.randint(0, 256, (2, 8))
    output = model(inputs, targets=inputs)
    assert output.logits.shape == (2, 8, 256)
    assert output.loss is not None
    output.loss.backward()
    assert model.token_emb.weight.grad is not None
    assert model.lm_head.weight.data_ptr() == model.token_emb.weight.data_ptr()


def test_cached_logits_match_full_forward():
    torch.manual_seed(7)
    model = make_model().eval()
    inputs = torch.randint(0, 256, (1, 6))
    full = model(inputs).logits
    cache = None
    parts = []
    for position in range(inputs.size(1)):
        output = model(inputs[:, position : position + 1], past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        parts.append(output.logits)
    cached = torch.cat(parts, dim=1)
    torch.testing.assert_close(cached, full, atol=1e-5, rtol=1e-4)

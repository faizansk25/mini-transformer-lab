"""Sampling strategies: temperature, top-k, top-p."""

import torch
import torch.nn.functional as F


def sample_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Sample from logits with temperature scaling."""
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    return F.softmax(logits / temperature, dim=-1).multinomial(1).squeeze(-1)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only top-k values, set rest to -inf."""
    if k <= 0:
        return logits
    top_k = min(k, logits.size(-1))
    threshold = torch.topk(logits, top_k)[0][..., -1]
    logits = logits.masked_fill(logits < threshold, float("-inf"))
    return logits


def top_p_filter(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """Nucleus sampling: keep smallest set of tokens with cumulative probability >= p."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[mask] = float("-inf")
    logits = logits.scatter(-1, sorted_indices, sorted_logits)
    return logits


def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    use_cache: bool = True,
) -> torch.Tensor:
    """Autoregressive generation with sampling controls."""
    model.eval()
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated, use_cache=use_cache)
            next_logits = logits[:, -1, :]
            next_logits = top_k_filter(next_logits, top_k)
            next_logits = top_p_filter(next_logits, top_p)
            next_token = sample_with_temperature(next_logits, temperature)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

    return generated

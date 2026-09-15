"""Validated autoregressive sampling utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    if not 0.0 < p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if p == 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probabilities = F.softmax(sorted_logits, dim=-1)
    remove = torch.cumsum(probabilities, dim=-1) > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, filtered)


def sample_token(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature < 0.0:
        raise ValueError("temperature cannot be negative")
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    return torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).squeeze(-1)


def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    use_cache: bool = True,
) -> torch.Tensor:
    """Generate tokens with optional KV caching."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if input_ids.size(1) + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus generated tokens exceeds max_seq_len")
    model.eval()
    generated = input_ids
    cache = None
    current = input_ids
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            output = model(current, past_key_values=cache, use_cache=use_cache)
            logits = top_k_filter(output.logits[:, -1, :], top_k)
            logits = top_p_filter(logits, top_p)
            next_token = sample_token(logits, temperature)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            cache = output.past_key_values if use_cache else None
            current = next_token[:, None] if use_cache else generated
    return generated

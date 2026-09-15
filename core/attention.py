"""Causal attention primitives with RoPE and grouped-query attention."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

KVCache = Tuple[torch.Tensor, torch.Tensor]


class RotaryPositionEmbedding(nn.Module):
    """Apply rotary position embeddings to queries and keys."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10_000.0):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        positions = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        frequencies = torch.outer(positions, self.inv_freq)
        angles = torch.repeat_interleave(frequencies, 2, dim=-1)
        self.cos_cached = angles.cos()
        self.sin_cached = angles.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((-odd, even), dim=-1).flatten(-2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        end = position_offset + q.size(-2)
        if end > self.cos_cached.size(0):
            self._build_cache(end)
        cos = self.cos_cached[position_offset:end].to(device=q.device, dtype=q.dtype)
        sin = self.sin_cached[position_offset:end].to(device=q.device, dtype=q.dtype)
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


class MultiHeadAttention(nn.Module):
    """Causal multi-head attention with optional grouped-query attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        if self.num_kv_heads <= 0 or num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.head_dim = d_model // num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.w_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len) if use_rope else None

    def _project(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        q = self.w_q(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    @staticmethod
    def _causal_mask(query_len: int, key_len: int, past_len: int, device: torch.device) -> torch.Tensor:
        query_positions = torch.arange(past_len, past_len + query_len, device=device)
        key_positions = torch.arange(key_len, device=device)
        return key_positions.unsqueeze(0) > query_positions.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        if x.ndim != 3:
            raise ValueError("attention input must have shape [batch, sequence, d_model]")
        q, k, v = self._project(x)
        past_len = 0 if kv_cache is None else kv_cache[0].size(2)
        if self.rope is not None:
            q, k = self.rope(q, k, position_offset=past_len)
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            if cached_k.size(0) != x.size(0):
                raise ValueError("KV cache batch size does not match the input")
            k = torch.cat((cached_k, k), dim=2)
            v = torch.cat((cached_v, v), dim=2)
        if k.size(2) > self.max_seq_len:
            raise ValueError("sequence exceeds configured max_seq_len")

        new_cache = (k, v) if use_cache else None
        expanded_k = k.repeat_interleave(self.num_kv_groups, dim=1)
        expanded_v = v.repeat_interleave(self.num_kv_groups, dim=1)
        scores = torch.matmul(q, expanded_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self._causal_mask(q.size(2), expanded_k.size(2), past_len, x.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = torch.matmul(weights, expanded_v)
        output = output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return self.w_o(output), new_cache

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return deterministic causal attention weights for visualization."""
        q, k, _ = self._project(x)
        if self.rope is not None:
            q, k = self.rope(q, k)
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self._causal_mask(x.size(1), x.size(1), 0, x.device)
        minimum = torch.finfo(scores.dtype).min
        return F.softmax(scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), minimum), dim=-1)

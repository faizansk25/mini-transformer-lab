"""
Multi-Head Attention with Grouped-Query Attention (GQA) support.

Implements:
- Scaled dot-product attention
- Multi-head attention (MHA)
- Grouped-query attention (GQA) for efficient inference
- Causal masking for autoregressive generation
- KV cache for fast autoregressive decoding

Reference: "Attention Is All You Need" (Vaswani et al., 2017)
           "GQA: Training Generalized Multi-Query Transformer Models" (2023)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes position information directly into attention queries and keys
    using rotation matrices, enabling better length generalization.

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for max sequence length
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        if seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len)
            self.max_seq_len = seq_len

        cos = self.cos_cached[:seq_len].to(q.dtype)
        sin = self.sin_cached[:seq_len].to(q.dtype)

        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)
        return q_rot, k_rot

    def _apply_rotation(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotation to half the dimensions."""
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with optional Grouped-Query Attention (GQA).

    MHA: Every head has its own Q, K, V projections
    GQA: K and V are shared across groups of heads (fewer KV heads)
    MQA: K and V are shared across all heads (1 KV head)

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA). None = MHA
        dropout: Attention dropout rate
        max_seq_len: Maximum sequence length for causal mask
        use_rope: Whether to use Rotary Position Embeddings
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_rope: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        # Validate dimensions
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        assert num_heads % self.num_kv_heads == 0, f"num_heads must be divisible by num_kv_heads"

        self.num_kv_groups = num_heads // self.num_kv_heads

        # Projections
        self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        # RoPE
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)

        # Causal mask (register as buffer)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            kv_cache: Optional cached K, V from previous steps
            use_cache: Whether to return updated KV cache

        Returns:
            Output tensor [batch, seq_len, d_model]
            Optional updated KV cache
        """
        B, S, _ = x.shape

        # Project Q, K, V
        q = self.W_q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.use_rope:
            q, k = self.rope(q, k)

        # Update KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # Repeat KV heads for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Causal mask
        kv_len = k.shape[2]
        if kv_len > S:
            # With cache: only mask new tokens against future positions
            causal = self.causal_mask[S - 1 : S, :kv_len].unsqueeze(0).unsqueeze(0)
        else:
            causal = self.causal_mask[:S, :S].unsqueeze(0).unsqueeze(0)

        attn = attn.masked_fill(causal, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Compute output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.W_o(out)

        return out, new_cache

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for visualization (no dropout)."""
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q, k = self.rope(q, k)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        causal = self.causal_mask[:S, :S].unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(causal, float("-inf"))
        return F.softmax(attn, dim=-1)

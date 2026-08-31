"""Transformer block and full GPT model."""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import SwiGLU
from .norm import RMSNorm
from .config import TransformerConfig


class TransformerBlock(nn.Module):
    """Single Transformer block: Attention + FFN with pre-norm."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        Norm = RMSNorm if config.use_rmsnorm else nn.LayerNorm

        self.norm1 = Norm(config.d_model)
        self.attn = MultiHeadAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            use_rope=config.use_rope,
        )
        self.norm2 = Norm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff, dropout=config.dropout)

    def forward(self, x, kv_cache=None, use_cache=False):
        # Pre-norm attention
        h = x + self.attn(self.norm1(x), kv_cache=kv_cache, use_cache=use_cache)
        # Pre-norm FFN
        out = h + self.ffn(self.norm2(h))
        return out


class MiniTransformer(nn.Module):
    """
    GPT-style decoder-only Transformer language model.

    Architecture:
    - Token embedding (no learned position — uses RoPE)
    - N Transformer blocks with pre-norm
    - Final RMSNorm
    - Linear head (tied with embedding)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final norm + LM head
        Norm = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.norm = Norm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None, use_cache=False):
        x = self.token_emb(input_ids)
        cache = []
        for block in self.blocks:
            x, new_cache = block(x, use_cache=use_cache)
            if use_cache:
                cache.append(new_cache)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @classmethod
    def from_config_file(cls, path: str) -> "MiniTransformer

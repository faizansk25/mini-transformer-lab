"""Decoder-only Transformer model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import KVCache, MultiHeadAttention
from .config import TransformerConfig
from .feedforward import SwiGLU
from .norm import RMSNorm


@dataclass
class ModelOutput:
    """Output of a language-model forward pass."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple[KVCache, ...]] = None


class TransformerBlock(nn.Module):
    """Pre-normalized causal attention and feed-forward block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        norm = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.norm1 = norm(config.d_model)
        self.attn = MultiHeadAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            use_rope=config.use_rope,
        )
        self.norm2 = norm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        attended, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache, use_cache=use_cache)
        hidden = x + attended
        return hidden + self.ffn(self.norm2(hidden)), new_cache


class MiniTransformer(nn.Module):
    """Small GPT-style language model with tied input/output embeddings."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        norm = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.norm = norm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[KVCache]] = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.numel() and (input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size):
            raise ValueError("input_ids contain a token outside the configured vocabulary")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one cache per layer")
        hidden = self.token_emb(input_ids)
        next_cache = []
        for index, block in enumerate(self.blocks):
            layer_cache = None if past_key_values is None else past_key_values[index]
            hidden, cache = block(hidden, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache and cache is not None:
                next_cache.append(cache)
        logits = self.lm_head(self.norm(hidden))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return ModelOutput(logits, loss, tuple(next_cache) if use_cache else None)

    @classmethod
    def from_config_file(cls, path: str | Path) -> "MiniTransformer":
        return cls(TransformerConfig.from_json(path))

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config.to_dict(), "state_dict": self.state_dict()}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, map_location: str = "cpu") -> "MiniTransformer":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(TransformerConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model

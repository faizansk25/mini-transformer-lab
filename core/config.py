"""Validated model configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 256
    d_model: int = 128
    num_heads: int = 4
    num_kv_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 256
    dropout: float = 0.1
    use_rope: bool = True
    use_rmsnorm: bool = True

    def validate(self) -> None:
        sizes = (
            self.vocab_size,
            self.d_model,
            self.num_heads,
            self.num_kv_heads,
            self.num_layers,
            self.d_ff,
            self.max_seq_len,
        )
        if any(value <= 0 for value in sizes):
            raise ValueError("all size fields must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if (self.d_model // self.num_heads) % 2 and self.use_rope:
            raise ValueError("RoPE requires an even attention head dimension")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_json(cls, path: str | Path) -> "TransformerConfig":
        data: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        config = cls(**data)
        config.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def approximate_num_params(self) -> int:
        embeddings = self.vocab_size * self.d_model
        query_output = 2 * self.d_model * self.d_model
        key_value = 2 * self.d_model * self.num_kv_heads * (self.d_model // self.num_heads)
        attention = self.num_layers * (query_output + key_value)
        feed_forward = self.num_layers * (3 * self.d_model * self.d_ff)
        return embeddings + attention + feed_forward

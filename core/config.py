"""Model configuration."""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    d_model: int = 128
    num_heads: int = 4
    num_kv_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 256
    dropout: float = 0.1
    use_rope: bool = True
    use_rmsnorm: bool = True
    bias: bool = False

    @classmethod
    def from_json(cls, path: str) -> "TransformerConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def num_params(self) -> int:
        # Rough estimate
        embed = self.vocab_size * self.d_model
        attn = self.num_layers * (4 * self.d_model * self.d_model)
        ffn = self.num_layers * (2 * self.d_model * self.d_ff)
        return embed + attn + ffn

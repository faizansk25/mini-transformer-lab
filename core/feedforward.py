"""Feed-Forward Network with SwiGLU activation."""

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """
    SwiGLU activation: Swish(xW1) ⊙ xW2.

    Used in LLaMA, PaLM, and most modern LLMs.
    ~2x more efficient than standard ReLU FFN.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x)))

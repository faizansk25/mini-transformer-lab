"""RMSNorm — faster normalization than LayerNorm."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler and faster than LayerNorm — no mean centering, just RMS scaling.
    Used in LLaMA, Gemma, and most modern LLMs.

    Reference: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight

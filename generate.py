"""Generate text from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from core.sampling import generate
from core.tokenizer import BPETokenizer
from core.transformer import MiniTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--checkpoint", default="checkpoints/mini_transformer.pt")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    tokenizer_path = checkpoint.with_suffix(".tokenizer.json")
    if not checkpoint.exists() or not tokenizer_path.exists():
        raise SystemExit("checkpoint or tokenizer is missing; run train.py first")
    model = MiniTransformer.load_checkpoint(checkpoint)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokens = tokenizer.encode(args.prompt)
    output = generate(
        model,
        torch.tensor([tokens], dtype=torch.long),
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()

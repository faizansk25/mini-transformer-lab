"""Train Mini Transformer Lab on UTF-8 text."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from core.config import TransformerConfig
from core.tokenizer import BPETokenizer
from core.transformer import MiniTransformer

DEFAULT_TEXT = (
    "The transformer architecture learns patterns by predicting the next token. "
    "Small controlled experiments make model behavior easier to understand. "
) * 200


class TokenDataset(Dataset):
    """Create fixed-length next-token training windows."""

    def __init__(self, tokens: Sequence[int], seq_len: int, stride: int | None = None):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        stride = stride or seq_len
        self.samples = [
            torch.tensor(tokens[start : start + seq_len + 1], dtype=torch.long)
            for start in range(0, max(0, len(tokens) - seq_len), stride)
            if len(tokens[start : start + seq_len + 1]) == seq_len + 1
        ]
        if not self.samples:
            raise ValueError("training text is too short for the configured sequence length")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        return sample[:-1], sample[1:]


def train(
    config_path: str = "configs/tiny.json",
    text_path: str | None = None,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    output_path: str = "checkpoints/mini_transformer.pt",
    seed: int = 42,
) -> MiniTransformer:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    random.seed(seed)
    torch.manual_seed(seed)
    config = TransformerConfig.from_json(config_path)
    text = Path(text_path).read_text(encoding="utf-8") if text_path else DEFAULT_TEXT
    tokenizer = BPETokenizer.train([text], target_vocab_size=config.vocab_size)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            f"text produced {tokenizer.vocab_size} tokens, but config requires {config.vocab_size}; "
            "use more training text or a smaller vocab_size"
        )

    dataset = TokenDataset(tokenizer.encode(text), config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(epochs):
        model.train()
        losses = []
        for inputs, targets in dataloader:
            output = model(inputs.to(device), targets=targets.to(device))
            assert output.loss is not None
            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(output.loss.item())
        print(f"epoch={epoch + 1}/{epochs} loss={sum(losses) / len(losses):.4f}")

    model = model.cpu()
    model.save_checkpoint(output_path)
    tokenizer.save(Path(output_path).with_suffix(".tokenizer.json"))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.json")
    parser.add_argument("--text")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", default="checkpoints/mini_transformer.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.config, args.text, args.epochs, args.batch_size, args.learning_rate, args.output, args.seed)

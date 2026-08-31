"""Train a mini transformer language model."""

import json
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch
from core.config import TransformerConfig
from core.transformer import MiniTransformer


class TextDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 128):
        tokens = [ord(c) % 256 for c in text]
        self.sequences = []
        for i in range(0, len(tokens) - seq_len, seq_len):
            self.sequences.append(torch.tensor(tokens[i:i+seq_len+1], dtype=torch.long))
        if not self.sequences:
            self.sequences.append(torch.zeros(seq_len + 1, dtype=torch.long))
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]


def train(config_path: str = "configs/tiny.json", epochs: int = 5):
    config = TransformerConfig.from_json(config_path) if Path(config_path).exists() else TransformerConfig()
    model = MiniTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sample_text = "The transformer architecture has revolutionized natural language processing. " * 100
    dataset = TextDataset(sample_text, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            _, loss = model(x, targets=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1} Batch {batch_idx} Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1}/{epochs} Avg Loss: {total_loss/max(len(dataloader),1):.4f}")
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/mini_transformer.pt")
    print("Model saved to checkpoints/mini_transformer.pt")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.json")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    train(args.config, args.epochs)

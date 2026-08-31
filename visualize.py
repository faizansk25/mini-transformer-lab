"""Visualize attention patterns from a trained transformer."""

import torch
import argparse
from core.config import TransformerConfig
from core.transformer import MiniTransformer


def visualize_attention(model, input_text: str):
    tokens = [ord(c) % 256 for c in input_text]
    x = torch.tensor([tokens], dtype=torch.long)
    emb = model.token_emb(x)
    for i, block in enumerate(model.blocks):
        attn_weights = block.attn.get_attention_weights(emb)
        print(f"\n--- Layer {i+1} Attention Shape: {attn_weights.shape} ---")
        head0 = attn_weights[0, 0].detach().numpy()
        for row in head0[:min(10, len(tokens))]:
            line = "".join(["#" if v > 0.05 else "." for v in row[:min(20, len(tokens))]])
            print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="Hello world")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mini_transformer.pt")
    args = parser.parse_args()
    config = TransformerConfig()
    model = MiniTransformer(config)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    except FileNotFoundError:
        print("No checkpoint found. Using random weights.")
    model.eval()
    visualize_attention(model, args.input)


if __name__ == "__main__":
    main()

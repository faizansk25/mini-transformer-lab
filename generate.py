"""Generate text using a trained transformer model."""

import argparse
import torch
from core.config import TransformerConfig
from core.transformer import MiniTransformer
from core.sampling import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The ")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mini_transformer.pt")
    args = parser.parse_args()
    config = TransformerConfig()
    model = MiniTransformer(config)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {args.checkpoint}")
    except FileNotFoundError:
        print("No checkpoint found. Using untrained model.")
    tokens = [ord(c) % 256 for c in args.prompt]
    input_ids = torch.tensor([tokens], dtype=torch.long)
    output = generate(model, input_ids, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)
    generated = "".join([chr(t % 256) for t in output[0].tolist()])
    print(f"\n--- Generated ---\n{generated}")


if __name__ == "__main__":
    main()

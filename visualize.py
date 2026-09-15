"""Save attention maps from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from core.tokenizer import BPETokenizer
from core.transformer import MiniTransformer


def save_attention_maps(model: MiniTransformer, token_ids: torch.Tensor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hidden = model.token_emb(token_ids)
    for index, block in enumerate(model.blocks):
        weights = block.attn.get_attention_weights(block.norm1(hidden))[0].detach().cpu()
        hidden, _ = block(hidden)
        figure, axes = plt.subplots(1, min(4, weights.size(0)), squeeze=False)
        for head, axis in enumerate(axes[0]):
            axis.imshow(weights[head].numpy(), cmap="viridis", aspect="auto")
            axis.set_title(f"head {head}")
        figure.tight_layout()
        figure.savefig(output_dir / f"layer_{index + 1}.png", dpi=160)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Transformers are")
    parser.add_argument("--checkpoint", default="checkpoints/mini_transformer.pt")
    parser.add_argument("--output-dir", default="attention_maps")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    model = MiniTransformer.load_checkpoint(checkpoint)
    tokenizer = BPETokenizer.load(checkpoint.with_suffix(".tokenizer.json"))
    model.eval()
    ids = torch.tensor([tokenizer.encode(args.input)], dtype=torch.long)
    save_attention_maps(model, ids, Path(args.output_dir))


if __name__ == "__main__":
    main()

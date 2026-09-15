# Mini Transformer Lab

An educational decoder-only Transformer implemented with PyTorch. The repository contains a byte-level BPE tokenizer, causal multi-head and grouped-query attention, RoPE, RMSNorm, SwiGLU, KV-cached generation, attention visualization, validated configurations, tests, and CI.

This is a learning implementation, not a production language model or a claim of novel research.

## Verified capabilities

- Deterministic byte-level BPE training, encoding, decoding, and serialization
- Multi-head attention, grouped-query attention, causal masking, and RoPE
- Pre-normalized decoder blocks with RMSNorm and SwiGLU
- Tied token embeddings and language-model head
- KV cache whose token-by-token logits are tested against a full forward pass
- Temperature, top-k, and top-p sampling
- Tiny, small, and medium configurations
- Automated compile, lint, and test checks in GitHub Actions

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pytest -q
```

## Train and generate

```bash
python train.py --config configs/tiny.json --epochs 5
python generate.py --checkpoint checkpoints/mini_transformer.pt --prompt "The future of AI"
python visualize.py --checkpoint checkpoints/mini_transformer.pt --input "Transformers are"
```

Pass `--text path/to/corpus.txt` to train on your own UTF-8 corpus. The training command stores a checkpoint and matching tokenizer. The bundled text is only a smoke-test corpus.

## Repository structure

```text
core/
  attention.py      # causal MHA/GQA, RoPE, KV cache
  config.py         # validated model configuration
  feedforward.py    # SwiGLU
  model.py          # stable model exports
  norm.py           # RMSNorm
  sampling.py       # generation filters and loop
  tokenizer.py      # educational byte-level BPE
  transformer.py    # decoder blocks and language model
configs/            # tiny, small, and medium configurations
tests/              # attention, caching, sampling, and tokenizer tests
train.py
generate.py
visualize.py
```

## Limits

- The tokenizer is intentionally simple and not optimized for large corpora.
- The training script is single-process and intended for experiments.
- No pretrained weights or benchmark claims are included.
- Configuration names describe relative scale inside this repository, not competitive LLM sizes.

## References

- Vaswani et al., *Attention Is All You Need* (2017)
- Su et al., *RoFormer* (2021)
- Ainslie et al., *GQA* (2023)
- Zhang and Sennrich, *Root Mean Square Layer Normalization* (2019)

MIT License.

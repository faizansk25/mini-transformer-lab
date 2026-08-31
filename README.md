<div align="center">

# 🧠 Mini Transformer Lab

### *Build a Transformer from scratch — understand every matrix multiplication*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**From-scratch Transformer implementation with training, attention visualization, and sampling controls**

</div>

---

## 📖 What is this?

Most people use transformers as a black box. This project **opens the box**.

Built based on *Attention Is All You Need* (2017), *GPT-2* (2019), and modern techniques like RoPE, RMSNorm, and GQA:

1. 🔤 **Custom tokenizer** — BPE tokenization from scratch
2. 🧮 **Full Transformer** — Every layer implemented, no shortcuts
3. 📊 **Attention visualization** — See what the model learns
4. 🎛️ **Sampling controls** — Temperature, top-k, top-p
5. 📈 **Training dashboard** — Real-time loss and metrics
6. ✅ **Tests** — Verified against PyTorch reference operations

> **Why this matters:** Understanding transformers at the implementation level is the single most important skill for LLM engineering. This project proves you don't just *use* transformers — you *understand* them.

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **BPE Tokenizer** | Custom byte-pair encoding implementation |
| **Multi-Head Attention** | Q/K/V projections, scaled dot-product, causal masking |
| **RoPE** | Rotary Position Embeddings for better length generalization |
| **RMSNorm** | Faster normalization than LayerNorm |
| **GQA** | Grouped-Query Attention for efficient inference |
| **KV Cache** | Autoregressive generation with cached key/values |
| **Temperature/Top-k/Top-p** | Controllable text generation |
| **Attention Maps** | Visualize what each head attends to |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│            Input Tokens                  │
└──────────────┬──────────────────────────┘
               │
       ┌───────▼───────┐
       │  BPE Tokenizer │  Convert text → tokens
       └───────┬───────┘
               │
       ┌───────▼───────┐
       │  Token Embedding│  Lookup table
       │  + RoPE        │  Position encoding
       └───────┬───────┘
               │
   ┌───────────▼───────────┐
   │   Transformer Block   │  × N layers
   │  ┌─────────────────┐  │
   │  │  RMSNorm         │  │
   │  │  Multi-Head Attn │  │  Self-attention
   │  │  + Residual      │  │
   │  ├─────────────────┤  │
   │  │  RMSNorm         │  │
   │  │  Feed-Forward    │  │  SwiGLU activation
   │  │  + Residual      │  │
   │  └─────────────────┘  │
   └───────────┬───────────┘
               │
       ┌───────▼───────┐
       │  Final Norm    │
       │  + LM Head     │  Predict next token
       └───────┬───────┘
               │
       ┌───────▼───────┐
       │  Output Logits │  Temperature/top-k/top-p sampling
       └───────────────┘
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/faizansk25/mini-transformer-lab.git
cd mini-transformer-lab
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train a tiny model
python train.py --config configs/tiny.json

# Generate text
python generate.py --checkpoint checkpoints/tiny_best.pt --prompt "The future of AI" --temperature 0.8

# Run attention visualization
python visualize.py --checkpoint checkpoints/tiny_best.pt --input "Transformers are"
```

---

## 🧪 Test Suite

```bash
pytest tests/ -v
```

Tests verify:
- Attention output matches PyTorch `nn.MultiheadAttention`
- Gradient flow through entire model
- RoPE correctness at various positions
- KV cache produces identical output to non-cached forward
- Tokenizer encode/decode roundtrip

---

## 📁 Project Structure

```
mini-transformer-lab/
├── core/
│   ├── tokenizer.py          # BPE tokenizer
│   ├── attention.py          # Multi-head + GQA attention
│   ├── feedforward.py        # SwiGLU FFN
│   ├── transformer.py        # Full transformer block
│   ├── model.py              # GPT-style language model
│   ├── embeddings.py         # Token + RoPE embeddings
│   ├── norm.py               # RMSNorm
│   ├── sampling.py           # Temperature, top-k, top-p
│   └── config.py             # Model configurations
├── tests/
│   ├── test_attention.py     # vs PyTorch reference
│   ├── test_model.py         # Forward/backward pass
│   ├── test_tokenizer.py     # Encode/decode roundtrip
│   └── test_generation.py    # Sampling correctness
├── configs/
│   ├── tiny.json             # 1M params
│   ├── small.json            # 10M params
│   └── medium.json           # 50M params
├── train.py
├── generate.py
├── visualize.py
├── requirements.txt
└── README.md
```

---

## 📊 Model Configurations

| Config | Params | Layers | Heads | d_model | Context |
|--------|--------|--------|-------|---------|---------|
| tiny | ~1M | 4 | 4 | 128 | 256 |
| small | ~10M | 6 | 8 | 256 | 512 |
| medium | ~50M | 8 | 12 | 512 | 1024 |

---

## 📚 References

- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2, 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)](https://arxiv.org/abs/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer Models (2023)](https://arxiv.org/abs/2305.13245)

---

## 👨‍💻 Author

**Faizan Muktar Shaikh**
- 🔗 [LinkedIn](https://linkedin.com/in/faizansk25) | [GitHub](https://github.com/faizansk25)

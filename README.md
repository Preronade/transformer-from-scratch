# Transformer From Scratch

A GPT-style, decoder-only Transformer implemented entirely from first principles in PyTorch — no `nn.MultiheadAttention`, no shortcuts. Every core component (scaled dot-product attention, multi-head attention, causal masking, positional embeddings, residual connections, layer norm) is written by hand to demonstrate real understanding of the architecture.

Trained as a character-level language model on the Tiny Shakespeare dataset.

## Result

Training loss dropped from 4.28 (random) to 1.79 over 2000 steps on CPU. The model learns real structural patterns — capitalization, punctuation, contractions, character-name formatting — despite its small size (0.82M parameters) and short training time.

Sample output after training:

I have plove is thes ame, wep word
Ro? Edway'd Thy sperver's of Gevoste's and,
Romed belt bothinter haverse,

## Architecture

- `SelfAttentionHead` — scaled dot-product attention with a causal (triangular) mask, computed manually
- `MultiHeadAttention` — 4 attention heads run in parallel, concatenated, and projected
- `FeedForward` — position-wise MLP with GELU activation
- `TransformerBlock` — pre-norm residual block combining attention and feedforward
- `GPT` — token + positional embeddings, stacked Transformer blocks, final projection to vocabulary

## Tech stack

Python 3.12 · PyTorch (CPU) · pytest

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
mkdir -p data
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python src/train.py
```

## Tests

```bash
pip install pytest
pytest -q
```

## Author

Prerona De — B.Tech CSE (AI-driven DevOps), Jain University, Bangalore
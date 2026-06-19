# GPT-from-Scratch

A from-scratch implementation of a small decoder-only transformer (~10M parameters,
nanoGPT-scale), trained on a small text corpus, with sampling and KV-cached inference.

## Goal

Build operational fluency in transformer internals by implementing every component by
hand in PyTorch. No reference implementations copied. No high-level abstractions
(`nn.MultiheadAttention`, `F.scaled_dot_product_attention`, etc.).

## Constraint

I write 100% of the code. Claude Code is a tutor only — explains, questions,  
reviews, hints. Never produces code that goes into the codebase.

## Stages

1. Data loading + character-level tokenization
2. Token + learned positional embeddings
3. Scaled dot-product attention, single head, with causal mask
4. Multi-head attention with output projection
5. LayerNorm
6. Pointwise FFN with GELU
7. Transformer block (pre-norm)
8. Full GPT model (stack + final norm + unembedding)
9. Training loop: AdamW, cross-entropy, LR warmup + cosine decay
10. Train on TinyShakespeare; verify loss decreases, samples improve
11. Sampling: greedy → temperature → top-k → top-p
12. KV cache; benchmark speedup vs naive generation
13. Replace absolute pos embeddings with RoPE; retrain
14. (Optional) Replace GELU MLP with SwiGLU
15. (Optional) GQA

## Success criteria

- Trains end-to-end, loss decreases, samples qualitatively resemble training data
- I can derive the math behind every choice (√d, softmax, pre-norm, RoPE rotation, etc.)
on paper, unprompted
- Each core module has a unit test (shape + non-NaN forward, gradient sanity check)
- The full repo is mine — no copy-paste from nanoGPT, minGPT, HuggingFace, etc.

## Stack

PyTorch only. No transformers / accelerate / lightning. Local execution on Apple Silicon.

## Setup

Hardware: M4 Pro, 48 GB unified memory. Backend: MPS (Metal).

```python

device = torch.device("mps" if [torch.backends.mps.is](http://torch.backends.mps.is)_available() else "cpu")

```

Environment:

- `PYTORCH_ENABLE_MPS_FALLBACK=1` — lets unsupported ops fall back to CPU instead of crashing
- Stay in fp32 for now (skip mixed precision)
- Skip `torch.compile` (hit-or-miss on MPS, not needed at this scale)
- Test tolerances: use `atol=1e-4` (MPS has ~1e-5 numerical drift vs CPU)


# Stage 2 — Embedding Concepts

## 2026-05-11

## What I worked on
Conceptual probe for stage 2 — token + learned positional embeddings — before writing any code.

## Key concepts
- Stage 2 input: `(B, T)` int64 from `get_batch`. Output: `(B, T, d_model)` float32. The `d_model` dimension is the new thing — every downstream tensor inside the model carries it.
- Self-attention is permutation-**equivariant**: `f(π(x)) = π(f(x))`. Positional embeddings break this symmetry by attaching a position-specific signal to each token before attention sees it.
- Token embedding parameter count: `V · d_model`. Learned positional embedding: `T_max · d_model`. For `V=65, T_max=256, d_model=384` → 24,960 vs 98,304 — positional has ~4× more (since `T_max > V`).
- Sinusoidal positional encoding (Vaswani 2017) and RoPE (Su 2021) have **0 parameters** — deterministic functions of position, not learned tables.
- Composition is elementwise sum: `x = tok_emb(ids) + pos_emb(positions)`. Concatenation is *not* used because (a) it roughly quadruples downstream parameters, (b) it provides zero expressiveness gain once a linear layer sees it.

## What I got wrong
- Called self-attention "permutation-symmetric". The strict term is **equivariant** (`f(π(x)) = π(f(x))`); invariance is the strictly weaker statement `f(π(x)) = f(x)` and is *not* what attention has. The equation I wrote was right; the name to attach to it was off.
- Was vague on sum vs concat. Only caught "wider model = longer to train". Missed the bigger reasons: (1) the param count roughly *quadruples* per attention block downstream (`(2d, 6d)` QKV instead of `(d, 3d)`), not just doubles at the input; (2) sum and concatenation-followed-by-linear are mathematically the same operation class — `[t; p] @ W = t @ W_t + p @ W_p` for the appropriate block decomposition. Concat costs more for nothing.
- Suggested `d_model ~ 12-16` for V=65, citing Johnson-Lindenstrauss. JL is right about *token-identity capacity* alone but `d_model` is the **residual-stream dimension**, not an embedding dimension. The residual stream needs to carry everything the model has computed across all layers — syntactic role, n-gram context, attention outputs from prior layers — all superimposed in the same `d_model`-dim vector at each position. Choking it at 16 dims would crush the model's bandwidth for compositional structure, not just token storage.
- Gave an example for permutation equivariance ("dog bites man") but didn't articulate the consequence mechanically: an equivariant function's output at position `i` depends only on the *multiset* of input tokens, not their order, so any permutation of a sentence gives a representation that's a permutation of the original. The model literally can't tell word orders apart.

## Why this works
- **Sum ≡ concat-then-linear, expressiveness-wise.** For `[tok; pos] ∈ R^{2d}` and `W ∈ R^{2d × d}` partitioned into top/bottom halves `W = [W_t; W_p]`, the matmul `[tok; pos] @ W = tok @ W_t + pos @ W_p`. That's the sum of two linearly-transformed copies. By summing at the input we've collapsed concat+projection into one step, with the embedding matrices playing the role of `W_t, W_p` — already learned via backprop. Concatenation downstream is pure overhead.
- **`d_model` as residual-stream bandwidth.** Olah/Elhage's "Mathematical Framework for Transformer Circuits" framing: the residual stream is a shared communication channel. Every block reads from it, writes back to it, the surviving signal is the linear superposition. `d_model` is the channel bandwidth — not "the space tokens are embedded into" but "the space the entire forward computation lives in". Pick big enough that the *non-embedding* uses don't get crushed.
- **Permutation equivariance of attention.** Output at position `i` is `softmax(Q_i K^T / √d) @ V` — a weighted sum of value rows where the weights depend on the *similarity* between Q_i and each K_j, not on the *position* j. Permute Q, K, V's rows the same way, the output rows permute the same way. No mechanism singles out position `j` over position `k`; that's why positional embeddings are mandatory.

## Open questions
- Practical `d_model` choice for the project. Going with 128 unless a concrete reason to bump to 384 (nanoGPT-shakespeare default) or down to 64.
- `n_heads` selection — head_dim = `d_model / n_heads` should be 32–128. For `d_model=128`, `n_heads=4` gives `head_dim=32` (tight but workable). Deferred to stage 4.
- `T_max` (context window length, a.k.a. `block_size`): need to commit before instantiating the positional embedding table. nanoGPT-shakespeare uses 256. Decide before code.

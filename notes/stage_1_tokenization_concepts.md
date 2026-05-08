# Stage 1 — Tokenization Concepts

## 2026-05-09

## What I worked on
Conceptual probe of the stage-1 data pipeline — file on disk → input to the first `nn.Linear`. No code yet, just nailing the mental model.

## Key concepts
- Pipeline shape: `corpus.txt → str of length L → encoded corpus int64 (L,) → batched int64 (B, T) → nn.Embedding(V, d_model) → float32 (B, T, d_model) → first nn.Linear`.
- **Vocab** and **encoded corpus** are different objects. Vocab is metadata of size `V`; encoded corpus is data of length `L`. They interact only at the embedding lookup.
- Vocab is `sorted(set(corpus))` for char-level. The sort pins determinism — Python uses hash randomization for strings, so set iteration order can permute across separate Python invocations. Without sorting, embeddings get associated with the wrong characters across runs.
- `stoi: Dict[str, int]` for encoding; `itos: Dict[int, str]` for decoding.
- Sizes split into data-determined (`L`, `V`) and hyperparameter-chosen (`T`, `B`, `d_model`, etc.). The model never sees `L`; it sees `(B, T)` windows sampled from the length-`L` corpus.
- Embedding lookup ≡ `e_i^T · W` where `e_i ∈ R^V` is a one-hot. Indexed lookup is the sparse-matmul shortcut — same math, different implementation.

## What I got wrong
- Used `N` for both corpus length and vocab size in the chain. Wrong — these are different scales (~10⁶ vs ~10²). Conflating them obscured every shape downstream. Now `L` and `V`.
- Said the encoded corpus is `List[Int]` of length `V`. Wrong — it's of length `L`. I confused "vocab indices" (the integers `0..V-1`, which label vocab entries) with "the encoded corpus" (a sequence of `L` such integers, one per character). These are entirely different objects.
- Skipped the embedding step in my chain — went directly from `(B, T)` integer batch to `nn.Linear`. Wrong: `nn.Linear` operates on floats, so the embedding lookup is mandatory between.
- Said "B = number of batches". Wrong — `B` is **batch size** (sequences per batch). Number of batches per pass ≈ `L / (B·T)`.
- Swapped the type signatures of `stoi`/`itos` against their descriptions. Encoder: `Dict[str, int]`. Decoder: `Dict[int, str]`. Will matter when I write code.
- Imagined the "embedded corpus" as a precomputed `(L, d_model)` object. Wrong: embedding is **inside** the model's forward pass — the embedding matrix `(V, d_model)` is a learned parameter, updated by gradient descent. Pre-materializing would eat GBs and freeze embeddings at their random init values.
- Q3 first answer: "integers don't have rounding errors." Marginal benefit, missed the actual point. The real reason: scalar encoding `c → W·c + b` constrains all V chars to a 1-D affine line in `R^{d_model}` — total `d_model + 1` parameters for the entire vocabulary. Integer-indexed embeddings give each char its own free vector — `V · d_model` parameters. The geometric collapse is the killer issue, not the float rounding.
- Did not know what "one-hot" means. Now: `e_i ∈ R^V` with `1` at position `i`, `0` elsewhere. Algebraic encoding of "categorical, no ordering imposed".
- Wrote `W_E · e_i` for the matmul-equivalent of the embedding. Shape doesn't conform (`(V × d_model) · (V)` is invalid). Correct: `e_i^T · W_E` — row vector times matrix, result is the i-th row of `W_E`. Physics column-vector instinct misfires in ML, where row-vector convention dominates (`x @ W`, not `W x`).
- Said `32 bits = 3 bytes`. No — 4 (8 bits per byte). Got the right ~3 GB answer for the one-hot tensor only because count was 25% over-estimated and bytes 25% under-estimated; errors cancelled. Won't always cancel.
- Said "128 KB is much larger" than 3 GB. Confused kilo with giga. Late at night, tired.

## Why this works
- **Indexed lookup** is mathematically the matmul `e_i^T W` written sparsity-aware: `O(d_model · B · T)` FLOPs instead of `O(V · d_model · B · T)`, and ~25,000× less memory at GPT-2 vocab. The math is identical; the implementation respects the 1-of-V sparsity of the input.
- **Integer-indexed embeddings** are the right inductive bias for categorical inputs. Characters have no continuous structure to interpolate between (`'b'` is not the average of `'a'` and `'c'`), so any continuous encoding throws representational capacity away. The categorical-as-points-in-learned-space view is the foundation for everything that comes later (positions, attention scores, etc.).
- **Char-level vs BPE structural contrast**: char-level vocab is *closed* under the data (`|set(corpus)|`, 0 hyperparameter choices); BPE is *open* and user-extended (`|base| + n_merges`, where each merge step adds exactly one new vocab entry from the most frequent adjacent pair). Different design philosophies, different tradeoffs in vocab size vs sequence length vs semantic granularity.

## Open questions
- How is `d_model` chosen in practice for a small char-level model on tiny-shakespeare? Any rule of thumb tying it to `V` or `T`?
- Which transformer ops still lack MPS kernels — i.e., when does `PYTORCH_ENABLE_MPS_FALLBACK=1` actually fire?
- The `(x, y)` "pair" structure for next-token prediction — `y` is `x` shifted by 1. To be made concrete in the dataloader step.
- BPE: at what vocab size does the compression-vs-coverage tradeoff saturate? Empirical, presumably, but is there a heuristic?

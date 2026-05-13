# Stage 3: Scaled dot-product attention (single head, causal mask)

## Summary

This stage built the algorithmic heart of the transformer: single-head scaled dot-product self-attention with a causal mask. The `Attention` module consumes a `(B, T, d_model)` float32 residual-stream tensor (the output of stage 2's embeddings), projects it through a fused QKV linear layer into three sub-tensors of widths `d_k, d_k, d_v`, computes pairwise content-similarity scores `Q · K^T / √d_k`, applies a lower-triangular causal mask via `-∞` before softmax, normalizes into a per-query attention distribution over keys, takes the resulting weighted sum of value vectors, and projects back to `d_model` through an output linear layer. Attention is the *only* position-mixing operation in the transformer — every other layer (embeddings, MLPs, normalization) operates per-token — so this is where information actually flows across the sequence. The causal mask is what makes the shift-by-1 supervision setup from stage 1 honest: future tokens cannot influence the prediction at position `i`, so the `B·T` parallel training examples are genuinely independent next-token-prediction problems rather than copy tasks.

## The math

**`Q · K^T / √d_k` — derivation of the √d scaling.** Assume `Q_i, K_i` are iid with mean 0 and variance 1. Then for `Q · K = Σ_{i=1}^{d_k} Q_i K_i`:

- `E[Q_i K_i] = E[Q_i] E[K_i] = 0` (independence).
- `Var(Q_i K_i) = E[(Q_i K_i)²] − E[Q_i K_i]² = E[Q_i²] E[K_i²] − 0 = 1 · 1 = 1`.
- `Var(Σ_i Q_i K_i) = Σ_i Var(Q_i K_i) = d_k` (independence of distinct `(Q_i, K_i)` pairs).
- Therefore `std(Q · K) = √d_k`. Dividing by `√d_k` normalizes pre-softmax scores to unit variance.

**Why this matters — softmax saturation.** Softmax Jacobian: `∂ softmax_j / ∂ x_i = softmax_j · (δ_{ij} − softmax_i)`. When one input `x_k` dominates (saturated softmax: `softmax_k ≈ 1`, others `≈ 0`):

- `i = j = k`: `softmax_k · (1 − softmax_k) ≈ 1 · 0 = 0`.
- All other cases also evaluate to 0 (each has at least one factor of `softmax_{not-k} ≈ 0` or `(1 − softmax_k) ≈ 0`).
- The entire softmax Jacobian is effectively a zero matrix at saturation.

Chain rule consequence: gradients through softmax → 0 → gradients to `W_Q, W_K` → 0 → no parameter updates → training stalls. Without `√d_k`, `Q · K` has std `√d_k` (growing with `d`), so softmax saturates from initialization onward and training never escapes the no-gradient regime. The `√d_k` scaling is an **initialization-stability** argument, not a fundamental constraint — after training, the model can learn arbitrary attention sharpness via the QK projections.

**Permutation equivariance of attention (motivates the causal mask).** Pure attention is permutation-equivariant: `f_attn(π(x)) = π(f_attn(x))`. Output at position `i` depends only on the *multiset* of input tokens, not their order. Positional embeddings (stage 2) break this for the input; the causal mask further restricts attention to attend only to past positions, making sequence order load-bearing throughout.

## The code

- `src/attention.py:9` — `Attention(nn.Module)` class.
  - `__init__(T_max, d_k, d_v, d_model)`: registers `(T_max, T_max)` bool tril mask as a buffer; stores `d_k, d_v` as attributes; creates `qkv_proj = nn.Linear(d_model, 2*d_k + d_v)` (fused) and `out_proj = nn.Linear(d_v, d_model)`.
  - `forward(x)`: fused QKV → split via `qkv.split([d_k, d_k, d_v], dim=-1)` → `Q @ K.transpose(-1, -2) / √d_k` → mask via `masked_fill(self.mask[:T, :T].logical_not(), -inf)` → `F.softmax(dim=-1)` → `attn @ V` → `out_proj`.
- `tests/test_attention.py` — 5 pytest cases:
  - `test_attn_shape_and_type` — input `(B, T, d_model)` → output `(B, T, d_model)`, float32.
  - `test_qkv_parameter_count` — verifies `qkv_proj.weight.numel()` and `qkv_proj.bias.numel()` with non-equal `d_k, d_v` to break coincidence.
  - `test_attn_total_parameter_count` — sum of all parameters matches expected (catches missing `super().__init__()`).
  - `test_attn_causality` — modifying input at position `T-1` doesn't change earlier output positions. The structural test for the mask.
  - `test_attn_mask_is_non_parameter` — mask is in `state_dict()` but not in `parameters()`.

## Design choices and why

- **Fused QKV projection** (single `nn.Linear(d_model, 2·d_k + d_v)`) instead of three separate `nn.Linear`s. Same parameter count, better memory locality, one kernel call instead of three. Implementation cost: `qkv.split([d_k, d_k, d_v], dim=-1)` instead of three forward calls. Justified by the practice in tensor manipulation that pays off in stage 4's multi-head reshape.
- **`split` over `chunk`** for the QKV decomposition. `chunk(3, dim=-1)` would work for `d_k = d_v` but break for `d_k ≠ d_v`; `split([d_k, d_k, d_v], dim=-1)` is explicit and generalizes. Worth the slightly more verbose call for forward-compatibility.
- **General `d_k, d_v`** in the constructor rather than hard-coding `d_k = d_v = d_model`. Pedagogical clarity (they play distinct roles in the math) and forward-compatibility with stage 4 where they become `d_model / n_heads`. The smoke test uses `d_k = d_v = d_model = 128`; the tests parametrize with `d_k = d_model // 4, d_v = d_model // 2` to break the equality coincidence.
- **`register_buffer` for the causal mask.** The mask is a non-parameter constant (`bool` tensor, shape `(T_max, T_max)`, identical across batches and forward passes). Registering it as a buffer means it moves with `.to(device)`, gets saved/loaded in `state_dict()`, and is excluded from `parameters()`. Alternatives (`self.mask = torch.tril(...)`, `self.mask = nn.Parameter(...)`) are both wrong: the first doesn't move with the module, the second would be updated by the optimizer.
- **Mask polarity stored as "allowed" (True for past, False for future).** The tril-of-ones convention has `mask[i, j] == True` for `j ≤ i` (past + present). Since `masked_fill` fills where the mask is True, the forward inverts via `.logical_not()` to flip the polarity to "blocked" before passing to `masked_fill`. The alternative — storing the mask as "blocked" directly — is also valid; the current form is one extra op per forward but makes the *semantic* polarity (the tril *allows* positions) explicit.
- **Mask applied before softmax**, not after. Before: `-∞ + score = -∞`, then softmax exponentiates to 0, and the softmax denominator only sums over allowed positions; one operation. After: would require softmax-then-zero-then-renormalize; two operations. Mathematically equivalent; before-with-`-inf` is cheaper and more standard.
- **`transpose(-1, -2)`** with negative dim indices rather than `transpose(1, 2)`. Works whether the tensor is 3-D `(B, T, d_k)` or 4-D `(B, n_heads, T, head_dim)` from stage 4 onward. Forward-compatibility with the multi-head reshape.
- **Scaling by `√d_k` (not `√d_model`).** For single-head with `d_k = d_model` they coincide; for multi-head `d_k = d_model / n_heads`, so the scaling has to use `d_k` to keep per-head softmax inputs at unit variance. Writing `√d_k` now means stage 4's reshape doesn't require a scaling change.
- **`T_max` as architectural cap, `T` as runtime length.** Same static/dynamic distinction as stage 2's positional embedding. The mask buffer is sized to `T_max` once; forward slices `self.mask[:T, :T]` to the runtime size (slice is a view, no copy).
- **Single `Attention` class** rather than separate QK and OV submodules. The QK/OV decomposition is a conceptual interpretability lens, not a code-organization choice. Splitting forces ugly inter-module coupling (QK and OV both need the same input; QK's attention pattern feeds OV); single class with the decomposition implicit in the math is cleaner.

## Errors and corrections

- **Called self-attention "permutation-symmetric"** when the term is **equivariant** (`f(π(x)) = π(f(x))`). Invariance is the strictly weaker `f(π(x)) = f(x)`. Equation right, name wrong.
- **Claimed Q=K=V would collapse to per-token nonlinearity.** Wrong — Q=K=V still mixes positions, but constrains attention to symmetric patterns with V coupled to K/Q. Three separate projections give independent control over "what am I looking for / what do I look like / what do I contribute".
- **`√d` derivation initially given as one-liner without showing the work.** Pushed to derive: `Var(Q · K) = Σ Var(Q_i K_i) = d` via mean-0, var-1 iid assumption, using independence of distinct `(Q_i, K_i)` pairs to suppress covariance terms.
- **Softmax-saturation gradient claim wrong on first attempt.** Initially wrote `∂_i softmax_j = δ_{ik} δ_{jk}` (1 at `i = j = k`, 0 elsewhere). Actually all four cases of the Jacobian evaluate to 0 at saturation: `softmax_k(1 - softmax_k) ≈ 1·0 = 0` for the diagonal, and the other entries vanish because each product contains a factor near 0.
- **Tried `qkv.chunk([d_k, d_k, d_v], dim=-1)`** — confused signatures. `chunk` takes an `int`; `split` takes a list. The two look similar but error in different ways.
- **Mask polarity bug**: initially wrote `scores.masked_fill(self.mask[:T, :T], -inf)`. The tril-of-ones bool mask is True at past positions, so this filled `-inf` at past (allowed) positions, leaving future positions untouched — exactly inverted. Fixed with `.logical_not()`.
- **`super.__init__()` without parens.** `super` is the class; `super()` is the bound instance. Same trap as `dict` vs `dict()`. Raises at runtime.
- **`from _pytest.monkeypatch import V`** — Cursor autocomplete artifact (second occurrence in the project). The bogus import silently allowed `out = attn @ V` (capital) to typecheck instead of NameError-ing at the typo. Two coupled bugs masked by one bad import.
- **`out_proj_bias_num = d_v`** in the parameter-count test. Test passed because `d_v = d_model = 128` numerically in the test config — a "passes for the wrong reason" bug. Fixed by trace from the `nn.Linear` declaration: `out_proj = nn.Linear(d_v, d_model)` → bias has `d_model` entries. Tests subsequently parametrized with `d_k ≠ d_v ≠ d_model` to break the coincidence.
- **`x_modified = x` in the causality test** is not a copy — modifies `x` in place. Doesn't break this specific test (because `out` was already computed before the mutation), but is a code smell and would break analogous patterns. Fix: `x_modified = x.clone()`.
- **Arithmetic slip in toy shape prediction**: wrote `2*2 + 2 = 8` instead of 6. Caught because downstream q, k, v shapes summed to 6 inconsistently with the qkv shape claimed at 8.
- **"Projectors" vs "projections"** terminology in the line-by-line walkthrough. Minor stylistic but consistent ML usage is "projections".

## Self-quiz

1. Derive `Var(Q · K) = d_k` from first principles assuming `Q_i, K_i` iid mean-0 var-1. Where does independence enter? What part of the derivation breaks if `Q_i` and `K_i` are correlated within position `i`?
2. State the softmax Jacobian, `∂ softmax_j / ∂ x_i = ?` (in terms of `softmax_j` and `softmax_i`). Evaluate it at a fully-saturated point (`softmax_k ≈ 1`, others `≈ 0`) for all four cases (i = j = k, i = k j ≠ k, i ≠ k j = k, i ≠ k j ≠ k). Conclude what happens to gradients through softmax in that regime. Why does this make training fail without `√d_k` scaling?
3. Self-attention is permutation-**equivariant**, not invariant. State both definitions as equations. Why is equivariance strictly weaker as a constraint than invariance? Why does language modeling require positional information to break the equivariance?
4. The mask is `(T_max, T_max)` bool stored via `register_buffer`. What are *three* PyTorch behaviors that `register_buffer` provides over a plain `self.mask = torch.tril(...)` assignment, and what bug would each catch if it were missing?
5. The polarity of the tril-of-ones mask is "True at past positions, False at future". `masked_fill(M, -inf)` fills where `M` is True. Why does the forward apply `.logical_not()` to the mask before `masked_fill`? What would happen to training if the `.logical_not()` were omitted (i.e., the mask polarity were inverted)?
6. The QK circuit `(W_Q · W_K^T)` and the OV circuit `(W_V · W_O)` are the two factored "circuits" in mechanistic-interpretability analysis of an attention head. Explain what each circuit controls and why this decomposition is a useful analytical lens despite being implementation-coupled (i.e., the circuits aren't separable modules in code).
7. Why does the scaling factor in the standard implementation use `√d_k` rather than `√d_model`? When are the two equal? When do they differ? What changes about the variance of `Q · K` between single-head and multi-head?
8. The fused QKV projection is `nn.Linear(d_model, 2*d_k + d_v)`. The output is split via `qkv.split([d_k, d_k, d_v], dim=-1)`. Why `split` (not `chunk`)? Under what `d_k`, `d_v` configurations would `chunk(3, dim=-1)` produce wrong shapes silently? Compute the parameter count for both `qkv_proj` and `out_proj` in terms of `d_model, d_k, d_v` (including biases).

## What this enables

- **Stage 4 (multi-head attention)** is structurally a tensor-reshape on top of stage 3's attention. The fused QKV projection's output dim becomes `2 * d_model + d_model = 3 * d_model` for `d_k = d_v = d_model / n_heads`; reshape splits across `n_heads` and `head_dim`. The transpose(-1, -2) pattern, the `√d_k` scaling, the causal mask, and the OV circuit all carry over unchanged. Multi-head is "stage 3 with an extra leading dim of size `n_heads` in the attention dimensions".
- **Stage 5 (LayerNorm)** wraps the residual stream (input to attention and to MLP); doesn't change attention internally. The `(B, T, d_model)` interface lets LayerNorm slot in cleanly.
- **Stage 7 (transformer block)** stacks attention + LayerNorm + MLP into one residual block. Stage 3's `Attention` becomes the position-mixing half of every block; the MLP becomes the per-token feature-processing half.
- **Stage 12 (KV cache)** exploits the same QK/OV decomposition but caches K and V across generation steps. Stage 3's separation of Q (recomputed each step) from K, V (cacheable) is what makes that optimization possible.
- **Stage 13 (RoPE)** replaces the additive positional embedding (stage 2) with a position-dependent rotation applied to Q and K *inside* this attention module — not as an additive input. Restructures the attention forward to inject rotation between Q/K computation and the `QK^T` step.

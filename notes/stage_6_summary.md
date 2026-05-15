# Stage 6: Pointwise FFN / MLP with GELU

## Summary

This stage built `MLP`, the pointwise feed-forward network that pairs with attention in every transformer block. Mechanically tiny — two `nn.Linear` layers with `F.gelu` between them — but architecturally essential: attention does position-mixing only (gathers information across tokens via learned patterns), and MLP does per-token feature processing only (applies the same nonlinear function independently to each token's representation). Neither does the other's job. The transformer block alternates these two phases, and the MLP is structurally where the model's expressive nonlinearity lives — attention's only intrinsic nonlinearity is softmax, which is constrained to producing probability distributions and can't compute arbitrary feature transformations. Standard architecture: `Linear(d_model, d_ff)` expands the residual stream by 4× (Vaswani convention), `F.gelu` applies the smooth nonlinearity, `Linear(d_ff, d_model)` projects back down. MLP carries ~2/3 of each transformer block's parameters (the 4× expansion makes its parameter count `8·d_model²` per block vs attention's `4·d_model²`).

## The math

**Forward pass:**
```
up    = x @ W₁ᵀ + b₁     (Linear(d_model, d_ff); b₁ optional)
gated = F.gelu(up)       (smooth elementwise nonlinearity)
out   = gated @ W₂ᵀ + b₂ (Linear(d_ff, d_model); b₂ optional)
```

**GELU** (Gaussian Error Linear Unit):
- Exact: `GELU(x) = x · Φ(x) = 0.5 · x · (1 + erf(x / √2))` where Φ is the standard normal CDF.
- Tanh approximation (BERT/GPT-2 era): `GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`.
- Limits: `GELU(x) → x` as `x → +∞`; `GELU(x) → 0` as `x → -∞`; smooth "bump" of slight negative output around `x ≈ -0.5`.
- Derivative is smooth everywhere — no kink at zero like ReLU.

**Parameter counts:**
- `bias=True`: `d_model · d_ff (W₁) + d_ff (b₁) + d_ff · d_model (W₂) + d_model (b₂) = 2 · d_model · d_ff + d_ff + d_model`.
- `bias=False`: `2 · d_model · d_ff` (just the weight matrices, no biases).
- With `d_model=128, d_ff=512`: 131,712 (bias=True) vs 131,072 (bias=False). The bias contribution is ~0.5% of total — tiny but compounding at scale.

**Attention vs MLP parameter ratio per block** (no biases, standard ratios):
- Attention (Q, K, V, O each `d × d`): `4 · d_model²`.
- MLP (`d_ff = 4 · d_model`): `2 · d_model · 4·d_model = 8 · d_model²`.
- Ratio: MLP : attention = 2 : 1. MLP holds ~2/3 of each block's parameters.

## The code

- `src/mlp.py:8` — `MLP(nn.Module)` class.
  - `__init__(d_model, d_ff=None, bias=False)`: stores `d_model`, derives `d_ff = 4·d_model` when `None`, creates `up_proj = nn.Linear(d_model, d_ff, bias)` and `down_proj = nn.Linear(d_ff, d_model, bias)`.
  - `forward(x: (B, T, d_model)) → (B, T, d_model)`: `self.down_proj(F.gelu(self.up_proj(x)))`. Three calls, no intermediate variables.
- `tests/test_mlp.py` — 3 pytest test functions (12 parametrized cases):
  - `test_mlp_shape_and_type`: shape/dtype preservation.
  - `test_mlp_parameter_count` (parametrized over `d_ff ∈ {128, 256, 384, 512, 768}` × `bias ∈ {False, True}` = 10 cases): verifies the parameter-count formula for each.
  - `test_mlp_nonlinearity_applied`: homogeneity violation check — `mlp(a·x) ≠ a·mlp(x)` with `bias=False` and `a=3`. Tests that GELU is actually in the chain.

## Design choices and why

- **`d_ff = 4 · d_model` default** (Vaswani convention). Empirically battle-tested as a sweet spot between expressivity and parameter cost. Variations exist: T5 tried 4-8×; SwiGLU uses `8/3 · d` to keep param-matched with vanilla 4d GELU FFN. 4× is the standard default; the `d_ff` argument is exposed so it can be overridden when the project eventually grows.
- **`bias=False` default.** Modern convention (LLaMA, Mistral, Qwen, PaLM). Reasons stacked: (a) pre-norm LayerNorm has its own β that comes before the linear, partially subsuming the bias's role; (b) empirical work (LLaMA, PaLM) found no measurable improvement from biases; (c) slight parameter savings compound at scale; (d) marginal training-stability gains in long runs. Vaswani 2017 / GPT-2 / BERT had biases; the shift happened with PaLM (2022) and LLaMA (2023). The `bias` argument is still exposed for ablation.
- **GELU over ReLU.** GELU is smooth everywhere (no kink at zero), no dead-unit problem (slight negative output for slightly-negative inputs, with nonzero gradient), and has a probabilistic gating interpretation (`x · P(X ≤ x)` for `X ~ N(0,1)`). Empirically lower loss in transformers; this has been standard since BERT (2018). The anti-shortcut rule in CLAUDE.md allows `F.gelu` because the GELU formula has subtle implementation choices (exact-erf vs tanh approximation, fp16 numerical stability) and the library function is correct; reimplementing wouldn't teach much beyond what's covered in this summary.
- **`up_proj` / `down_proj` naming.** LLaMA convention. Semantic — `up_proj` expands to `d_ff`, `down_proj` projects back. Alternatives (`fc1/fc2`, `c_fc/c_proj`) are less informative. Pays off when reading mech-interp papers where these names are standard.
- **No intermediate variables in `forward`.** `self.down_proj(F.gelu(self.up_proj(x)))` is one expression. Could be split into named intermediates (`up = self.up_proj(x); gated = F.gelu(up); return self.down_proj(gated)`) but adds no debugging value for a 3-op chain.
- **`module(x)`, not `module @ x`.** `nn.Linear` is a callable module via `__call__` → `forward`, not a matrix. The `@` operator (matmul) isn't defined on `nn.Module`. Same convention used since stage 2.

## Errors and corrections

- **`self.up_proj @ x` and `self.down_proj @ ...`** in first draft. Used matmul (`@`) on `nn.Linear` modules. `nn.Linear` is a callable module, not a matrix — `@` operator isn't defined for `nn.Module`. Correct: `self.up_proj(x)`. Same module-call convention as all prior stages; the bug was a momentary lapse, not a misunderstanding.
- **Unnecessary `def sigma(y): return F.gelu(y)`** local wrapper in first draft. Pure rename of `F.gelu` with no added functionality. Inlined `F.gelu(...)` directly.
- **`4*d_model if not d_ff else d_ff`** initially. `not d_ff` treats `d_ff=0` the same as `d_ff=None` (both falsy). The semantically correct check for "is this the sentinel" is `d_ff if d_ff is not None else 4*d_model`. Edge case probably never matters in practice, but `is None` is the precise check.
- **Type annotation typos** (`"B T d_mode"` and `"B T d_mdoel"`) in first draft. Fixed input first, missed the output for one revision.
- **Test parametrize syntax confusion**: tried `@pytest.mark.parametrize("d_ff, bias", [list_of_d_ffs, list_of_biases])` — wrong. Pytest expected a list of `(d_ff, bias)` tuples; instead got two lists. Fix: stacked parametrize decorators (Cartesian-product semantics) — `@parametrize("d_ff", [...])` stacked with `@parametrize("bias", [...])` generates all 10 combinations automatically.
- **`RuntimeError: Boolean value of Tensor with more than one value is ambiguous`** in the nonlinearity test. Initially used `mlp(a*x) != a*mlp(x)` which returns an *element-wise comparison tensor*, not a Python `bool`. Fix: `not torch.allclose(mlp(a*x), a*mlp(x))` — `allclose` returns a Python `bool` directly, which `assert not` can evaluate.
- **GELU "switching off the nonlinearity" reasoning** (during Q4 bias discussion). Said large positive `b_2` could "switch off the nonlinearity". Actually: `GELU(large +x) ≈ x` (passthrough, not "off"); `GELU(large -x) ≈ 0` (off). The bias-removal argument is better grounded in LayerNorm-β redundancy and empirical-no-benefit at scale, not in nonlinearity-switching mechanics.

## Self-quiz

1. State the GELU formula in both exact and tanh-approximation forms. Identify three concrete reasons GELU is preferred over ReLU in transformers (smoothness, dead-unit avoidance, stochastic-gating interpretation), with one sentence each.
2. The `d_ff = 4 · d_model` ratio is conventional. Explain (a) why expansion is needed at all (i.e., why not `d_ff = d_model`), (b) why specifically 4× and not some other multiplier, (c) what SwiGLU does differently to achieve matched-compute lower loss.
3. Modern open models (LLaMA, Mistral, Qwen) drop biases throughout. Give three substantive reasons. Specifically address: how does pre-norm LayerNorm's β interact with a downstream `nn.Linear` bias, and why does that interaction make the bias partially redundant?
4. Compute the per-block parameter count ratio of MLP to attention for a standard transformer with `d_ff = 4 · d_model`. State the ratio numerically. Then explain why this ratio drives the total-parameter accounting at scale (i.e., why MLP dominates parameter count in large LLMs).
5. The MLP is "pointwise" — applied per-token independently. Why is this the right design? What would change if the MLP could mix across token positions? (Hint: think about what role attention plays vs MLP, and what would happen to the model's computational pattern if both layers did the same thing.)
6. Write down the homogeneity-violation test in math: for what classes of functions does `f(a·x) = a·f(x)` hold for all scalar `a`? For what classes does it fail? Why does the MLP fail this property at `bias=False, a=3` (the test you wrote), but trivially fail at any `a ≠ 1` if biases are turned on?
7. The Geva et al. 2021 "Transformer Feed-Forward Layers Are Key-Value Memories" framing assigns roles to `W₁` (keys) and `W₂` (values), with GELU as the gate. Describe the mechanism: how does a specific input pattern in the residual stream activate a specific "key", and how does that key activation determine which "value" gets written back?
8. The transformer block has two phases per layer: attention (position-mixing) and MLP (per-token feature processing). State two computations that *cannot* be performed by attention alone (without MLP), and explain why MLP specifically enables them. Reference the structural reason that softmax-based attention is limited.

## What this enables

- **Stage 7 (transformer block)** is the immediate next stage — composes `MultiHeadAttention` (stage 4) + `MLP` (stage 6) + `LayerNormalization` (stage 5) into the canonical pre-norm residual block: `x = x + attn(LN_1(x)); x = x + mlp(LN_2(x))`. Stage 6's MLP is one half of every block; stage 7 is where the pieces finally compose.
- **Stage 8 (full GPT)** stacks `n_layers` transformer blocks plus token/positional embeddings (stage 2) plus a final LayerNorm + unembedding head. MLP is reused per-block.
- **Stage 9 (training loop)** sees MLP receive gradients through cross-entropy on the next-token prediction objective. MLP's `up_proj` and `down_proj` weights are where most parameter updates happen during training (since they hold most of the model's parameters).
- **Stage 14 (optional, SwiGLU)** replaces this stage's `Linear → GELU → Linear` chain with `(SiLU(W₁x) ⊙ (W₃x)) → W₂`. Forward reference saved at `notes/stage_14_swiglu_reference.md` from the discussion during this stage. Same role in the transformer block; different internal structure (multiplicative gating + three projections instead of two). Param count rebalanced via `d_ff ≈ 8/3 · d_model` to match GELU FFN's compute.
- **Mechanistic interpretability later**: MLP-as-key-value-memory framing (Geva et al. 2021) is the standard analytical lens for MLPs in trained transformers. The `up_proj` rows are the keys; the `down_proj` columns are the values; GELU is the per-key gate. Stage 6's clean structure is exactly what enables this kind of analysis once you have a trained model.

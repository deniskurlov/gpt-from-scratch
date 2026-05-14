# Stage 5: LayerNormalization

## Summary

This stage built `LayerNormalization`, the per-token normalization layer that wraps around attention and MLP in every transformer block. Mechanically the smallest module so far — 4 substantive lines of math (mean, variance, normalize, affine) — but conceptually one of the most important: LayerNorm is what makes deep residual networks of arbitrary depth trainable without learning-rate warmup or specialized initialization tricks. It normalizes each token's representation across the `d_model` feature dimension to zero mean and unit variance, then applies a learnable per-coordinate affine transformation (γ, β) to restore expressiveness lost to strict normalization. The normalization happens per-token (per `(B, T)` slice) and never mixes across batch or sequence axes — preserving autoregressive causality and avoiding the train/eval skew that plagues BatchNorm in transformer settings. Stage 7 will compose this with attention and MLP via pre-norm residual blocks; this stage produced the module itself.

## The math

**LayerNorm formula** (per token, across `d_model`):

```
μ      = (1/d_model) · Σ_i x_i                       (mean across last dim)
σ²     = (1/d_model) · Σ_i (x_i - μ)²                (biased variance)
x̂_i   = (x_i - μ) / √(σ² + ε)                       (normalize, ε-regularized)
y_i    = γ_i · x̂_i + β_i                            (per-coordinate affine)
```

where:
- **μ, σ²**: computed at runtime from the input.
- **γ, β**: learned per-feature parameters, both shape `(d_model,)`. γ initialized to ones, β to zeros — so at init the affine is the identity on `x̂`.
- **ε**: hyperparameter, typically `1e-5` in fp32; placed *inside* the sqrt for numerical stability.

**Why ε inside the sqrt.** The form `√(σ² + ε)` has bounded derivatives everywhere (since `σ² + ε ≥ ε > 0`), so gradients through LayerNorm flow well even when `σ² = 0`. The alternative `√σ² + ε` (equivalently `|σ| + ε`) has a non-smooth derivative at `σ = 0` — the kink in the absolute value — which can destabilize backprop when token features are nearly constant. Forward-pass-wise both give finite values; the backward pass is where placement matters.

**Why `√head_dim` is not what's used here.** Stage 3/4 used `√d_k` inside attention to normalize the score matrix; stage 5's `√(σ² + ε)` is a completely different normalization (the per-token feature std), unrelated to attention scaling. Same `sqrt` symbol, different roles.

## The code

- `src/normalization.py:7` — `LayerNormalization(nn.Module)` class.
  - `__init__(d_model, eps=1e-5, bias=True)`: stores `d_model, eps, bias`; declares `γ = nn.Parameter(torch.ones(d_model))`; conditionally declares `β = nn.Parameter(torch.zeros(d_model))` when `bias=True`.
  - `forward(x: (B, T, d_model)) → (B, T, d_model)`: computes mean and biased variance along last dim with `keepdim=True`; normalizes via `(x - μ) / √(σ² + ε)`; applies `γ * x_norm + β` (elementwise multiply + add with broadcasting); returns same-shape output.
- `tests/test_normalization.py` — 3 pytest cases:
  - `test_layer_norm_shape_and_type`: shape preserved, dtype `torch.float32`.
  - `test_layer_norm_mean_and_var`: at init (γ=1, β=0), output mean ≈ 0 (`atol=1e-6`) and var ≈ 1 (`atol=1e-5`) along last dim. The substantive "normalization actually normalizes" test.
  - `test_layer_norm_param_count`: `2*d_model` parameters when `bias=True`, `d_model` when `bias=False`. Catches missing `super().__init__()`.

## Design choices and why

- **Normalize across the last dim (`d_model`), not batch or sequence.** Per-token normalization is what makes LayerNorm correct for autoregressive sequence models. Normalizing across batch (BatchNorm) couples examples, breaks at batch-size-1 inference, and causes train/eval skew. Normalizing across `T` would mix future tokens into past tokens' normalization statistics, structurally breaking causality even with a correct attention mask.
- **Biased variance (`unbiased=False`).** Matches `nn.LayerNorm`'s convention. For `d_model = 128`, the factor `n/(n-1) ≈ 1.008` between biased and unbiased — under 1% numerical difference. Not statistically meaningful for ML normalization (we're not estimating population variance; we're scaling by the sample's own spread).
- **γ as `nn.Parameter`, not `nn.Linear`.** γ is a learnable vector of shape `(d_model,)` applied elementwise per coordinate. `nn.Linear` would be the wrong abstraction — it's a full matrix-based linear projection, structurally more general than the diagonal scaling γ provides. The right primitive for "learnable tensor that operates via broadcasting, not matmul" is `nn.Parameter`. Same applies to β.
- **`bias=True` toggle (γ + β) vs `bias=False` (γ only).** The toggle lets the class also represent the γ-only structure that RMSNorm-style variants use (combined with dropping the mean centering, but here we keep mean). For stage 5, `bias=True` (full LayerNorm) is the default; `bias=False` is available as a future ablation knob without needing a separate class.
- **`elementwise_affine=False` toggle NOT exposed.** `nn.LayerNorm` has a separate flag for "no γ AND no β simultaneously", which `bias=False` doesn't cover (it keeps γ). For this project's scope, the `bias` toggle is sufficient; the rare "pure normalization with no learned affine at all" use case can wait if it ever comes up.
- **ε placement inside `sqrt(var + eps)`** — covered in math section. Smooth gradient at `σ² = 0`; the alternative form has a derivative kink at the boundary.
- **`keepdim=True` on the reductions.** Both `mean` and `var` use `keepdim=True` so the resulting `(B, T, 1)` tensor broadcasts cleanly against the input `(B, T, d_model)` in the subtraction and division. Without `keepdim`, broadcasting would right-align `(B, T)` against `(B, T, d_model)` — mismatched dims, would error or do something unintended.
- **Pre-norm placement decision deferred to stage 7.** The `LayerNormalization` class itself is agnostic to whether it's applied before attention/MLP (pre-norm) or after (post-norm). Stage 7's transformer block will compose it in pre-norm form (`x = x + attn(LN(x))`) — the modern convention since GPT-2.

## Errors and corrections

- **Initial walkthrough wording: "σ is the variance"** — σ² is variance, σ is standard deviation. Formula was right; symbol naming was off. Caught and corrected mid-discussion.
- **"Per-feature" terminology pushback** — Denis (correctly) pushed on what "feature" means in the LayerNorm context. Resolved: "feature" = "coordinate of the d_model-dim residual stream vector" (sense 1, coordinate-level). Distinct from mechanistic-interp "feature" = "direction in activation space encoding an interpretable concept" (sense 2). γ, β are per-coordinate (sense 1), which is unambiguous if you say "per coordinate" rather than "per feature".
- **γ as `nn.Linear` confusion.** First implementation attempt used `nn.Linear(d_model)` for both γ and β. Multiple problems: (a) `nn.Linear` signature is `(in, out)` not `(dim,)`; (b) γ and β are vectors applied elementwise, not matrices applied via matmul; (c) the right primitive for learnable vectors applied via broadcasting is `nn.Parameter`. Same conceptual error driving multiple syntactic bugs.
- **`mean = sum(x) / len(x)`** — Python's `sum` iterates over the first dim and `len` returns the first-dim size, so this computed mean over batch, not over `d_model`. Correct: `torch.mean(x, dim=-1, keepdim=True)`.
- **`sqrt(var ** 2 + eps)`** — `var` is already σ². Squaring it gave σ⁴, completely wrong inside the sqrt. Correct: `sqrt(var + eps)`. Also `sqrt` from `math` doesn't operate on tensors — needed `torch.sqrt()` (or `tensor ** 0.5`).
- **`gamma @ x_norm`** — used matmul `@` instead of elementwise multiply `*`. Conceptual error: γ is a per-coordinate scaling factor (diagonal scaling), not a general linear transformation. The right operator is `*` with broadcasting.
- **Missing `super().__init__()`** in first draft (class didn't inherit from `nn.Module` either). Two coupled issues: the class header was `class LayerNormalization():` with no parent. Fixed both at the same time.
- **`from torch import layer_norm`** in the imports — Cursor autocomplete artifact, **third occurrence** in the project (after `_pytest.monkeypatch.V` in stage 3 and `torch._dynamo.utils.V` in stage 2). This one was especially treacherous: the name `layer_norm` *looks* relevant to the file's purpose, so an inattentive review would let it through. Removed once flagged.
- **Test bug: `[-1].view(d_model)` on the wrong-shape tensor.** Trying to test "mean of output is 0", Denis indexed `[-1]` on a `(B, T, 1)` tensor (getting `(T, 1) = (4, 1)` = 4 elements) and tried to reshape to `(d_model,) = (128,)`. Element-count mismatch would have errored at runtime. Replaced with `torch.allclose(torch.mean(x_out, dim=-1), torch.zeros_like(...), atol=1e-5)` — no shape manipulation needed; compare reductions to zeros of matching shape directly.
- **`keepdim` confusion** — needed to be explicitly explained that `keepdim=True` keeps the reduced dim as size 1 (vs `keepdim=False` which removes the dim), and that the keep-vs-remove choice affects subsequent broadcasting. Useful in the forward pass (`mean` needs to broadcast back against `x`); irrelevant in the test (`mean` is the final answer, no further broadcasting).
- **Boredom mid-stage.** Stage 5 was the simplest module so far, and the full protocol felt like ceremony. Calibrated to "3 tests, brief walkthrough, shape-only toy prediction" — the criteria still hold but the intensity was scaled down. Worth noting because stages 7+ will be substantial again and the full protocol returns there.

## Self-quiz

1. Write the LayerNorm formula given input `x ∈ R^d`. Identify which symbols are runtime-computed vs learned vs hyperparameters. State the shape of γ and β in terms of `d_model`.
2. Trace the broadcasting in `(x - mean) / torch.sqrt(var + self.eps)` where `x: (B, T, d_model)`, `mean, var: (B, T, 1)`. What rule makes the elementwise operations work, and what would go wrong without `keepdim=True` on the reductions?
3. Derive why ε is placed *inside* the sqrt (`√(σ² + ε)`) rather than *outside* (`√σ² + ε`). State the difference in terms of derivative behavior at `σ = 0`, and explain why that matters for backprop.
4. Why does LayerNorm normalize across the last dim (`d_model`) rather than across batch or across `T`? Give the specific structural reasons each alternative would fail (BatchNorm: 2 reasons; across-T: 1 reason).
5. γ is initialized to ones and β to zeros. What does this mean for the *function* LayerNorm computes at the start of training? Why are these specific initialization values (not, say, γ = 0, β = something else) — what optimization principle do they correspond to?
6. The `bias=False` toggle keeps γ and drops β. RMSNorm also drops the mean centering entirely. Express RMSNorm's formula in terms of `x`, `γ`, and `ε`. Then state what's empirically the same and what's different between RMSNorm and `bias=False` LayerNorm.
7. Pre-norm transformers use `x_{l+1} = x_l + f_l(LN(x_l))`. Why does this form give gradients that propagate well through arbitrary depth, while post-norm (`x_{l+1} = LN(x_l + f_l(x_l))`) attenuates gradients? Reference both the residual identity-path argument and the LN-Jacobian argument.
8. Suppose ε is set to `1e-30` instead of `1e-5`. In fp32, what happens to the LayerNorm forward pass when `σ² ≈ 0`? Why does the standard value `1e-5` work and `1e-30` not?

## What this enables

- **Stage 6 (MLP / pointwise FFN with GELU)** is structurally independent of LayerNorm — just two `nn.Linear`s and a nonlinearity. But stage 7 puts LayerNorm and MLP together inside the transformer block.
- **Stage 7 (transformer block)** is where stage 5 lands operationally. The block is `x = x + attn(LN_1(x))` followed by `x = x + mlp(LN_2(x))` (pre-norm). Each block has two `LayerNormalization` instances, each with its own γ, β. This is the structural reuse pattern: one module class, many independently-parameterized instances across the model.
- **Stage 8 (full GPT)** stacks `n_layers` transformer blocks, then applies a final `LayerNormalization` before the unembedding head. So a full GPT model has `2 * n_layers + 1` LayerNorm instances — each one of these classes.
- **Stage 13 (RoPE)** doesn't interact with LayerNorm directly, but the broader pattern of "normalization is per-token, positional info is separate" persists. RoPE applies rotation inside attention; LayerNorm normalizes the residual stream around attention. Two orthogonal mechanisms.
- **Future: RMSNorm migration**. Many modern models (LLaMA, Mistral, Qwen, Gemma) use RMSNorm instead of LayerNorm. The current `LayerNormalization` class with `bias=False` is partway there (γ only); a future swap would also drop the mean centering and divide by RMS instead of std. Architectural drop-in replacement.

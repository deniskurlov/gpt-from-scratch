# Stage 5 — LayerNorm

## 2026-05-14

## What I worked on
Building `LayerNormalization` from scratch in `src/normalization.py`: conceptual probe (per-token normalization, γ/β, ε placement, pre-norm vs post-norm, RMSNorm comparison), implementation, and 3 tests.

## Key concepts
- Formula: `y = γ * (x - μ) / √(σ² + ε) + β`, where μ and σ² are computed over the last dim (`d_model`) per token, γ and β are learned per-coordinate scale and offset, ε is a hyperparameter.
- **Normalize across `d_model` only, never across batch or T**. Across batch (BatchNorm): fails at batch=1 inference, train/eval skew via running stats, couples examples. Across T: would silently break causality (future-position info leaking into past-position normalization stats).
- γ, β are learnable **vectors** of shape `(d_model,)` applied via elementwise broadcast — diagonal scaling, **not** a full linear projection. So `nn.Parameter(torch.ones(d_model))` / `nn.Parameter(torch.zeros(d_model))`, **not** `nn.Linear`. The diagonal structure is intentional — general linear mixing is the job of attention's QKV projections and MLP weights, not of LayerNorm.
- ε is placed **inside** the sqrt: `√(σ² + ε)`. Gives bounded derivatives at `σ² = 0`; the alternative form `√σ² + ε` has a derivative kink at `σ = 0`.
- Pre-norm vs post-norm: pre-norm is `x' = x + f(LN(x))`; post-norm is `x' = LN(x + f(x))`. Modern standard (GPT-2 onward) is pre-norm. Pre-norm preserves a linear residual stream across blocks; post-norm wraps each block's update in LN, making the stream's evolution non-linear.

## What I got wrong
- **σ vs σ²** — wrote "σ is the variance"; σ is std, σ² is variance. Formula was correct; symbol naming was off.
- **"Per-feature" terminology pushback** — "feature" in LayerNorm context = coordinate (sense 1: standard-basis dim of `d_model`). Distinct from mech-interp "feature" (sense 2: arbitrary direction in activation space, potentially encoding an interpretable concept across multiple coordinates). γ, β are per-coordinate; mech-interp features are typically non-axis-aligned. Use "per coordinate" to avoid the ambiguity.
- **γ as `nn.Linear`** — first implementation attempt typed γ and β as `nn.Linear`. Wrong because γ is a learnable vector applied via elementwise broadcast (`*`), not a matrix applied via matmul. `nn.Linear` is reserved for full matrix-based linear projections; `nn.Parameter` is the right primitive for any learnable tensor that isn't a linear projection.
- **`mean = sum(x) / len(x)`** — Python's `sum` iterates over the first dim, `len` returns first-dim size. Computed mean over batch, not over `d_model`. Correct: `torch.mean(x, dim=-1, keepdim=True)`.
- **`sqrt(var ** 2 + eps)`** — `var` is already σ². Squaring gives σ⁴. Correct: `sqrt(var + eps)`. Also `math.sqrt` doesn't operate on tensors; needed `torch.sqrt`.
- **`gamma @ x_norm`** — matmul instead of elementwise. Driven by the conceptual error of treating γ as a matrix.
- **Missing `super().__init__()` + class not inheriting from `nn.Module`** — both at once on first draft. Two coupled bugs.
- **Test bug: `[-1].view(d_model)` on a `(B, T, 1)` tensor** — indexed wrong dim, tried to reshape 4 elements into 128. Replaced with `torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)` — no shape manipulation needed.
- **`from torch import layer_norm`** — Cursor autocomplete artifact, third project occurrence (after `_pytest.monkeypatch.V` and `torch._dynamo.utils.V`). Especially treacherous because the name *looks* plausibly relevant to the file's purpose, so inattentive review lets it through.

## Why this works
- **ε inside the sqrt**: `d/du[1/√u] = −1/(2 u^{3/2})`, singular at `u = 0`. With `u = σ² + ε ≥ ε > 0`, the gradient is bounded everywhere. With `u = σ²` and `+ε` outside, degenerate inputs (constant features → σ² = 0) blow up the gradient. ε-inside ensures backprop stability even at pathological corner cases.
- **γ, β recover expressive power**: pure normalization forces every coordinate to mean 0, variance 1 — strong constraint. γ, β are per-coordinate scale/offset that let the model learn what magnitude/mean each coordinate should actually have. Init γ=1, β=0 means LayerNorm starts as pure normalization; training learns departures from neutral.
- **Pre-norm gradient flow**: `∂x'/∂x = 1 + ∂(f(LN(x)))/∂x` in pre-norm; the "1" comes from the identity branch of the residual, giving an unattenuated gradient path. Pre-norm gradients stay O(1) at any depth; post-norm gradients decay as O(1/√L)^L (Xiong et al. 2020). Pre-norm enables warmup-free training at depth.
- **Linear residual stream in pre-norm**: `x_L = x_0 + Σ_l f_l(LN(x_l))`. The stream is a linear accumulator of block contributions plus the initial embedding — clean decomposition that makes mech-interp QK/OV circuit attribution tractable. Post-norm scrambles per-block contributions through inter-block LNs.

## Open questions
- RMSNorm migration later (drops mean centering, used by LLaMA / Mistral / Qwen). The `bias=False` toggle already supports γ-only LayerNorm; full RMSNorm requires also dropping `x - μ`. Worth an ablation at stage 10+ once a working LayerNorm baseline trains.
- Mixed-precision LayerNorm subtleties (fp16/bf16 internals usually promoted to fp32 then cast back). Irrelevant for my fp32-on-MPS project; relevant when scaling beyond Shakespeare.

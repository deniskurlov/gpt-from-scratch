# Stage 6 — MLP / Pointwise FFN

## 2026-05-15

## What I worked on
Building `MLP` (pointwise FFN with GELU) in `src/mlp.py` — conceptual probe (attention vs MLP division of labor, 4× expansion, GELU vs ReLU, bias=False rationale), 3 tests (1 + 10-parametrized + 1), brief walkthrough, toy prediction.

## Key concepts
- **Division of labor**: attention is the *only* position-mixing layer in a transformer; MLP is the *only* per-token nonlinear feature processor. Each block alternates: mix (attention) → transform (MLP). Neither does the other's job.
- **Attention's intrinsic nonlinearity is softmax** — structurally constrained (outputs are probabilities, sum to 1). MLP's GELU is structurally richer (gating + d_ff expansion gives ~4·d_model independent nonlinear gates per layer).
- **`d_ff = 4 · d_model` is the empirical sweet spot** between expressivity and param cost; not theoretical. MLP holds ~2/3 of each block's parameters (`8 · d_model²` vs attention's `4 · d_model²`).
- **GELU(x) = x · Φ(x)** where Φ is standard normal CDF. Smooth (no kink), no dead-unit problem, probabilistic gating interpretation (`x · P(X≤x)` for `X~N(0,1)`). Two forms: exact (erf) and tanh approximation; PyTorch's `F.gelu` defaults to exact.
- **`bias=False` is the modern default** (LLaMA, Mistral, Qwen). Three reasons: (a) pre-norm LayerNorm's β subsumes some of bias's role; (b) empirically no benefit at scale (Touvron 2023, Chowdhery 2022); (c) minor param savings compound.

## What I got wrong
- **`self.up_proj @ x`** — used `@` (matmul) on `nn.Linear` modules. `nn.Linear` is a callable module, not a matrix. Should be `self.up_proj(x)`. Same convention used since stage 2 — this was a regression, not a misunderstanding. (Insidious because the error message would have pointed at "no `@` operator" rather than the conceptual confusion.)
- **`def sigma(y): return F.gelu(y)`** local wrapper. Pure rename of `F.gelu` with no added functionality. Inline `F.gelu(...)` directly.
- **`4*d_model if not d_ff else d_ff`** — falsy check treats `d_ff=0` as None. Correct: `d_ff if d_ff is not None else 4*d_model`. Edge case probably never matters, but `is None` is the semantically precise check for "is this the sentinel value".
- **Type annotation typos** (`"B T d_mode"` and `"B T d_mdoel"`). Caught one revision, missed the other. Always re-read string literals.
- **`@pytest.mark.parametrize("d_ff, bias", [list_of_d_ffs, list_of_biases])`** — wrong structure for parametrize. Pytest wants a list of `(d_ff, bias)` *tuples*, not two parallel lists. The desired cross-product is done via **stacked decorators**: `@parametrize("d_ff", [...])` stacked with `@parametrize("bias", [...])` generates all combinations automatically.
- **`mlp(a*x) != a*mlp(x)`** in the nonlinearity test → `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`. `==` and `!=` between tensors produce element-wise bool tensors, not Python bools. `assert` can't evaluate a multi-element bool tensor. Fix: `not torch.allclose(mlp(a*x), a*mlp(x))` — `allclose` returns a Python `bool` directly.
- **"Large positive bias switches off the nonlinearity"** reasoning during bias-discussion Q4. Wrong: GELU(large +x) ≈ x (passthrough, not off). GELU(large -x) ≈ 0 (off). Bias-removal argument is better grounded in LayerNorm-β redundancy + empirical-no-benefit, not in nonlinearity-switching mechanics.

## Why this works
- **Sum of two linear maps + one nonlinearity ≠ a single linear map** (when there's expansion). The 4× expansion + GELU at the intermediate makes the MLP capable of representing functions that no single `nn.Linear(d_model, d_model)` can — including arbitrary continuous functions of the input by universal approximation. The expansion is what gives the intermediate enough independent gates to express compositional features.
- **Homogeneity violation test** (`mlp(a·x) ≠ a·mlp(x)` for `a ∉ {0, 1}` with bias=False) is the cleanest structural check for nonlinearity. A linear function `L` satisfies `L(a·x) = a·L(x)` (homogeneity). Any function that breaks this is non-linear. With `bias=False`, the test isolates GELU specifically; with biases, the test catches "biases or nonlinearity".
- **Pre-norm β redundancy with linear bias.** Pre-norm puts LayerNorm before the linear; LN has its own learnable β shift per coordinate. Bias on the linear is partially redundant — and the redundancy doesn't compose meaningfully because the next block's pre-norm will recenter again, washing out the bias contribution. The optimizer can't usefully distinguish "Linear's bias" from "extra weight on LN's β".

## Open questions
- **Whether to revisit attention to also use `bias=False`** for consistency. Stage 3/4's attention uses bias=True (default of `nn.Linear`); stage 6's MLP uses bias=False. Cosmetic inconsistency; could be made uniform at stage 7 if it bothers me. Probably worth doing — modern convention is bias=False throughout.
- **SwiGLU at stage 14** — reference saved at `notes/stage_14_swiglu_reference.md`. Open: re-verify the sigma-pi-vs-sigma framing when revisiting. The "result solid, theory fuzzy" disclaimer about Shazeer 2020 is worth a fresh look — maybe 2025+ papers have settled it.

# Stage 7 — Transformer Block

## 2026-05-16

## What I worked on
Building `Block` (the pre-norm transformer block) by composing stage-2 through stage-6 sub-modules in `src/model.py` — two LayerNorms, one MultiHeadAttention, one MLP, two Dropouts, wired with residual additions. 5 tests covering shape/dtype, causality, parameter count, residual structure, train/eval dropout behavior.

## Key concepts
- **Pre-norm two-phase forward**: `x ← x + Dropout(submodule(LN(x)))`, once for attention, once for MLP. Each phase is `(I + Dropout ∘ sub ∘ LN)` applied additively to the residual stream.
- **Two separate LayerNorms** (`ln1`, `ln2`), each with own γ/β. Not shared — they see different input distributions (LN_2 sees post-attention residual stream).
- **Dropout on the contribution, not the residual.** `x + Dropout(f(x))` zeroes a fraction of the *update*; the original `x` flows through unattenuated. `Dropout(x + f(x))` would corrupt the accumulated residual stream. Structural rule for any residual architecture.
- **`module.train()` vs `module.eval()`**: dropout active vs no-op. Tests that assert deterministic behavior MUST use `eval()` mode or `dropout=0.0`. PyTorch defaults to `training=True`.
- **Block is shape-preserving**: `(B, T, d_model)` in, same out, every intermediate. Residual stream stays at `d_model` throughout the block.
- **Per-block parameter count** ≈ 200K at `d_model=128, n_heads=4, d_ff=512`: MLP ~66%, attention ~33%, LayerNorms ~0.26%. MLP:attention ratio is 2:1.

## What I got wrong
- **"Stability" answer to Q2 was incomplete.** Said "you want a normalized residual stream before each nonlinearity for stability" — covers *why have LN at all* (correct), but not *why two separate LNs with own γ/β*. The substantive answer is different input distributions: LN_1 sees the residual stream entering the block; LN_2 sees it after attention writes its contribution. Each LN's γ/β learns the per-coordinate scale/offset for its specific distribution. Sharing forces one set of γ/β to fit both — less expressive.
- **"Wastes the MLP on first layer" reasoning for attention-then-MLP order.** Wrong framing — MLP isn't "wasted" under MLP-first; it still computes its per-token transformation. The substantive reason for attention-first: MLP gets *richer* (post-attention, context-aware) inputs vs *poorer* (pre-attention, isolated tokens). Also enables 2-layer compositional circuits (induction heads, name-movers) that structurally require attention to come first in each layer.
- **Parameter count arithmetic errors**. (a) LN size: wrote `2·128 + 2·512` — confused d_model with d_ff. Both LayerNorms operate on d_model (the residual stream's width); d_ff only exists inside the MLP, where there's no LayerNorm. (b) MLP: wrote `(4·128)² = 2¹⁸` — that's squaring d_ff. Correct: `2·d_model·d_ff = 2·128·512 = 2¹⁷`. Same arithmetic-on-shapes pattern I've hit at stages 3, 4, 5; the defensive habit (write shape arithmetic on paper, never compute in head) still hasn't fully landed.
- **Single Dropout instance vs two**. Initially wrote `self.dropout = nn.Dropout(p=dropout)` and called it twice. Tutor pushed back; I correctly argued dropout is stateless so functionally identical; ended up with two separate instances for clarity per the original 4b commitment. Convention-aligned but the discussion confirmed both work.
- **`d_ff` accepted by Block but not threaded to MLP**. Constructor signature had `d_ff: int | None = None`, but `__init__` did `MLP(d_model=d_model)` — silently dropping `d_ff`. Any caller passing `Block(..., d_ff=256)` would silently get MLP's own default of `4·d_model`. Silent-ignored-argument is the worst bug class because the call site looks right.
- **`ModuleNotFoundError`** when running `python src/model.py`. Path-resolution issue resolved at stage 2: must use `python -m src.model` from project root, not `python src/model.py`. The convention exists for a reason; I forgot it.
- **Causality test failed under default training mode**. Forgot that newly-constructed `nn.Module` is in `training=True` mode by default, so dropout is active and forward passes are nondeterministic. Test fix: `block.eval()` before testing the architectural property. The architectural property (causality) is independent of dropout's randomness; eval mode isolates it.
- **`super().__init__()` "inherits parent's parameters" wording** — **sixth stage** of the same imprecision. Methods are inherited automatically; `super().__init__()` initializes parent *instance state* (`_parameters`, `_modules`, `_buffers` dicts). Six stages of this minor wording slip; not yet fully internalized. Worth flagging explicitly in stage 8's walkthrough.

## Why this works
- **Pre-norm preserves a linear residual stream across blocks**: `x_L = x_0 + Σ_l (attn_l + mlp_l)`. Stage 5's argument extends to the block composition: each block writes additively to the stream, so the stream accumulates contributions cleanly. Post-norm wraps each block in LN, scrambling the per-block decomposition.
- **Gradient O(1) at any depth** under pre-norm (vs post-norm's O(1/√L)^L decay, Xiong et al. 2020). The identity branch of the residual gives an unattenuated gradient path: `∂x_{l+1}/∂x_l = 1 + ∂(f(LN(x_l)))/∂x_l` — the "1" is what enables warmup-free training at arbitrary depth.
- **Two separate LayerNorms** because the two phases see different input distributions. LN_1: post-MLP residual stream from the previous block. LN_2: post-attention residual stream from this block's first phase. Different statistical regimes; each γ/β learns its own optimum. The ~256 parameters per LN are negligible cost (~0.26% of block); the expressiveness gain is real.
- **Attention-then-MLP** because the MLP's per-token transformation is more useful on inputs that have already been routed across positions. Compositional circuits (induction heads, name-movers, etc.) require this ordering structurally.
- **Dropout on the contribution, not the residual**: `x + Dropout(f(x))` zeroes some elements of the update; `x` flows through unchanged. `Dropout(x + f(x))` would zero parts of the accumulated stream — destroys information from earlier layers. The residual stream is sacred; the per-layer updates are negotiable. Regularization belongs on the latter.

## Open questions
- **Attention `bias=True` inconsistency with MLP `bias=False`**. Currently attention uses `nn.Linear`'s default `bias=True`; MLP explicitly `bias=False` per modern convention. Cosmetic inconsistency, no correctness impact. Could fix during stage 8 (add `bias=False` to attention's QKV and out projections) for uniform convention with LLaMA-style models. Not urgent; the test for parameter count would need to adjust.
- **Recurring `super().__init__()` wording imprecision** — six stages now. Still not internalized. Flag aggressively in stage 8 if it recurs.
- **Attention-weight dropout** (the third Vaswani-2017 placement) deferred. Probably not needed at this scale; revisit if validation loss in stage 9 plateaus prematurely.

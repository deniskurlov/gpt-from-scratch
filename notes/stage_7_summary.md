# Stage 7: Transformer Block

## Summary

This stage built `Block` — the canonical pre-norm transformer unit that composes stages 3-6 into a single coherent module. Mechanically small (two LayerNorms + one MultiHeadAttention + one MLP + two Dropouts, wired with residual additions), but architecturally pivotal: this is the first stage where the previous components meet, and the design choices here (pre-norm vs post-norm placement, two LayerNorms vs shared, attention-then-MLP ordering, dropout placement on the contribution rather than the residual) are load-bearing for stage 8's full GPT and stage 9's training. The block does **mix-then-transform** in each layer: attention gathers information across token positions via its learned QK/OV circuits, then MLP processes the gathered information per-token via its `up_proj → GELU → down_proj` chain. Stacked `n_layers` times in stage 8, the block becomes the iterative-refinement engine that transforms initial embeddings into context-aware representations.

## The math

**Pre-norm block forward** (two phases, each `update = sub(LN(x))`, applied additively to the residual stream):

```
x ← x + Dropout₁(Attention(LN₁(x)))     # phase 1: position-mixing
x ← x + Dropout₂(MLP(LN₂(x)))           # phase 2: per-token transformation
```

In operator form: `Block(x) = (I + Dropout₂ ∘ MLP ∘ LN₂) ∘ (I + Dropout₁ ∘ Attention ∘ LN₁)(x)`.

Crucially, the residual stream `x` is updated by **additive** contributions in both phases. With pre-norm, the stream stays a linear accumulator across blocks: `x_L = x_0 + Σ_l (attn_l + mlp_l)`. This is the structural property that makes deep transformers trainable and mechanistically analyzable (stage 5's pre-norm argument extends to the block composition).

**Dropout placement on the contribution, not the residual.** `x + Dropout(f(x))` zeroes a fraction of the *update* to the stream; the original `x` flows through unattenuated. The alternative `Dropout(x + f(x))` would zero parts of the accumulated stream itself, destroying information from earlier layers. The structural rule: regularization belongs on the contribution, never on the linear accumulator.

**Per-block parameter count** (for `d_model=128, n_heads=4, d_ff=512, attention bias=True, MLP bias=False`):
- 2× LayerNormalization: `2 · 2 · d_model = 512 = 2⁹`.
- MultiHeadAttention: `4 · d_model² + 4 · d_model = 4·128² + 4·128 = 66,048 ≈ 2¹⁶`.
- MLP: `2 · d_model · d_ff = 2·128·512 = 131,072 = 2¹⁷`.
- 2× Dropout: 0 (stateless).
- **Total per block: ≈ 197,632.** MLP ≈ 66%, attention ≈ 33%, LayerNorms ≈ 0.26%. MLP:attention parameter ratio is 2:1.

## The code

- `src/model.py:28` — `Block(nn.Module)` class.
  - `__init__(T_max, n_heads, d_model, d_ff=None, dropout=0.1)`: instantiates two `LayerNormalization(d_model)`, one `MultiHeadAttention(T_max, n_heads, d_model)`, one `MLP(d_model, d_ff)`, and two `nn.Dropout(p=dropout)`. Each sub-module owns its own configuration; no raw hyperparams stored on `self` beyond what sub-modules carry.
  - `forward(x: (B, T, d_model)) → (B, T, d_model)`: two pre-norm phases with residual additions. Three substantive lines (phase 1, phase 2, return). Shape-preserving on the residual stream throughout.
  - The class lives alongside stage-2's `TokenEmbedding` and `LearnedPositionalEmbedding` in `src/model.py` rather than getting its own file.
- `tests/test_block.py` (or extension of `test_model.py`) — 5 pytest cases:
  - `test_block_shape_and_type`: shape/dtype preservation.
  - `test_block_causality`: position-`T-1` modification doesn't affect earlier outputs, in `eval()` mode (dropout disabled).
  - `test_block_parameter_count`: with `d_ff = 6 · d_model` (intentionally breaking the 4× coincidence), verifies the sum of contributions from 2 LNs + attention + MLP.
  - `test_block_residual_structure_at_p0`: at `dropout=0.0`, manual residual composition matches `block(x)` via `torch.allclose`. Catches composition-order bugs.
  - `test_block_dropout_train_vs_eval`: `block.train()` gives different outputs across calls (dropout randomness); `block.eval()` gives identical outputs (dropout off). Catches train/eval-mode bugs.

## Design choices and why

- **Pre-norm structure** (`x = x + f(LN(x))`, not `x = LN(x + f(x))`). Resolved at stage 5 — pre-norm preserves a linear residual stream across blocks, gives O(1) gradient flow at any depth (vs post-norm's O(1/√L)^L attenuation), enables warmup-free training, and makes mechanistic-interpretability circuit analysis tractable. Modern standard since GPT-2.
- **Two separate `LayerNormalization` instances** per block, each with its own γ/β, rather than one shared LN. LN_1 sees the residual stream entering the block (post-MLP output from the previous block); LN_2 sees it after attention has written its contribution. Two different input distributions; each LN's γ/β learns the appropriate per-coordinate scale/offset for its specific distribution. Sharing would force one γ/β to fit both — strictly less expressive at marginal parameter savings (~0.26% of block params).
- **Attention before MLP** in each block. Attention gathers cross-position context; MLP processes the gathered context per-token. Reversing the order (MLP first) would have the MLP transform tokens in isolation before attention can route context — qualitatively poorer inputs for MLP. The convention also enables 2-layer compositional circuits (induction heads, name-movers, etc.) that depend on attention-first ordering. Mechanistic-interp circuits would be impossible to form under MLP-first.
- **Two separate `nn.Dropout` instances** (functionally identical to a shared instance, since `nn.Dropout` is stateless). Convention from nanoGPT and most reference codebases. Sharing one instance works correctly but obscures the two distinct dropout points; two instances make the structure visible in `__init__`.
- **Dropout placed on the contribution, not on the residual.** `x + Dropout(f(x))` zeroes some of the *update*; `x` flows through unchanged. The alternative `Dropout(x + f(x))` would corrupt the accumulated residual stream. The "regularization on contribution, never on accumulator" rule extends to any residual architecture.
- **`p = 0.1` dropout default.** Vaswani 2017 / BERT / GPT-2 convention. For Shakespeare-scale training (1MB corpus, model may overfit), dropout is beneficial. Modern open-scale models (LLaMA, Mistral) use `p = 0` because dataset size provides regularization, but at this project's scale the small dropout helps. `dropout` is constructor-exposed for tuning during stage 9.
- **Attention-weight dropout NOT included** (the third Vaswani-2017 placement, inside the attention softmax). Would require modifying `src/attention.py`; deferred for now. Output-position dropouts (on attn-output and MLP-output, both pre-residual) cover the main regularization use case.
- **Minimal `self` attributes.** Only the five sub-modules (`ln1, ln2, attn, mlp, dropout1, dropout2`); no raw hyperparams duplicated on `self`. Sub-modules already store their own config (`block.attn.n_heads`, `block.ln1.d_model`, etc.). Avoids the silent-divergence bug class where a stored hyperparam on Block conflicts with the sub-module's actual config.
- **`d_ff` threaded through to MLP.** First-pass implementation silently dropped `d_ff` (constructed `MLP(d_model=d_model)` without passing `d_ff`). Fixed: `MLP(d_model=d_model, d_ff=d_ff)`. "Silent ignored argument" is one of the worst bug classes — the call site looks right but the model is built differently than requested.

## Errors and corrections

- **Initial `forward` block math** used "1 + MLP ∘ LN_2" operator notation with composition applied to a value. The "1" is the identity *operator* (not scalar 1); the outer `∘` is operator-application, not function composition. Substance was right; notation cleanup needed.
- **"Stability" answer to Q2 was incomplete.** Said "you want a normalized residual stream before each nonlinearity for stability" — which addresses *why have LayerNorm before each phase* (correct) but not *why two separate LayerNorms with their own γ/β*. The separate-vs-shared distinction is about γ/β learning per-input-distribution scales (LN_1 sees one distribution, LN_2 sees another after attention writes). Sharing forces one set of γ/β to fit both.
- **"Wastes MLP on the first layer" reasoning for attention-then-MLP order**. Partial; MLP isn't *wasted* under either ordering — it computes a per-token transformation either way. The substantive reason is that attention-first gives MLP *richer* (context-aware) inputs; the MLP can then transform features that have already been routed across positions. And 2-layer compositional circuits (induction heads) structurally require attention-first per block.
- **Parameter count arithmetic errors**: (a) LN size used `2·128 + 2·512` — confused d_model with d_ff; LayerNorms operate on d_model both times. Fix: `2·128 + 2·128 = 512`. (b) MLP used `(4·128)² = 2¹⁸` — that's d_ff², not the correct `2·d_model·d_ff = 2·128·512 = 2¹⁷`. Same arithmetic-on-shapes error pattern as previous stages; defensive habit (write shape arithmetic on paper, don't compute in head) still hasn't fully landed.
- **Single Dropout instance vs two**. Wrote `self.dropout = nn.Dropout(p=dropout)` and called it twice. Pushed back during review: dropout is stateless so functionally identical. Accepted the reuse-one-instance pattern, but also committed to two separate for clarity. Final code has two — matches convention; either works.
- **`d_ff` silently dropped**. Constructor accepted `d_ff` parameter but didn't pass it to MLP. Fix: `self.mlp = MLP(d_model=d_model, d_ff=d_ff)`. Caught in review; would have silently used MLP's default of `4 · d_model` for any caller specifying `d_ff` differently.
- **`from src.data import ...` ModuleNotFoundError** when running `python src/model.py`. Path-resolution issue from stage 2: running scripts directly puts `src/` on sys.path but not project root, so `from src.X` imports fail. Fix: `python -m src.model` from project root. The convention was already documented in stage 2's discussion; refresher worth pinning.
- **Causality test originally without `block.eval()`** would have failed because dropout (active in training mode by default) makes two forward passes nondeterministic. Fix: `block.eval()` before testing the architectural property. The train/eval mode distinction is the new artifact introduced in stage 7 (compared to stages 3-6 which had no train-mode-dependent layers).
- **Recurring `super().__init__()` wording imprecision** — sixth stage of "inherits parent's parameters/methods" mis-framing. Still hasn't been internalized. Worth flagging more aggressively if it recurs at stage 8.

## Self-quiz

1. Write the pre-norm block's forward in math (two-line form). For each phase, identify what the LayerNorm normalizes (input shape and dim), what the sub-module returns (shape), where the Dropout acts, and where the residual addition happens. Then state why "regularization on the contribution, never on the residual" is structurally important.
2. Compare per-block gradient flow for pre-norm vs post-norm. State the gradient norm scaling at depth L for each (cite Xiong et al. 2020 if remembered). Why does this difference make pre-norm trainable without learning-rate warmup at any depth?
3. The block has two `LayerNormalization` instances rather than one shared LN. State the structural reason in one sentence (each LN sees a different input distribution). What would shared γ/β cost the model in terms of expressiveness, and how much would it save in parameters (compute the percentage at d_model=128)?
4. Why is the canonical ordering attention-then-MLP per block, not MLP-then-attention? Give two reasons: (a) MLP's input quality under each ordering, (b) what 2-layer compositional circuits (e.g., induction heads) depend on.
5. Compute per-block parameter count for a typical small-model config (`d_model=128, n_heads=4, d_ff=4·d_model=512, attention bias=True, MLP bias=False`). Sum the contributions from 2 LayerNorms + attention + MLP. State the percentage breakdown (LN, attention, MLP).
6. What does `nn.Dropout(p=0.1)` do (a) in `module.train()` mode, (b) in `module.eval()` mode? State the scaling factor for surviving activations in train mode and explain why it's there (preserves expected magnitude across modes). What goes wrong if you forget `model.eval()` in an inference loop?
7. Explain why `x = x + Dropout(f(x))` is the correct placement vs the wrong alternative `x = Dropout(x + f(x))`. Reference the linear-accumulator property of the pre-norm residual stream (stage 5) and what happens to information from earlier layers under each form.
8. A trained transformer's `Block` is one of `n_layers` such units stacked into the full model. The residual stream entering block `l` is `x_0 + Σ_{l'<l} (attn_{l'} + mlp_{l'})` — a sum of contributions plus the embedding. What property of pre-norm makes this clean decomposition possible? What would post-norm look like for the same depth, and why is the decomposition no longer clean?

## What this enables

- **Stage 8 (full GPT)** stacks `n_layers` `Block` instances plus token+positional embeddings (stage 2) plus a final `LayerNormalization` + unembedding head into the canonical decoder-only architecture. Stage 7's `Block` is the structural unit that gets repeated; everything else in stage 8 is the wrapper (embeddings in, blocks-stacked, projection to vocab logits).
- **Stage 9 (training loop)** drives many forward passes through the full GPT (which is many `Block` calls per pass) and computes gradients via cross-entropy. The dropout in stage 7's Block is what regularizes training — `model.train()` activates it, `model.eval()` disables it for validation.
- **Stage 12 (KV cache)** caches K and V across generation steps. The Block remains structurally unchanged; what changes is `MultiHeadAttention` internally — KV cache lives there. Stage 7's clean module boundary makes this drop-in.
- **Stage 13 (RoPE)** replaces the additive positional embedding (stage 2) with rotation applied to Q and K inside attention. Block is unchanged; attention is restructured. Same modular benefit as stage 12.
- **Mechanistic-interpretability analysis** (when stage 9 produces a trained model): each `Block` instance can be analyzed as `(I + Dropout ∘ Attention ∘ LN) ∘ (I + Dropout ∘ MLP ∘ LN)`, and the additive residual stream lets you attribute each block's contribution separately. Stages 7-8's clean design is what makes circuit attribution (Elhage et al. 2021) tractable on the eventually-trained model.

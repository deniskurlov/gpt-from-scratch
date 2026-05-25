# Stage 13: RoPE + Sliding-Window KV Cache

## Summary
Replaced absolute positional embedding (`LearnedPositionalEmbedding`, used inside `GPT.forward` since stage 2) with Rotary Position Embedding (RoPE) applied to Q and K inside `MultiHeadAttention`. Built a new `RoPE` class in `src/embedding.py` that rotates Q and K vectors in 2D subspaces of head_dim at d/2 frequencies `θ_i = base^(-2i/d)` with `base=10000`. Refactored RoPE to compute cos/sin **on the fly** (no T_max-sized cached table) and the causal mask to be constructed dynamically (no `T_max × T_max` buffer). Extended `KVCache` to support sliding-window: `max_size` cap, `total_appended` counter, `window_start` property; oldest entries dropped automatically when cache exceeds max_size. `GPT.generate` now creates cache with `max_size=self.T_max`. Retrained on TinyShakespeare with RoPE (training trajectory tracked stage-10's absolute-pos-emb curve to within ±0.05 final loss as predicted). Long-context generation (max_new_tokens=1000+) now works without architectural crashes; cache size stays bounded at T_max via sliding. The T_max ceiling that stage 12 surfaced as an architectural limit is now genuinely lifted — both because RoPE has no precomputation ceiling and because the cache slides instead of growing unboundedly.

## The math

**RoPE's defining property** (relative-position dependence of inner products):

Rotation `R_m` applied to query `q_m`, `R_n` to key `k_n`. The score:
```
<R_m · q, R_n · k> = q^T R_m^T R_n k = q^T R_{n-m}^T k       (using R^T = R^{-1})
                  = q^T R_{m-n} k                            (substituting)
```
This works because rotations are orthogonal (`R_n^T = R_n^{-1} = R_{-n}`) and form a group (`R_a R_b = R_{a+b}` when they commute). The score depends only on `m − n`, not on `m` and `n` individually.

**Block-diagonal structure** (why pairwise 2D rotations, not arbitrary d×d):

`SO(d)` for `d ≥ 3` is non-abelian: `R_a R_b ≠ R_b R_a` generically, so `R_a R_b ≠ R_{a+b}`. The relative-position property requires commutativity. The **maximal abelian subgroups** of `SO(2n)` (maximal tori) are isomorphic to `SO(2)^n` — direct sums of 2D rotations. So RoPE *must* decompose the d-dimensional head into d/2 independent 2D subspaces, each rotating in its own plane. Within each 2D subspace, SO(2) is abelian, so the group law holds and the relative-position property follows from direct-sum.

The full rotation:
```
R_m = diag(R(m·θ_0), R(m·θ_1), ..., R(m·θ_{d/2-1}))
```
where each `R(m·θ_i)` is a 2×2 rotation matrix.

**Frequency schedule** `θ_i = base^(-2i/d)`:

Exponential spacing across d/2 frequencies. For `head_dim=64, base=10000`:
- Pair 0: θ_0 = 1, wavelength ~6 positions (fast).
- Pair d/2-1: θ_{31} ≈ 10^-4, wavelength ~62800 positions (slow).
- Coverage: 4 orders of magnitude in position scale.

Two reasons exponential is right:
1. **Multi-scale resolution**: fast pairs discriminate local positions; slow pairs discriminate long-range. With linear spacing all pairs would be at similar scale.
2. **No periodicity collapse**: with d/2 different frequencies (and `θ_i / θ_{i+1}` irrational), the joint period is astronomically large.

Inherited from Vaswani 2017's sinusoidal positional encoding — same frequency schedule, different application (rotational vs additive).

**Why rotate Q and K but not V**:

The relative-position property emerges from the inner product `<R_m q, R_n k> = q^T R_{m-n} k`. V isn't part of any dot product with another rotated quantity — it's the content vector retrieved by the attention scores. If V were also rotated, the output `Σ_n score_{m,n} · V_n` would carry absolute-position content into the residual stream; downstream layers would have to learn to "un-rotate." Design principle: position info affects *which positions attend to which* (Q·K^T); V remains a content channel.

**Sliding-window mask formula**:

The on-the-fly mask: `mask[i, j] = (j ≤ T_total - T_new + i)`. This handles training, prompt-load, and sliding-window uniformly because `window_start` cancels:
- Query at relative pos i (sorted index) → absolute pos `window_start + T_total - T_new + i`.
- Key at relative pos j → absolute pos `window_start + j`.
- Causal: key_pos ≤ query_pos ⟹ `window_start + j ≤ window_start + T_total - T_new + i` ⟹ `j ≤ T_total - T_new + i`. ✓ The window_start terms cancel; only relative spacing matters for causality.

## The code

- `src/embedding.py` — new `RoPE(nn.Module)` class:
  - `__init__(head_dim, base=10000)`: computes `inv_freq = base ** (-2 * arange(head_dim // 2) / head_dim)`; registers as buffer with `persistent=False`.
  - `forward(x, start_pos)`: computes `positions = arange(start_pos, start_pos + T_new)` on the fly with `device=x.device`; `angles = outer(positions, inv_freq)`; cos and sin from angles; rotation via paired-dim arithmetic and reassembly with stack + flatten.
  - Smoke test verifies inner-product invariance under shared relative position (scores at (2,5) match scores at (7,10), both with m−n=−3).

- `src/cache.py` — `KVCache` extended for sliding-window:
  - `__init__(max_size=None)`: accepts optional window cap; tracks `total_appended` counter.
  - `append`: increments `total_appended` by `T_new`; drops oldest if exceeds `max_size` (slice on dim=-2).
  - `window_start` property: derives absolute position of oldest cached entry from `total_appended - cache_size`.

- `src/attention.py` — `MultiHeadAttention.forward`:
  - Constructor now requires `rope_base`; instantiates `self.rope = RoPE(head_dim, base=rope_base)`.
  - Removed precomputed mask buffer (commented out).
  - In forward: records `total_appended_before` before append; uses `cache.window_start` after append.
  - RoPE applied to Q at `start_pos=total_appended_before`; to K_full at `start_pos=cache.window_start` (the latter is 0 unless sliding has occurred).
  - Causal mask computed on the fly: `j_range[None, :] <= (T_total - T_new + i_range[:, None])`. No T_max dependency.

- `src/model.py`:
  - `Block.__init__` propagates `rope_base` to `MultiHeadAttention`.
  - `GPT.__init__` propagates `rope_base` (and now all params required; no defaults).
  - `GPT.forward` removed `pos_emb(positions)` addition; just `self.tok_emb(ids)`. The `start_pos`/`positions` lines are commented out (preserved for the eventual three-way pos-encoding ablation).
  - `GPT.generate` creates per-layer cache with `max_size=self.T_max` for sliding window; new optional `verbose: bool = False` parameter prints cache state every 10 steps for debugging.

- `src/config.py`: `GPTConfig` gained `rope_base: float = 10_000.0` field. `TrainConfig.total_steps` unchanged at 5000.

- `src/train.py`: passes `rope_base=cfg.model.rope_base` to GPT construction. Otherwise unchanged.

- `src/sample.py`: unchanged; loads checkpoint via `GPT(**ckpt['config']['model'])` which now includes `rope_base`.

No new tests added. Stage 13 was integration-tested via: RoPE inner-product invariance check (in `src/embedding.py`'s `__main__`), the model.py smoke test (initial loss ≈ log V), retraining (loss converged to ~1.6 train / ~1.7 eval, within ±0.05 of stage 10), and long-context sample generation (`max_new_tokens=1000` runs without crash; verbose cache state output confirms `len(cache[0])` caps at T_max=256 and `window_start` grows monotonically past step 256).

## Design choices and why

- **Store unrotated K in cache** (not rotated). Sliding-window cache is incompatible with pre-rotated K: when the window slides, the cached K's were rotated at their *original* absolute positions, which no longer match their position within the current window. Storing unrotated K means we re-apply RoPE at attention time at whatever positions the cache currently represents. Cost: O(T) extra compute per step (re-rotating cached K every forward). Benefit: sliding-window correctness without re-decoding cached values.

- **Compute cos/sin and mask on the fly** rather than precomputing buffers. Eliminates the T_max ceiling that the precomputed approach inherited from stage 8's absolute pos emb. Trade-off: tiny per-forward overhead (a few `torch.cos` calls on small tensors, a couple of `arange` + comparison ops); benefit: no architectural ceiling regardless of generation length. The precomputed approach is a production optimization; for pedagogical clarity and ceiling-removal, on-the-fly is correct.

- **`inv_freq` registered with `persistent=False`**. The frequency schedule is deterministic from `head_dim` and `base` — recomputable at init. Saving it to state_dict is redundant and creates forward-compatibility friction (different T_max or head_dim across checkpoints). `persistent=False` keeps device tracking via `register_buffer` without bloating the checkpoint.

- **Sliding-window cache, not unbounded growth**. With `max_size=T_max`, the model behaves at inference exactly as it was trained: each query attends to at most T_max preceding tokens. Memory stays bounded regardless of generation length. The alternative (unbounded cache + on-the-fly mask) works without crashing but grows memory linearly, and the model wasn't trained at positions beyond T_max, so attention over a larger window doesn't help generation quality at this scale.

- **`max_size=self.T_max` as the sliding window default**. Matches training-time context length. Could be exposed as a `generate` argument for tuning; not done for stage 13's minimal scope.

- **Mask formula unification** to `j ≤ T_total - T_new + i`. The earlier `j ≤ T_cached_before + i` formulation broke when sliding caused `T_total ≠ T_cached_before + T_new`. The new formulation uses post-append cache state (`T_total = K_full.shape[-2]`) and derives the offset from there; the `window_start` cancels because both Q and K share that offset.

- **`rope_base` propagated through the full constructor chain** (GPTConfig → GPT → Block → MultiHeadAttention → RoPE). Each level gains the parameter; same pattern as `dropout` and `d_ff`. The dataclass remains the single source of truth for defaults.

- **`GPT.__init__` made all-required** (no defaults for `d_ff`, `dropout`). Strict construction enforced; ad-hoc calls (smoke test) must pass everything. The dataclass + `**asdict(cfg.model)` is the canonical call pattern.

- **Removed pos_emb from GPT.forward but kept the import path commented**. RoPE-only setup for stage 13; the absolute-pos-emb code path is preserved as comments for the planned three-way ablation (Learned / Sinusoidal / RoPE).

## Errors and corrections

- **Frequency formula factor-of-2 bug, twice**. First attempt: `inv_freq = base ** (-2 * torch.arange(0, head_dim, 2) / head_dim)`. Two stacked factors of 2: `-2 *` AND `arange(step=2)`. Resulted in exponents `-4i/d` instead of `-2i/d`. Fix: drop the `-2 *` (since step=2 provides the 2i factor) or use `arange(head_dim // 2)` (step=1 with the multiplier). Both yield correct frequencies. The bug recurred briefly because the slogan "exponent times 2 over d" doesn't pin down which factor of 2 lives where.

- **`base = 1000.0` instead of `10000.0`** as default. RoPE convention is 10000 (Vaswani-2017 inheritance). Off by an order of magnitude; would have produced shorter wavelengths (slowest pair wrap every 6280 positions instead of 62800). Cosmetic for a 256-context model but conceptually wrong.

- **`RoPE.forward` returned the unrotated `x`** in the first draft. The rotation was computed (`torch.stack(...).flatten(...)`) but never bound to a variable; the return statement returned the input. The class silently no-op'd. Caught only by writing the inner-product invariance smoke test, which immediately failed: scores didn't match for shared relative position.

- **`inv_freq` not registered as buffer** initially. Plain attribute assignment (`self.inv_freq = inv_freq`) doesn't migrate with `model.to(device)`. Cross-device error surfaces at first forward call after the move (`torch.outer(positions_on_mps, inv_freq_on_cpu)`). Same lesson surfaced in stage 12 with the mask buffer; same fix (`register_buffer`).

- **`K_full = self.rope(...)` assigned to wrong variable**. First attempt wrote `k = self.rope(K_full, ...)`, then `scores = q @ K_full.transpose(...)`. The rotation was computed and discarded; the score matmul used unrotated K. Insidious because the code "reads as if RoPE is applied" but the variable name on LHS doesn't match the consumer downstream. Caught only because the next operation used K_full directly.

- **`rope_base` as required positional argument**. Adding `rope_base: float` (no default) to `MultiHeadAttention.__init__` broke every existing call site (Block, smoke tests). Cascaded fix through Block → GPT, and required updating the model.py smoke test to pass all params explicitly. Lesson: when adding required params to constructors with many existing call sites, audit the chain before testing.

- **T_max overflow in cached RoPE generation**. First implementation precomputed `cos_cached, sin_cached` to shape `(T_max, head_dim/2)`. When generation exceeded T_max=256, `self.cos_cached[start_pos : start_pos + T_new]` returned a truncated slice (Python slicing is lenient — capped at the buffer size). Downstream `a * cos` then mismatched the larger tensor dim. Surface error was deep in attention (shape mismatch in `masked_fill`); root cause was the precomputed table size. Fix: compute cos/sin on the fly in `RoPE.forward` using `start_pos` to generate the right positions; eliminates the table-size ceiling entirely.

- **Naive truncation worked past T_max but cached generation didn't, even with RoPE**. Initial confusion: "if naive can generate past T_max via context truncation, why can't cached generation past T_max also work?" Resolution: naive truncation discards old K, V and recomputes fresh ones at positions reset to [0, T_max), masking the model's positional embedding bounds. Cached generation preserves the cached K, V at their original absolute positions, which keeps growing the position counter past T_max. The truncation "lie" worked specifically because naive doesn't preserve state. Cached needs a different mechanism: sliding window (which we implemented).

- **`type=bool` argparse footgun, recurrence**. `--use-cache False` in sample.py was parsed as `True` because `bool("False") = True` in Python (any non-empty string → True). The fix is `action=argparse.BooleanOptionalAction` (or `action='store_false'` with renaming). Same lesson as stage 11's argparse fix; the gotcha is durable enough to bite twice.

- **State_dict mismatch when loading old checkpoint with new architecture**. After removing the precomputed mask buffer from MultiHeadAttention, the old checkpoint's state_dict still contained `blocks.X.attn.mask` keys that the new model didn't have. `load_state_dict` errored on unexpected keys. Quick fix: `strict=False` (safe because the mask was deterministic, not learned). Proper fix: retrain. The lesson: any buffer registered with `persistent=True` (default) becomes a checkpoint compatibility burden when the buffer is later removed.

- **MPS silent garbage on out-of-range nn.Embedding indices** (carried over from stage 12). The T_max overflow in attention mask manifested as a shape mismatch in `masked_fill` because MPS's lenient handling of out-of-range positional indices in the (formerly used) `pos_emb` returned wrong-sized tensors instead of raising IndexError. Fixed structurally by removing pos_emb from the architecture; RoPE has no such failure mode (computed in closed form from any position).

- **Sliding-window mask formula derivation**. Initial confusion about whether `T_cached_before` or `T_total - T_new` should be the offset in the mask. The sliding case revealed they're not the same: in steady-state sliding, `T_cached_before = max_size` (cache was full before append) but the *effective* offset is `T_total - T_new = max_size - T_new`. The post-append `T_total - T_new` is the right formulation — it's the number of pre-existing entries (post-drop) in the cache. The non-sliding case has `T_cached_before = T_total - T_new` by construction (no drops), so the new formula reduces correctly. Derivation: query at sorted-pos i lives at absolute position `window_start + T_total - T_new + i`; key at sorted-pos j lives at `window_start + j`; causal constraint cancels window_start.

- **Variable shadowing in Q/K assignments** (recurrent from stage 12). The pattern "k = self.rope(K_full, ...)" assigning rotated K_full to the wrong-named variable, while the score matmul uses `K_full`, was a refactor hazard that bit twice. Lesson: when renaming variables across multi-step computations, the LHS at each step must match the consumer downstream.

## Self-quiz

1. **The defining identity.** Derive `<R_m · q, R_n · k> = q^T R_{m-n} k`. Which property of rotation matrices makes this work? Why does this property only hold cleanly for direct sums of `SO(2)` rotations, and what would break if you tried to use a general `R_m \in SO(d \geq 3)`?

2. **Block-diagonal necessity.** Suppose you parameterized RoPE with a single d×d rotation `R_m \in SO(d)` rotating in some d-dimensional plane (not block-diagonal). Demonstrate that the relative-position property `R_a R_b^T = R_{a-b}` fails for generic `a, b` when `d \geq 3`. What's the maximal abelian subgroup of `SO(2n)`, and how does that constrain the RoPE construction?

3. **Frequency schedule.** Why `θ_i = base^(-2i/d)` with `base=10000`? Compare to (a) linear spacing `θ_i = i/(d/2)`, (b) constant spacing (all pairs at same frequency), (c) `base=1` (all θ_i = 1). For each alternative, describe what RoPE would compute and why it's worse than the standard schedule.

4. **Q/K rotation, V not.** Explain in two ways why V is left unrotated: (a) algebraically — why doesn't the relative-position property emerge if V is rotated? (b) Semantically — what role does V play in attention, and why is it cleaner to keep V position-free.

5. **Sliding-window cache without RoPE.** Suppose you tried to build a sliding-window cache with absolute positional embeddings (stage 2's `LearnedPositionalEmbedding`) instead of RoPE. Why doesn't it work? What state does the cached K/V "remember" that breaks when the window slides? How does RoPE's "store unrotated K, re-rotate at attention time" pattern sidestep this?

6. **The unified mask formula.** Derive `mask[i, j] = (j ≤ T_total - T_new + i)` from absolute positions. Show how it reduces to (a) the lower-triangular causal mask during training, (b) all-True for single-token cached inference, (c) the correct shifted lower-triangular in the sliding-window case. Why does `window_start` cancel out in the derivation?

7. **Precomputation vs on-the-fly.** Stage 13's first attempt precomputed `cos_cached, sin_cached` to shape `(T_max, head_dim/2)` and ran into the T_max ceiling. Argue that switching to on-the-fly computation in `RoPE.forward` is *correct* (same numerics) and that the precomputed approach is a production optimization for inference at fixed max position. What's the cost of on-the-fly vs precomputed? Where in the attention pipeline does it become the bottleneck (if anywhere)?

8. **Why didn't RoPE significantly improve generation quality at this scale?** Even though the architecture lifted the T_max ceiling, the actual output samples past T_max look qualitatively similar to absolute-pos-emb output. What's the gap between "architecture supports long context" and "model actually uses long context"? Connect to model capacity, training data, and the difference between local-pattern memorization and long-range reasoning.

## What this enables

- **Stage 14 (optional, SwiGLU)**: pure architectural change inside MLP. Doesn't interact with RoPE or the cache. Same training pipeline; potentially marginal quality improvement at this scale.

- **Stage 15 (optional, GQA)**: Grouped Query Attention. Reduces K/V cache memory by factor of (n_heads / n_kv_heads). Particularly relevant for production inference — combines with sliding-window cache to give bounded memory across long contexts. Architectural change to attention; no impact on RoPE itself.

- **Three-way positional-encoding ablation** (the "stage 13b" we discussed): with the `LearnedPositionalEmbedding` import preserved as a comment, adding a `pos_emb_type: Literal["learned", "sinusoidal", "rope"]` config field + dispatch in `GPT.__init__` is a clean ~30-line extension. Train all three variants on the same config; compare training loss, eval loss, sample quality. Direct experimental evidence for the architectural progression.

- **Long-context inference**: the model now generates indefinitely without crashing. Quality past `T_max` is bounded by what was learned during training, but the architecture no longer imposes the ceiling. To meaningfully exploit long context would require larger model + training at longer sequence lengths — the architectural prerequisite is now in place.

- **Stage 14/15 → post-training (SFT/DPO) extension** (the nanochat-style trajectory we discussed): after the architecture work, the natural extension is to leave the model unchanged and add a supervised fine-tuning + preference-optimization layer. Stage 13's RoPE makes this practical at long context lengths.

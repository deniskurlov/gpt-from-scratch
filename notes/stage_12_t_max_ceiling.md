# Stage 12 — T_max Ceiling

## 2026-05-23

## What I worked on
Implementing the KV cache (stage 12) and discovering its T_max ceiling — the cached path breaks past the model's positional-embedding range, while the naive path silently truncates and keeps going.

## Key concepts
- **KV cache stores K and V per layer; Q is recomputed each step.** Q changes every step because it depends on the new token's hidden state; K and V from past tokens are functions of past hidden states (fixed once computed) and therefore cacheable. The name is literal — no QKV cache, no scores cache.
- **Cached attention shape**: at step T+1 with cache, score matrix shrinks from `(B, n_heads, T, T)` to `(B, n_heads, 1, T_total)` — a single new query against all cached + new keys. Per-step attention drops from O(T²) → O(T).
- **`start_pos = len(cache[0])`** drives the positional-embedding offset in `GPT.forward`. `positions = arange(start_pos, start_pos + T_new)`. Layer 0's cache is authoritative because all layers grow together.
- **Mask slice unifies training and inference**: `self.mask[T_cached_before : T_cached_before + T_new, : T_total]`. Reduces to `self.mask[:T, :T]` when no cache (training); becomes a single row of all-True for single-token inference.
- **Naive's `ids[:, -T_max:]` truncation lies to the model about absolute position** — positions restart from 0 each call, so `pos_emb` never sees out-of-range indices. The cached path can't tell that lie because cached K, V have absolute positions baked in via `pos_emb + tok_emb` before attention.

## What I got wrong
- **"We cache attention scores (Q·K^T)"** — wrong. Q changes every step, so no past computation involves the current Q; nothing about scores from prior steps is reusable. Only K and V are stable. Correction surfaced when I tried to articulate it the second time.
- **Multiple rename misses during the T → T_new refactor in attention**: forgot the `.reshape(B, T, self.d_model)` line. NameError at runtime. The variable got renamed everywhere it was *defined* but not everywhere it was *used*.
- **Tuple unpacking missed in Block.forward**: wrote `attn_out = self.attn(self.ln1(x), cache)`, then `self.dropout1(attn_out)` errored because `attn_out` was the `(output, cache)` tuple, not the tensor. Same kind of bug bit the smoke test in `attention.py` — `out = mha(x)` then `out.shape` on the tuple. **General lesson**: when a signature changes to return a tuple, every existing call site silently breaks. The refactor needs to be done in one sweep across the codebase.
- **KVCache.append signature shape `"B n_heads 1 head_dim"`** — too specific. The first forward call (prompt processing) appends T_prompt tokens at once, not 1. Generalized to `"B n_heads T_new head_dim"`. I'd over-fit to the steady-state per-step inference pattern.
- **KVCache.get annotation inconsistency**: `"T_cached"` for K, `"T_cached+1"` for V. K and V grow together; should match. (jaxtyping doesn't evaluate `+1` arithmetically, so the inconsistency is documentation-only — but still misleading.)
- **Slice typo carried over from stage 11**: `ids[:, -self.T_max]` (integer index, no colon) — errors when `ids` is short. Should be `ids[:, -self.T_max:]` (slice). Surfaced when benchmarking the naive path. Worked silently in stage 11 because prompts were short enough that the off-by-one didn't matter. Subtle.
- **MPS silent garbage on out-of-range `nn.Embedding` indices**: expected `IndexError` (which CPU/CUDA give); got cryptic shape mismatch in `masked_fill` several stack frames downstream. Real backend gotcha. The true cause (`pos_emb` lookup past row T_max) was masked by MPS returning garbage instead of raising; the garbage propagated through several layers before something *else* finally erred loudly with the wrong message.

## Why this works
- **Q vs K/V asymmetry is structural, not contingent**. At each generation step, Q is "what the current token is asking" (a function of the current new hidden state); K, V are "what each past token offers" (functions of past hidden states, which are fixed). Past tokens' offerings don't change; the current token's question is new every step. Therefore only K and V are cacheable.
- **Mask slice formula** in absolute coordinates: query at sorted position `i` corresponds to global position `T_cached_before + i`. Causal: query at global position `p` may attend to keys at positions `≤ p`. The slice `self.mask[T_cached_before : T_cached_before + T_new, : T_total]` correctly encodes "row i can attend to columns [0, T_cached_before + i]" via the lower-triangular structure of the precomputed mask.
- **T_max ceiling for absolute pos embedding + cache**: K and V vectors had `pos_emb(position_j)` summed into their inputs before being projected. The position info is baked in. To slide the cache window (drop oldest, keep last T_max), you'd be treating those K, V vectors as if at different positions — inconsistent with how they were computed. **RoPE fixes this** because position is applied to Q, K via rotation *at attention time*, not via addition before; you can re-rotate to shift positions. Hence stage 13 unblocks stage 12's ceiling.

## Open questions
- **Explicit assertion in cached generate** for `T_prompt + max_new_tokens ≤ T_max`? Currently the failure is cryptic — mask shape mismatch via MPS lenient embedding. An assertion would catch the misuse at the start with a clear message. Probably worth adding before stage 13's retrain, or punted because stage 13 lifts the ceiling anyway.
- **3.1× speedup is modest** vs theoretical O(T) ≈ 100× for 200-token generation. How much is per-step Python overhead, MPS kernel scheduling, or fixed-cost ops (LN, MLP, lm_head per step) that the cache doesn't help? Worth profiling with `torch.profiler` if I ever want to optimize further.
- **Sliding-window cache with RoPE**: the cost of re-rotating all cached K vectors when the window slides — does it dominate per-step inference? Open until I implement RoPE in stage 13 and benchmark.

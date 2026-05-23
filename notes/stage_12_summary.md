# Stage 12: KV Cache

## Summary
Implemented the KV (key/value) cache as a performance optimization for autoregressive generation. Built a new `KVCache` class in `src/cache.py` that holds per-layer K and V tensors and supports incremental `append` via `torch.cat`. Extended `MultiHeadAttention.forward` to accept and return an optional cache, with a unified mask-slice formula that works for both training (cache=None) and inference (cache populated). Threaded the cache through `Block.forward` and `GPT.forward`, with `start_pos = len(cache[0])` driving the positional embedding offset. Rewrote `GPT.generate` with a `use_cache: bool = True` flag enabling A/B comparison between the cached and naive paths. Verified correctness via `torch.equal(out_cached, out_naive) == True` for `T_prompt + max_new_tokens ≤ T_max`. Measured **~3.1× wall-clock speedup** on MPS for 200-token generation. The stage also surfaced a real architectural ceiling — the cached path cannot generate past T_max because absolute positional embeddings have no entries beyond their precomputed range. This motivates stage 13's RoPE, which removes the ceiling.

## The math

**Complexity per generation step** (attention dominates; n_layers factor scales linearly):

| | Per step | Over T generated tokens |
|---|---|---|
| Naive (no cache) | O(T²) attention | **O(T³)** |
| KV cache | O(T) attention | **O(T²)** |

The naive path recomputes Q, K, V for the *full* context every step (T² attention each step, T steps → T³ total). The cached path computes Q, K, V only for the *new* token, then attention is `q_new · K^T` against the full cached K — a single (1, T) dot product, O(T) per step, T² total.

**The Q vs K/V asymmetry — the load-bearing insight**:

- At step T+1, `q_{T+1}` is a NEW computation because it depends on the new token's hidden state. No past forward involved `q_{T+1}`; nothing about it can be reused.
- At step T+1, `k_j` and `v_j` for `j ∈ [0, T]` were computed at past forwards and are **functions only of past hidden states, which are fixed**. They're stable across steps → cacheable.
- This is why it's "KV cache" and not "QKV cache" or "scores cache." Q is fresh every step; only K and V can be saved.

**Attention with cache** (step T+1, new token):

```
q_new : (B, n_heads, 1, head_dim)              ← computed fresh
k_new : (B, n_heads, 1, head_dim)              ← computed fresh, appended to cache
v_new : (B, n_heads, 1, head_dim)              ← computed fresh, appended to cache
K_full: (B, n_heads, T+1, head_dim)            ← cache.K after append
V_full: (B, n_heads, T+1, head_dim)            ← cache.V after append

scores: q_new @ K_full.T → (B, n_heads, 1, T+1)
mask  : self.mask[T : T+1, : T+1]              ← row T of causal mask, [0..T] columns
attn  : softmax(scores) → (B, n_heads, 1, T+1)
out   : attn @ V_full → (B, n_heads, 1, head_dim)
```

**Mask-slice unification**. The general formula `self.mask[T_cached_before : T_cached_before + T_new, : T_total]` covers both modes:

- **Training (cache=None)**: `T_cached_before = 0`, `T_new = T`, `T_total = T`. Slice reduces to `self.mask[:T, :T]` — standard lower-triangular causal mask, same as pre-cache code.
- **Inference (cache populated)**: `T_cached_before = len(cache)` (before append), `T_new = 1` (typically), `T_total = T_cached_before + 1`. Slice is a single row `[T_cached_before : T_cached_before+1, :T_total]` — all True because at position T_cached_before, the model can attend to all positions `[0, T_cached_before]`.
- **First prompt forward** (cache initially empty, then appended): `T_cached_before = 0`, `T_new = T_prompt`, `T_total = T_prompt`. Same shape as training-mode forward; cache is populated as side effect.

One formula, three cases.

**Why no cache during training**:

1. **Structural**: training computes loss at every position simultaneously (`F.cross_entropy(logits.view(-1, V), targets.view(-1))`). The cache only produces output for the *new* position; past positions' outputs were consumed at past steps. To compute training loss with a cache you'd need to retain all past outputs, defeating the cache's purpose.
2. **Parallelism**: training processes all T positions in *one parallel forward* via batched matmuls. Cache forces a *sequential* T-step pattern — same work, but pipelined sequentially → giant GPU underutilization. Training's autoregressive dependency is masked by the causal mask in a single T×T matmul; inference's dependency is real (each token depends on prior sampled tokens), so the cache is the only optimization available.

## The code
- `src/cache.py` — new file. `KVCache` class with `K, V` attributes (initially None), `append(k_new, v_new)` (in-place cat along dim=-2), `__len__()` (returns `K.shape[-2]` or 0), `get()` (returns `(K, V)` tuple).
- `src/attention.py` — `MultiHeadAttention.forward` extended:
  - New signature: `forward(self, x, cache: KVCache | None = None) -> tuple[Tensor, KVCache | None]`.
  - Computes Q/K/V from input as before; if cache is provided, records `T_cached_before = len(cache)`, calls `cache.append(k, v)`, retrieves `K_full, V_full = cache.get()`.
  - Mask slice generalized to `self.mask[T_cached_before : T_total, : T_total]`.
  - Returns `(output, cache)`.
- `src/model.py`:
  - `Block.forward` accepts `cache` arg, threads to attention, returns `(x, cache)`. Mechanical change.
  - `GPT.forward` accepts `cache: list[KVCache] | None = None`. Computes `start_pos = 0 if cache is None else len(cache[0])`. `positions = torch.arange(start_pos, start_pos + T)`. Iterates blocks with per-layer cache.
  - `GPT.generate` rewritten with `use_cache: bool = True`. Cached path: build per-layer cache list, feed full prompt on step 0, single new token thereafter. Naive path: feed `ids[:, -self.T_max:]` every step (existing truncation behavior). Sampling block shared between paths.

No new tests added. Stage 12's integration test is the equivalence assertion `torch.equal(out_cached, out_naive) == True` plus the benchmark wall-clock comparison; both run cleanly in an inline script during the audit.

## Design choices and why

- **Always-return-tuple signature for attention/Block** (`return output, cache`). Consistent API; no mode-branching at the call site (no `"is this training or inference?"` logic in the caller). Training callers ignore the cache via `out, _ = attn(x)`. The verbosity cost is small; the clarity gain is significant. Alternative considered: conditional return type (`Tensor | tuple[Tensor, Cache]`) — rejected because it makes type checking and reasoning harder.

- **Per-layer cache as `list[KVCache]`** (one entry per layer), not a single centralized cache object. Layer-local state; `GPT.forward` just indexes by block number; `Block.forward` doesn't need to know it's "layer k" of n. Simpler reasoning.

- **Cache appends via `torch.cat`** along `dim=-2` (the sequence dim). O(T) memcpy per step but the constant is small (the cached tensor's total size is `B · n_heads · T · head_dim` floats). Production inference systems (vLLM, llama.cpp) preallocate the maximum-length buffer and write into slots in place — saves the repeated allocations. For our scale (T_max=256), the overhead is invisible against the actual matmul costs.

- **`start_pos = len(cache[0])`** in `GPT.forward`. Derived from cache state — no separate position tracker. All layers' caches grow together, so layer 0 is as good as any. If all layers' caches are out of sync, that's a bug elsewhere.

- **Mask slice unification** (`self.mask[T_cached_before : T_total, : T_total]`): one expression covers training, prompt-processing first-call inference, and per-step single-token inference. Avoids three separate code paths for the mask logic.

- **`use_cache: bool = True` flag in `generate`**. Enables direct A/B comparison without code duplication. Critical for benchmarking: cached output must be *bit-identical* to naive output (KV cache is a pure optimization, no numerical drift). The flag is the cleanest way to verify this property.

- **No automatic truncation in the cached path**. The cache grows unboundedly; the only constraint is the positional embedding's T_max rows. This is a *deliberate non-feature*: any truncation strategy for absolute pos embeddings is incorrect (cached K/V have positions baked in; you can't slide the window without shifting the implicit position info). The honest answer is "T_max is a hard limit; reduce max_new_tokens or use RoPE." RoPE is stage 13.

- **Naive path keeps `ids[:, -self.T_max:]` truncation**. This is the *original* truncation from stage 11 — it works because each forward is independent (positions restart from 0 each call), so the model never sees positions ≥ T_max. The cost: model forgets earliest tokens after T_max. The benefit: generation can run indefinitely. The cached path can't do this trick — the cache preserves absolute positions.

## Errors and corrections

- **"We cache Q·K^T values"** — initial conceptual confusion. Wrong: Q changes every step (depends on the current new token's hidden state), so no past computation involves the current Q; nothing about Q or Q·K^T from prior steps is reusable. **Correction**: we cache K and V — they depend only on past tokens' hidden states, which are stable across steps. The name "KV cache" is literal.

- **`T` undefined in attention's reshape** (`out.reshape(B, T, self.d_model)`): renamed `T` to `T_new` throughout the function but missed the reshape line. NameError at runtime. Trivial typo fix.

- **Tuple not unpacked in `Block.forward`**. Wrote `attn_out = self.attn(self.ln1(x), cache)` — `attn_out` was the whole `(output, cache)` tuple. Then `self.dropout1(attn_out)` errored on a tuple. Fix: unpack `attn_out, cache = self.attn(...)`.

- **`cache` arg had no default in Block.forward**, making it required. Added `= None` for parity with attention and convenient training-mode calls.

- **Smoke test in `attention.py`** not updated for tuple return: `out = mha(x)` followed by `out.shape` errored because `out` was the tuple. Fix: `out, _ = mha(x)`. Easy to miss — the smoke test silently broke when the signature changed.

- **Annotation inconsistency in `KVCache.get`**: wrote `"T_cached"` for the first return tensor and `"T_cached+1"` for the second. Both K and V grow together; should both be `"T_cached"`. Cosmetic but misleading. (jaxtyping doesn't evaluate the `+1` arithmetically — those strings are documentation, not enforced — but the inconsistency would confuse a reader.)

- **`append` signature shape `"B n_heads 1 head_dim"`**: too specific. The append is called both during the first prompt forward (where T_new = T_prompt > 1) and during per-step inference (where T_new = 1). Generalized to `"B n_heads T_new head_dim"`.

- **Stage-11 generate truncation bug carried over**: `ids[:, -self.T_max]` (integer index) instead of `ids[:, -self.T_max:]` (slice). Worked silently in stage 11's testing because most prompts were shorter than T_max — the bug only manifested on long sequences. Surfaced during the cache-vs-naive benchmark for the naive path. Missing colon.

- **T_max overflow in cached generation**: cache grows past T_max → `pos_emb` looks up out-of-range index → silently returns garbage on MPS (CPU would IndexError immediately) → garbage propagates → mask slice clamps to 256 cols while scores are 257 cols → `masked_fill` errors loudly with shape mismatch deep in attention. The naive path doesn't hit this because its own truncation (`ids[:, -T_max:]`) keeps the input at most T_max tokens, so positions always stay in range. **Lesson**: MPS's lenient handling of out-of-range embedding indices can turn a clean IndexError into a mysterious shape-mismatch error layers downstream. Worth knowing as a backend gotcha.

- **MPS-specific debugging**: the error surface for KV cache + T_max overflow was confusing because the actual problem (out-of-range position embedding) was masked by MPS returning garbage instead of raising. On CUDA or CPU, the error would have pointed at the embedding line directly with a clear "index out of bounds" message. On MPS, you trace down through 3+ stack frames to a shape mismatch that has nothing obvious to do with positional embedding.

- **Slogan-vs-mechanism, recurrence**: the framing "we cache the attention scores" was the canonical slogan-replaces-mechanism error for this stage. The mechanism is K and V; the slogan is "the cache makes attention faster" which is true but doesn't pinpoint *what* is cached. Recurring pattern across stages — worth flagging in PROGRESS.

## Self-quiz

1. **Complexity asymmetry**. Why is naive autoregressive generation O(T³) but training is O(T²) for the same total token count? KV cache restores inference to O(T²). What's the relationship between the cached-inference complexity and the training complexity, and why does that relationship hold?

2. **Q vs K/V asymmetry**. Articulate precisely why we cache K and V but never Q. Specifically: at step T+1, which of `q_{T+1}`, `k_{T+1}`, `v_{T+1}`, `q_j` (j<T+1), `k_j` (j<T+1) can be reused from prior computations? Why?

3. **The unified mask formula** `self.mask[T_cached_before : T_cached_before + T_new, : T_total]`. Plug in the three cases: (a) training (cache=None), (b) first prompt forward with cache, (c) single-token inference step. Verify each reduces to the correct mask shape.

4. **Why `start_pos = len(cache[0])`**. What invariant of the per-layer caches makes layer 0's length authoritative for the global position offset? What would break if the per-layer caches diverged in length?

5. **The "always return tuple" decision**. Trade-offs vs. conditional-return-type vs. two-separate-methods. Why is "always return tuple" preferred for this codebase despite the minor verbosity at training call sites?

6. **T_max ceiling and the naive truncation trick**. Why does the naive (no-cache) path generate past T_max successfully, while the cached path cannot? What does the naive path silently sacrifice in exchange? Why is this trick impossible to replicate cleanly with cache + absolute positional embeddings?

7. **Predict the wall-clock speedup**. For (a) 50-token generation, (b) 200-token generation, (c) 1000-token generation (hypothetically, if T_max were larger), rank the speedups in order. Why isn't the speedup constant?

8. **The MPS silent-garbage gotcha**. What was the chain of events from "cache grows past T_max" to "RuntimeError in masked_fill"? Why is this error surface confusing? On which backend (CPU/CUDA/MPS) would this manifest with the clearest error message?

## What this enables

- **Stage 13 (RoPE)**. Rotary Position Embedding encodes position via a rotation applied to Q and K at attention time, not via an added embedding before attention. The position information is *portable*: you can re-rotate K vectors to give them new effective positions. This makes sliding-window cache cleanly possible — when the cache reaches T_max, drop the oldest entries, re-rotate the remaining ones to the appropriate positions, append new K/V at the appropriate offset. The T_max ceiling encountered in stage 12 lifts naturally. RoPE is the architectural prerequisite for long-context production LLMs (GPT-3 onward, LLaMA, Mistral, Gemini, etc.).

- **Stage 14 (SwiGLU)**. Pure architectural change inside MLP; doesn't interact with cache. Same training pipeline + RoPE-extended cache work as before.

- **Stage 15 (GQA)**. Grouped Query Attention: fewer K/V heads than Q heads, so cache memory shrinks proportionally. GQA + KV cache is the standard inference-optimization recipe for production LLMs. The cache infrastructure built here is the prerequisite.

- **Production inference systems**. vLLM's PagedAttention, llama.cpp's KV cache management, ExLlama's quantized caches — all built around the conceptual machinery established here. Stage 12's KVCache class is the toy version of what these production systems implement at scale (preallocated, paged, quantized, batched across many requests).

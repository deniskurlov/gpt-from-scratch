# Progress

## Where we are
- **Stage:** 12 — KV Cache. **DONE.**
- **Sub-step:** Stage 13 not yet started (RoPE — replace absolute pos embeddings with rotary; retrain).
- **Stage 11 status:** complete in code (`src/sample.py` with greedy → temperature → top-k → top-p + argparse CLI) but no formal `/stage-done 11` summary. Acceptable gap; can backfill with `/stage-done 11` if desired.
- **Last completed:** Stage 12's KV cache. New `src/cache.py` with `KVCache` class (append/`__len__`/get). `MultiHeadAttention`, `Block`, `GPT.forward` extended to thread a per-layer cache list; unified mask slice `self.mask[T_cached_before : T_total, : T_total]` covers training + inference. `GPT.generate` now has `use_cache: bool = True` flag enabling A/B comparison. Verified: `torch.equal(out_cached, out_naive) == True` for `T_prompt + max_new_tokens ≤ T_max`. Measured **3.1× wall-clock speedup** on MPS for 200-token generation. Surfaced T_max ceiling: cached path errors past T_max because absolute pos embeddings have no entries beyond their precomputed rows; naive path silently truncates via `ids[:, -T_max:]` so doesn't hit it. Motivates stage 13 RoPE. Summary at `notes/stage_12_summary.md`; T_max-ceiling note at `notes/stage_12_t_max_ceiling.md`.

## Resume here
Begin Stage 13: RoPE (Rotary Position Embedding). Per README: "Replace absolute pos embeddings with RoPE; retrain." First conceptual step before code: derive RoPE's defining property — for queries and keys at positions `(m, n)`, the rotated inner product `<R_m·q, R_n·k>` should depend only on the relative position `m - n`, not on absolute m or n. Construction: pair up the head_dim into 2D subspaces, rotate each pair by an angle `m·θ_i` where `θ_i = base^(-2i/head_dim)` (base=10000, Su et al. 2021). Architectural change: remove `LearnedPositionalEmbedding` usage in `GPT`; apply RoPE inside `MultiHeadAttention` to Q and K (not V) immediately after the QKV projection. Retrain the model with RoPE — current `checkpoints/model.pt` weights are for absolute pos embedding and won't transfer. Sliding-window KV cache becomes naturally possible with RoPE; that's the stage-12 ceiling lifting.

## Open conceptual debts
- **T_max ceiling for cached generation**: real architectural limit of absolute pos embeddings. Stage 13 RoPE resolves it. The naive path's `ids[:, -T_max:]` truncation is a "lie" about position that only works because each forward is independent of past forwards; the cached path can't tell the lie because cached K, V have positions baked in.
- **Slogan-vs-mechanism, ongoing pattern across stages**. Stage 12 surfaced "we cache attention scores" (wrong — Q changes every step; only K, V are cacheable). Stage 13 watch-items: "RoPE encodes relative positions" (precise: rotates Q and K so the inner product is a function of m-n by construction — not a "relative position embedding" added to the input), "rotary is just sinusoidal" (related motif, different operation — Vaswani adds sinusoidal positional embeddings; RoPE multiplies via 2D rotations).
- **MPS gotcha — silent garbage on out-of-range `nn.Embedding`**. Real backend behavior. CPU/CUDA raise IndexError; MPS returns garbage that propagates several layers before failing with a cryptic shape mismatch. Worth remembering. Stage 13 may surface more MPS-specific behavior with RoPE's paired-real-valued rotation ops.
- **No explicit T_max assertion in cached `generate`**. Could be added; stage 13 lifts the ceiling anyway, so deprioritized.
- **Predict-then-check at architectural granularity**. Stage 12 had clear predict-points (correctness via `torch.equal`; mask formula reductions). Stage 13 analog: predict the rotation formula's effect on `Q·K^T` before coding; predict sliding-window cache behavior past T_max.
- **`bias=True` (attention QKV) vs `bias=False` (MLP, lm_head)** persistent cosmetic inconsistency. Still not urgent.
- **No `stage_11_summary.md`** despite stage 11 being functionally complete. Documentation gap. Optional to backfill.
- **Early-stopping / best-eval checkpoint** not implemented. Stage 13's retrain is a natural place to add it (longer runs are more likely to want it).
- **Per-param-group weight decay** (LN γ/β + biases excluded) not implemented. Standard nanoGPT recipe; trivial to add when needed.
- **Mechanistic interpretability framing** still un-engaged. Stage 12 (KV cache) and stage 13 (RoPE) are both directly relevant to "QK / OV circuits" (Elhage et al.). Reading "A Mathematical Framework for Transformer Circuits" before stage 13 would pay off twice over.
- **SwiGLU at stage 14** — forward reference still queued at `notes/stage_14_swiglu_reference.md`.
- **3.1× speedup is modest** vs theoretical O(T) ≈ 100× for 200-token gen. Most likely per-step Python loop overhead + MPS kernel-launch fixed cost + small-batch GPU underutilization. Worth profiling with `torch.profiler` if I ever want to optimize.

## Code state

**Source (`src/`)**
- `__init__.py` — package marker.
- `data.py` — Stage 1: `Tokenizer`, `TokenizedDataset.get_batch`. Tested.
- `embedding.py` — Stage 2: `TokenEmbedding`, `LearnedPositionalEmbedding`. Tested. (Stage 13 will likely deprecate `LearnedPositionalEmbedding`.)
- `attention.py` — Stages 3-4 + Stage 12. `Attention` (unchanged single-head). `MultiHeadAttention.forward(x, cache=None) → (output, cache)`. Mask slice unified across training/inference. Tested original behavior; cache behavior integration-tested via inline equivalence + benchmark.
- `normalization.py` — Stage 5: `LayerNormalization`. Tested.
- `mlp.py` — Stage 6: `MLP`. Tested.
- `cache.py` — **Stage 12 new file**: `KVCache` class. Append-via-`torch.cat`, in-place mutation. Inline smoke-tested only; no formal unit tests.
- `model.py` — Stages 7-8 + 11-12. `Block.forward(x, cache=None) → (x, cache)`. `GPT(forward + generate)` with cache threading, `start_pos` handling, `use_cache: bool = True` flag. Inline integration-tested.
- `train.py` — Stages 9-10. Unchanged through 11-12 (cache invisible to training). AdamW + warmup + cosine, eval every 200 steps, resumable checkpoint. Final val loss 1.69.
- `sample.py` — Stage 11. argparse CLI: `--prompt`, `--max-new-tokens`, `--temperature`, `--top-k`, `--top-p`, `--seed`. Loads checkpoint via `ckpt['config']['model']` reconstruction. Doesn't yet expose `--no-cache` for CLI A/B comparison; could add.
- `config.py` — `GPTConfig`, `TrainConfig` dataclasses (Stage 11 refactor). Used by both `train.py` and `sample.py`.

**Tests (`tests/`)** — 66 passing total; unchanged since stage 10.
- No unit tests for `cache.py`, `train.py`, `sample.py`.

**Other**
- `data/input.txt` — TinyShakespeare, 1.1MB, gitignored.
- `checkpoints/model.pt` — ~15MB resumable format (model + optimizer + scheduler + config). Val loss 1.69 at training end.
- `.gitignore` — includes `data/`, `checkpoints/`, `.venv/`, etc.
- `.venv/` — Python 3.13.5, MPS-enabled.
- `derivations/` — directory exists, empty. All math inline in conversation history.
- `notes/` — `PROGRESS.md` + 10 stage summaries (1-10, 12 — no 11) + ~13 working notes + 1 forward reference. New since last checkpoint: `stage_12_summary.md`, `stage_12_t_max_ceiling.md`.
- `.claude/commands/` — `stage-done.md`, `checkpoint.md`, `note.md`.

**Does not exist yet**: RoPE, SwiGLU, GQA. RoPE-retrained checkpoint. Sliding-window cache. Stage 13 onward open.

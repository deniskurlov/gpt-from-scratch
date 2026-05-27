# Progress

## Where we are
- **Stage:** 13 — RoPE + Sliding-Window KV Cache. **DONE.** Now in the "professionalization" arc on top of stage 13 (not a new numbered stage).
- **Sub-step:** Class-level design comments on `Block`, `MultiHeadAttention`, `RoPE`, `KVCache` — last remaining item of the 7-item professionalization plan.
- **Last completed:** Best/latest checkpoint refactor + logger infrastructure in `src/train.py` (this session). `SinusoidalPositionalEmbedding` added to `src/embedding.py` as a standalone reference class (commit `d70243e`). Three-way pos-emb ablation formally cancelled — won't-do, reasoning preserved below.

## Resume here
Add class-level design comments — 3-5 line bullet-point comments at the top of each of `Block`, `MultiHeadAttention`, `RoPE`, `KVCache` explaining design choices (separate Q/K/V projections, causal mask on the fly, sliding-window via `total_appended` + `window_start`, RoPE applied to Q/K not V, etc.). ~20 lines total across 4 classes. Pattern: terse bullet points, design rationale not implementation. This closes the professionalization arc; after it, the optional stages (14 SwiGLU / 15 GQA) or stage 11 summary backfill are the next plausible paths.

## Open conceptual debts
- **Three-way pos-encoding ablation: cancelled.** Decided in this session that the code-bloat cost (branching across `GPT`, `Block`, `MHA`, `GPTConfig`, plumbing for tagged-union–style dispatch) outweighs the experimental yield. Denis has already personally observed RoPE outperforming learned (~1.49 vs ~1.69 at 5K steps) qualitatively. `SinusoidalPositionalEmbedding` and `LearnedPositionalEmbedding` are preserved as legacy reference implementations (docstrings added). Won't-do, with reasoning. Move on.
- **`stage_11_summary.md` still missing** despite stage 11 being complete in code. Documentation gap, accepted.
- **No note documenting today's sampling experiments** — the U-curve discovery (20K-step val_loss U-curving back up to 1.68 from min ~1.46 at step 2600), the "Thou shall not pass" OOD prompt experiment, and the "The CEO announced quarterly earnings" snap-back-in-one-comma experiment. Pedagogically rich observations; should be written up as `notes/stage_13_sampling_behavior.md` or similar before they fade.
- **Class-level design comments** not yet added (the remaining professionalization item).
- **Slogan-vs-mechanism pattern**, ongoing. Today's surface: "the cache window has shifted past the prefix" was a sloppy mechanistic story for the "NNE:" sample artifact — sliding window moves one token at a time; the model's response migrates gradually with it. Denis correctly pushed back. Pattern to watch for: clean mechanistic stories that *sound* right but don't survive scrutiny.
- **Per-param-group weight decay** (LN γ/β + biases excluded from decay) not implemented. Standard nanoGPT recipe; trivial to add. Deferred indefinitely.
- **Env-var overrides for TrainConfig / GPTConfig** — deferred this session as premature; revisit if ablation sweeps become common.
- **`max_wallclock_seconds` budget + LR-decay-by-elapsed-time** — deferred this session; only relevant for time-bounded runs.
- **Mechanistic-interpretability framing** — Elhage et al. "A Mathematical Framework for Transformer Circuits" still un-read. Increasingly relevant; stages 12-13 (KV cache + RoPE) and the U-curve / snap-back observations all touch its territory.
- **Sliding-window position-interpolation** (YaRN, NTK-RoPE) — forward reference for any future long-context work.
- **Production KV cache features** (PagedAttention, quantized cache, prefix caching, speculative decoding) — forward references; all build on the basic sliding-window infrastructure now in place.
- **Configs-as-pointers anti-pattern** — surfaced this session. Denis initially typed `pos_emb_type: RoPE | SinusoidalPositionalEmbedding | LearnedPositionalEmbedding = RoPE` (class references in config). Resolved by switching to `Literal[...]` strings. Worth remembering: configs are values, not pointers.

## Code state

**Source (`src/`)**
- `__init__.py` — package marker.
- `data.py` — Stage 1: `Tokenizer`, `TokenizedDataset.get_batch`. Tested (7 tests).
- `embedding.py` — Stages 2 + 13 + this session. `TokenEmbedding`, `LearnedPositionalEmbedding` (now marked `"""Legacy reference; not used by the current GPT (which uses RoPE)."""`), `SinusoidalPositionalEmbedding` (this session, also marked legacy), `RoPE` (active). RoPE smoke-tested via inner-product invariance in `__main__`. Sinusoidal class has no smoke test.
- `attention.py` — Stages 3-4 + 12-13. `Attention` (unchanged); `MultiHeadAttention` rotates Q/K via `self.rope`, computes causal mask on the fly, sliding-window-aware cache integration. **Note**: at the pre-ablation state — `rope_base: float` constructor parameter, instantiates own RoPE internally. (The session attempted a refactor to `rope: RoPE | None` passed-in pattern, then reverted; not committed.)
- `normalization.py` — Stage 5: `LayerNormalization`. Tested (3 tests).
- `mlp.py` — Stage 6: GELU-MLP. Tested (12 tests). Stage 14 (optional) would replace with SwiGLU.
- `cache.py` — Stages 12-13. `KVCache` with `max_size` cap, `total_appended` counter, `window_start` property, sliding-window logic in `append`.
- `model.py` — Stages 7-8 + 11-13. `Block.forward`, `GPT.forward` (no pos_emb addition — RoPE inside MHA), `GPT.generate`, `GPT.stream` (B=1 guard). All GPT constructor params required (no defaults). Initial loss matches log V ≈ 4.17.
- `train.py` — Stages 9-10 + this session. **Major refactor this session**: `save_checkpoint` helper (atomic tmp+rename, fsync, encapsulated mkdir, accepts `val_loss` metadata). `best.pt` + `latest.pt` separation; post-loop best-check captures truly-final state. Per-run logfile via local `log()` closure: `logs/<timestamp>_<short-uuid>.txt`. Logs git commit + uncommitted diff at run start (replaces source-code dump). Top + bottom run timestamps for duration. All `print(...)` converted to `log(...)`.
- `sample.py` — Stage 11 + 13. Argparse CLI for prompt, temperature, top_k, top_p, max_new_tokens, use_cache (via `BooleanOptionalAction`), seed. Per-token streaming via `model.stream`. Currently loads `checkpoints/model.pt` — **may break** with the new `best.pt`/`latest.pt` naming; not verified this session. Worth checking before next sampling run.
- `config.py` — Stage 11 + 13. `GPTConfig` (V, T_max, d_model, n_heads, n_layers, `rope_base: float = 10_000.0`, d_ff, dropout). `TrainConfig` (total_steps=5000, all hyperparams from stages 9-10). No `pos_emb_type` field — ablation cancelled.

**Tests (`tests/`)** — 66 passing total; unchanged since stage 10. No unit tests for RoPE, KVCache sliding-window, on-the-fly mask, `save_checkpoint`, the logger closure, or `SinusoidalPositionalEmbedding`. All integration-tested via smoke runs + training + sample generation.

**Other**
- `data/input.txt` — TinyShakespeare, 1.1MB, gitignored.
- `checkpoints/` — `best.pt` + `latest.pt` from the most recent 20K-step run (the U-curve run). Best is at step ~2600 with val_loss ~1.46; latest is at step 19999 with val_loss ~1.68.
- `logs/` — per-run logfiles accumulate here. **Should be added to `.gitignore`** if not already (verify on next session).
- `.gitignore` — `data/`, `checkpoints/`, `.venv/`. Modified status not yet committed last I checked (was `M .gitignore` at session start; may still need `logs/`).
- `.venv/` — Python 3.13.5, MPS-enabled.
- `derivations/` — exists, empty. Math inline in conversation.
- `notes/` — `PROGRESS.md` + stage summaries (1-10, 12, 13 — no 11) + working notes + 1 forward reference (`stage_14_swiglu_reference.md`). No new notes from this session yet (sampling-experiments note pending).
- `references/train_gpt_mlx.py` — modded-nanoGPT MLX speedrun adaptation, referenced this session as professionalization source.
- `.claude/commands/` — `stage-done.md`, `checkpoint.md`, `note.md`.

**Does not exist yet**: SwiGLU MLP, GQA, three-way pos-encoding ablation (cancelled), RoPE position interpolation (YaRN), early-stopping-style training termination, per-param-group weight decay, post-training (SFT/DPO) pipeline. Class-level design comments not yet added.

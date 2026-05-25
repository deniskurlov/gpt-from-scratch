# Progress

## Where we are
- **Stage:** 13 — RoPE + Sliding-Window KV Cache. **DONE.**
- **Sub-step:** Stage 14 not yet started (optional — SwiGLU MLP). Alternative continuations: three-way pos-encoding ablation (Learned / Sinusoidal / RoPE), or GQA (stage 15).
- **Stage 11 status:** complete in code (`src/sample.py` with argparse CLI for sampling), no formal `/stage-done 11` summary. Documentation gap, accepted.
- **Last completed:** Stage 13's RoPE + sliding-window cache. `RoPE(nn.Module)` in `src/embedding.py` with on-the-fly cos/sin (no T_max ceiling). `MultiHeadAttention` rotates Q and K (V untouched); causal mask computed dynamically via `j ≤ T_total - T_new + i`. `KVCache` gained `max_size`, `total_appended`, `window_start`; oldest entries dropped on overflow. `GPT.forward` no longer adds positional embeddings (RoPE handles position inside attention). `GPT.generate` creates per-layer caches with `max_size=self.T_max`, optional `verbose=True` debug print. `rope_base` plumbed through `GPTConfig` → `GPT` → `Block` → `MultiHeadAttention` → `RoPE`. Retrained on TinyShakespeare; training trajectory matched stage-10's absolute-pos-emb curve within ±0.05 final loss. Long-context generation past T_max=256 works without crashes; cache stays bounded at T_max. Summary at `notes/stage_13_summary.md`; mechanism note at `notes/stage_13_rope_mechanism.md`.

## Resume here
Three plausible next paths, all optional per the README:
1. **Stage 14 (SwiGLU)** — replace GELU-MLP with SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))`. Three projections instead of two; ~50% more params. Forward reference at `notes/stage_14_swiglu_reference.md`. Architectural change inside `src/mlp.py`; doesn't interact with RoPE or the cache. Brief retrain after.
2. **Three-way pos-encoding ablation** — add a `pos_emb_type: Literal["learned", "sinusoidal", "rope"]` field to `GPTConfig` and dispatch in `GPT.__init__`. `LearnedPositionalEmbedding` import was preserved as a comment in `model.py` for exactly this. Direct experimental comparison; ~30 lines + 3 retrains. Pedagogically rich.
3. **Stage 15 (GQA)** — Grouped Query Attention. K/V cache memory reduced by `n_heads / n_kv_heads`; pairs well with sliding-window.

Denis's preference earlier was for the ablation (path 2). Stage 14/15 are README-canonical; (2) is a more-direct extension of stage 13 that demonstrates the architectural progression experimentally.

## Open conceptual debts
- **Three-way pos-encoding ablation** still not done. Stage 13 specifically preserved the `LearnedPositionalEmbedding` import-as-comment for this purpose. Direct experiment would resolve "do we actually see RoPE gains at this scale" (predicted: no, but architecturally only RoPE generates past T_max). Highest-value loose end.
- **Slogan-vs-mechanism pattern**, ongoing. Stage 13 surfaced: "we cache attention scores" (wrong — Q changes per step; only K, V are stable), "factor of 2 in the RoPE exponent" (stacked twice; recurring slip), "RoPE encodes relative positions" (precise: rotates Q and K so the inner product becomes a function of m-n via the abelian group law). Stage 14 watch-items if pursued: "gating gates information" (precise: element-wise multiplication of sigmoid branch with linear branch — *selective passing* of feature directions), "SwiGLU is twice as expressive" (loose; quality gain modest, mostly architectural elegance).
- **`type=bool` argparse footgun**, recurrent. `--use-cache False` parses as True because `bool("False") = True`. Bit twice (stages 11 and 13). Canonical fix: `action=argparse.BooleanOptionalAction`. Worth fixing in `src/sample.py` whenever next CLI work happens.
- **MPS silent-garbage on out-of-range indices**, recurrent backend gotcha. Stage 12 (out-of-range nn.Embedding for pos_emb past T_max) and stage 13 (precomputed cos_cached slicing past buffer size) both surfaced this as cryptic downstream shape errors. Structural fix in both cases: don't precompute fixed-size; compute on the fly. Keep on the MPS gotcha list.
- **`bias=True` (attention QKV) vs `bias=False` (MLP, lm_head)** — persistent cosmetic inconsistency, never urgent. Could be made uniform during any future attention refactor.
- **Predict-then-check at architectural granularity** — improved across stages 10-13. Stage 13's three predictions (training loss tracks stage 10, generation quality past T_max not significantly better, RoPE inner-product invariance) all held. Pattern to maintain.
- **Mechanistic-interpretability framing** — Elhage et al. "A Mathematical Framework for Transformer Circuits" still un-read. Increasingly relevant: stages 12-13 (KV cache + RoPE) both directly relate to the QK/OV circuits decomposition. Best read before any deeper interpretability or ablation work.
- **No `stage_11_summary.md`** despite stage 11 being complete in code. Documentation gap.
- **Early-stopping / best-eval checkpoint** not implemented. Stage 13's retrain was short (5000 steps) so not critical. Relevant if doing longer runs or ablation sweeps.
- **Per-param-group weight decay** (LN γ/β + biases excluded from decay) not implemented. Standard nanoGPT recipe; trivial to add.
- **Sliding-window position-interpolation** (YaRN, NTK-RoPE) — extend effective context past training T_max without retraining. Not implemented; forward reference for any future long-context work.
- **Production KV cache features** (PagedAttention, quantized cache, prefix caching, speculative decoding) — forward references; all build on the basic sliding-window infrastructure we now have.

## Code state

**Source (`src/`)**
- `__init__.py` — package marker.
- `data.py` — Stage 1: `Tokenizer`, `TokenizedDataset.get_batch`. Tested (7 tests).
- `embedding.py` — Stages 2 + 13. `TokenEmbedding`, `LearnedPositionalEmbedding` (preserved unused for ablation), `RoPE` (stage 13, on-the-fly cos/sin via `inv_freq` buffer + `forward(x, start_pos)`). RoPE smoke-tested via inner-product invariance in `__main__`.
- `attention.py` — Stages 3-4 + 12-13. `Attention` (unchanged); `MultiHeadAttention` rotates Q/K via `self.rope`, computes causal mask on the fly, sliding-window-aware cache integration via `total_appended_before` and `cache.window_start`. Integration-tested.
- `normalization.py` — Stage 5: `LayerNormalization`. Tested (3 tests).
- `mlp.py` — Stage 6: GELU-MLP. Tested (12 tests). Stage 14 (optional) would replace with SwiGLU.
- `cache.py` — Stages 12-13. `KVCache` with `max_size` cap, `total_appended` counter, `window_start` property, sliding-window logic in `append`.
- `model.py` — Stages 7-8 + 11-13. `Block.forward`, `GPT.forward` (no pos_emb addition, just `tok_emb`), `GPT.generate` (sliding-window cache, optional verbose). All GPT constructor params now required (no defaults). Initial loss matches log V ≈ 4.17.
- `train.py` — Stages 9-10, transparent through 11-13. Passes `rope_base=cfg.model.rope_base` to GPT construction.
- `sample.py` — Stage 11 + 13. Argparse CLI for prompt, temperature, top_k, top_p, max_new_tokens, use_cache, seed. Loads checkpoint via `GPT(**ckpt['config']['model'])`. `--use-cache` flag has the type=bool footgun; would be cleaner with `BooleanOptionalAction`.
- `config.py` — Stage 11 + 13. `GPTConfig` (includes `rope_base: float = 10_000.0`), `TrainConfig` (`total_steps=5000`).

**Tests (`tests/`)** — 66 passing total; unchanged since stage 10. No unit tests for RoPE, KVCache sliding-window, or the on-the-fly mask. All integration-tested via smoke tests + training run + sample generation.

**Other**
- `data/input.txt` — TinyShakespeare, 1.1MB, gitignored.
- `checkpoints/model.pt` — fresh RoPE-retrained checkpoint from stage 13 (~15MB resumable format).
- `.gitignore` — `data/`, `checkpoints/`, `.venv/`, etc.
- `.venv/` — Python 3.13.5, MPS-enabled.
- `derivations/` — exists, empty. Math inline in conversation.
- `notes/` — `PROGRESS.md` + 12 stage summaries (1-10, 12, 13 — no 11) + 14 working notes + 1 forward reference (`stage_14_swiglu_reference.md`). New since last checkpoint: `stage_13_summary.md`, `stage_13_rope_mechanism.md`.
- `.claude/commands/` — `stage-done.md`, `checkpoint.md`, `note.md`.

**Does not exist yet**: SwiGLU, GQA, three-way positional-encoding ablation, RoPE position interpolation, early-stopping checkpoints, per-param-group weight decay, post-training (SFT/DPO) pipeline.

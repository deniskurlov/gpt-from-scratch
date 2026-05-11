# Progress

## Where we are
- **Stage:** 2 — Token + learned positional embeddings. **DONE.** Summary at `notes/stage_2_summary.md`.
- **Sub-step:** Stage 3 not yet started — scaled dot-product attention (single head, causal mask).
- **Last completed:** Stage 2 finished, all three criteria met (16 tests passing, line-by-line explanation, toy prediction by hand on `corpus="abc"` with `B=1, T=2, d_model=4, T_max=8`). `/stage-done 2` produced the summary.

## Resume here
Begin Stage 3: single-head scaled dot-product attention with a causal mask. The work consumes the `(B, T, d_model)` float32 tensor produced by stage 2 and produces a tensor of the same shape (attention is shape-preserving on its input). Three new components: Q/K/V projections via `nn.Linear`, the scaled dot-product `softmax(QK^T / √d) V`, and the causal mask that zeroes out future-position attention scores. Same protocol as stages 1-2: conceptual probe first (Q/K/V intuition, why the √d scaling, why the mask, what shapes flow), then implementation as a new `nn.Module` (likely in `src/model.py` or `src/attention.py`), tests, line-by-line walkthrough, toy prediction, `/stage-done 3`. CLAUDE.md's anti-shortcut rule applies here: **no `nn.MultiheadAttention`, no `F.scaled_dot_product_attention`** — build it from `nn.Linear`, `F.softmax`, and basic tensor ops.

## Open conceptual debts
- **`n_heads` for stage 4.** Deferred from stage 2's `d_model=128` choice. With `d_model=128`, plausible options: `n_heads=4` (`head_dim=32`) or `n_heads=8` (`head_dim=16`). Becomes live at stage 4 (multi-head attention). For stage 3 (single head), it doesn't matter — `n_heads=1` is the default.
- **Device placement strategy.** Encoded corpus lives on CPU; batches will need `.to(device)` before the model in stage 9 (training loop). Exact pattern (per-batch transfer vs preallocated buffers) is open until then.
- **MPS-fallback awareness.** `PYTORCH_ENABLE_MPS_FALLBACK=1` set in README; not yet exercised. Likely to first fire on some attention or softmax op in stage 3, or on RoPE in stage 13. Watch for fallback warnings during first stage-3 run.
- **Mechanistic-interpretability framing.** Introduced in stage 2 ("residual stream as shared communication channel, `d_model` as bandwidth") but Denis hasn't deeply engaged. Pays off in stages 7 (transformer block) and 8 (full GPT); referenced casually in this project but the formal paper to read is "A Mathematical Framework for Transformer Circuits" (Olah, Elhage et al., Anthropic). Optional; surface again if `d_model` choices or activation reasoning gets fuzzy.
- **Wall-clock training time on M4 Pro.** Untested. `d_model=128`, `n_layers=4-6`, `T=256` should train tiny-shakespeare in tens of minutes; benchmark when stage 9 lands.
- **Vectorized broadcasting fluency.** Got real practice in stage 2 (`(B, T, d_model) + (T, d_model)` via leading-dim auto-broadcast). About to be hammered in stage 3 — the causal mask broadcasts `(T, T)` against `(B, n_heads, T, T)`, and QK^T builds via `Q @ K.transpose(-1, -2)`. The broadcasting frustration Denis flagged in stage 2 hasn't gone away; jaxtyping annotations and explicit shape comments are the partial defenses.
- **jaxtyping `"L not defined"` warning.** Ignored. Cosmetic. Will keep accumulating warnings as more annotated modules are added; revisit if it becomes noise.

## Code state
- `README.md` — 15 stages listed; setup commands; M4 Pro / MPS notes. ✓
- `CLAUDE.md` — tutoring rules + session-bootstrap pointer to PROGRESS.md; documentation-command output paths. ✓
- `.gitignore` — `.venv/`, `__pycache__/`, `*.pyc`, `data/`, `.DS_Store`, `.claude/sessions/`. ✓
- `requirements.txt` — `torch==2.11.0`, `numpy==2.4.4`, `jaxtyping==0.3.9`, `pytest==9.0.3`, plus transitives. ✓
- `.venv/` — Python 3.13.5, MPS available and verified. ✓
- `data/input.txt` — tiny-shakespeare, 1,115,394 bytes, gitignored. ✓
- `src/__init__.py` — empty package marker, enables `from src.data import ...` consistently. ✓
- `src/data.py` — Stage 1: `load_corpus()`, `Tokenizer` (vocab/stoi/encode/decode/encode_to_tensor), `TokenizedDataset.get_batch(B, T) → (x, y)`. ✓
- `src/model.py` — Stage 2: `TokenEmbedding(V, d_model)`, `LearnedPositionalEmbedding(T_max, d_model)`. Both wrap `nn.Embedding`. Forward signatures jaxtyping-annotated. `__main__` smoke test composes the two and prints output shape. ✓
- `tests/conftest.py` — shared `text` and `tok` fixtures, module-scoped. ✓
- `tests/test_data.py` — 7 tests covering Stage 1 (round-trip, vocab content/size, batch shape/dtype/range, shift-by-1 invariant). All passing. ✓
- `tests/test_model.py` — 9 tests covering Stage 2 (token-embedding shape, token-embedding param count, parametrized positional-embedding shape across `T ∈ {1, 8, 32, 128, 256}`, positional-embedding param count, positional-embedding out-of-range raises `IndexError`). All passing. ✓
- `derivations/` — directory exists, empty. No stage 1-2 derivations needed; stage 3 may want one for the √d scaling argument.
- `notes/PROGRESS.md` — this file.
- `notes/stage_1_tokenization_concepts.md`, `notes/stage_1_batch_sampling.md`, `notes/stage_1_summary.md` — stage 1 artifacts.
- `notes/stage_2_embedding_concepts.md`, `notes/stage_2_embedding_modules.md`, `notes/stage_2_summary.md` — stage 2 artifacts.
- **No attention code yet.** No transformer block, no full GPT, no training loop. Stages 3-15 not begun.

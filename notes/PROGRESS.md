# Progress

## Where we are
- **Stage:** 1 — Data loading + character-level tokenization. **DONE.** Summary at `notes/stage_1_summary.md`.
- **Sub-step:** Stage 2 not yet started — token + learned positional embeddings.
- **Last completed:** Full Stage 1 done, all three criteria met (7 tests passing, line-by-line explanation, toy prediction by hand on `corpus="abcab"`). `/stage-done 1` produced the summary.

## Resume here
Begin Stage 2: token + learned positional embeddings. The work consumes a `(B, T)` int64 tensor from `TokenizedDataset.get_batch` and produces a `(B, T, d_model)` float32 tensor — the input to attention. Two new objects to define: `nn.Embedding(V, d_model)` for token IDs → vectors, and `nn.Embedding(T_max, d_model)` (or analogous) for position IDs → positional vectors. The first design discussion to have: how to choose `d_model` (one of the open conceptual debts), and how token + positional embeddings compose (sum vs concatenate; standard choice is sum, but force the justification). Probable file: a new `src/model.py` (or `src/embeddings.py`) — but discuss layout before code. Same protocol as stage 1: predict-then-check on tensor shapes/dtypes; full predict-then-check fully re-engaged for tensor ops.

## Open conceptual debts
- **`d_model` selection.** Open since the conceptual probe in stage 1. No heuristic given yet for how to pick it for char-level on tiny-shakespeare. Becomes live the moment we instantiate `nn.Embedding(V, d_model)` in stage 2. Push for a justification before settling on a number.
- **Positional embedding composition.** Standard practice is `token_emb + pos_emb` (elementwise). Why not concatenate, and what would change? Force the answer before code.
- **MPS-fallback awareness.** `PYTORCH_ENABLE_MPS_FALLBACK=1` mentioned in README; Denis hasn't yet hit a missing-op case. Open until first surprise. Likely to surface in stage 3 (attention) or stage 13 (RoPE) where ops occasionally lack MPS kernels.
- **Vectorized broadcasting fluency.** Denis used `offsets[:, None] + arange(T)[None, :]` correctly in `get_batch`, but the broadcasting pattern is new and was learned mid-stage. Will be tested again in stage 3 (causal mask construction, pairwise QK^T scoring) where the same pattern dominates. Watch for fluency, push back if shapes get muddled.
- **jaxtyping static-checker warning (`"L" not defined`).** Currently ignored. Cosmetic; harmless. Worth revisiting only if the warning multiplies to the point of becoming noise.
- **BPE mechanism articulation.** Resolved during stage 1 — Denis can now cleanly state "each merge step adds exactly one new vocab entry; user picks number of merges". Closed.

## Code state
- `README.md` — written, lists 15 stages and success criteria. ✓
- `CLAUDE.md` — tutoring rules + session-bootstrap pointer to PROGRESS.md. ✓
- `.gitignore` — `.venv/`, `__pycache__/`, `*.pyc`, `data/`, `.DS_Store`, `.claude/sessions/`. ✓
- `requirements.txt` — direct deps `torch==2.11.0`, `numpy==2.4.4`, `jaxtyping==0.3.9`, `pytest==9.0.3`, plus transitives. ✓
- `.venv/` — Python 3.13.5, MPS available and verified. ✓
- `data/input.txt` — tiny-shakespeare, 1,115,394 bytes, gitignored. ✓
- `src/data.py` — `load_corpus()`, `Tokenizer` class (`vocab`, `vocab_size`, `stoi`, `encode`, `decode`, `encode_to_tensor`), `TokenizedDataset` class (`encoded`, `get_batch(B, T) -> (x, y)`). Compiles, runs, behaves correctly. ✓
- `tests/test_data.py` — 7 pytest cases, all passing: `test_encoding_roundtrip`, `test_vocab_size`, `test_vocab`, `test_batch_size` (shape), `test_batch_type` (dtype), `test_batch_range` (both `x` and `y` in `[0, V)`), `test_shift_by_1_invariant`. ✓
- `derivations/` — directory exists, empty. No Stage 1 derivations needed yet.
- `notes/stage_1_tokenization_concepts.md` — Stage 1 conceptual probe notes with explicit error log. ✓
- `notes/stage_1_summary.md` — Stage 1 done artifact: math, code, design choices, errors, self-quiz, "what this enables". ✓
- `notes/PROGRESS.md` — this file. ✓
- **No model code yet.** No `src/model.py`, no embedding layers, no attention, no transformer block. That's stage 2 onward.

# Progress

## Where we are
- **Stage:** 3 — Scaled dot-product attention (single head, causal mask). **DONE.** Summary at `notes/stage_3_summary.md`.
- **Sub-step:** Stage 4 not yet started — multi-head attention.
- **Last completed:** Stage 3 finished with all three criteria met (21 tests passing total: 5 in `test_attention.py`, 9 in `test_model.py`, 7 in `test_data.py`; line-by-line walkthrough of `Attention.__init__` and `forward`; toy shape prediction with `B=1, T=2, d_k=d_v=d_model=2`). `/stage-done 3` produced the summary.

## Resume here
Begin Stage 4: multi-head attention. Mechanically this is a tensor-reshape on top of stage 3's `Attention` — split the `(B, T, d_model)` representation into `(B, n_heads, T, head_dim)` where `head_dim = d_model / n_heads`, run scaled dot-product attention independently per head, concatenate, project. Everything from stage 3 carries over (`√d_k` scaling, causal mask, `transpose(-1, -2)`, `register_buffer`, fused QKV) — the only structural change is one extra dimension and one extra reshape pair. Conceptual probe first: pick `n_heads` (with `d_model=128`, candidates are 4 or 8), justify; reason about why multiple heads instead of one bigger head; predict shapes through the head-reshape + attention + concat path. Then implementation as a new `MultiHeadAttention` class (likely in `src/attention.py` alongside the single-head version, or a new file). Same protocol as stages 1-3.

## Open conceptual debts
- **`n_heads` selection.** Becomes live at stage 4. With `d_model=128`, candidates: `n_heads=4` (`head_dim=32`, comfortable) or `n_heads=8` (`head_dim=16`, tight but workable). Plus the general "why multiple heads at all" question — the standard answer is "different heads can specialize in different attention patterns / circuits"; worth pushing for the substantive reasoning rather than accepting hand-wave.
- **Device placement strategy.** Encoded corpus lives on CPU; batches will need `.to(device)` before the model in stage 9. Exact pattern (per-batch transfer vs preallocated buffers) is open. Stage 3 still hasn't moved anything to MPS — the smoke test ran on CPU, tests run on CPU. The first MPS run will likely happen at stage 9 when training starts.
- **MPS-fallback awareness.** `PYTORCH_ENABLE_MPS_FALLBACK=1` documented in README; not yet exercised. Likely first fires on some attention or softmax op when stage 9 actually runs on MPS, or on RoPE in stage 13.
- **Cursor autocomplete inserting bogus imports.** Twice now — `from torch._dynamo.utils import V` in stage 2 and `from _pytest.monkeypatch import V` in stage 3. The pattern keeps masking real bugs (silently letting typos like `attn @ V` (capital) typecheck instead of NameError-ing). Should investigate Cursor's autocomplete config more thoroughly before stage 4 — turning off "Cursor Tab" last time was insufficient.
- **Shape-arithmetic slips.** Pattern noticed in stage 3 — toy prediction had `2*2 + 2 = 8` and the bias-count test had `out_proj_bias_num = d_v` (passed coincidentally because `d_v = d_model = 128`). Both arithmetic slips that wouldn't have happened on paper. Defensive habit: write shape arithmetic on paper even when it looks like it can be done in head.
- **Mechanistic-interpretability framing** (residual stream as shared communication channel, QK/OV circuit decomposition). Introduced in stage 2 and reinforced in stage 3 but Denis hasn't deeply engaged with the literature. Pays off in stages 7-8 (assembling the full GPT) and especially in any later mechanistic-interp investigations. "A Mathematical Framework for Transformer Circuits" (Olah, Elhage et al., Anthropic) is the foundational paper — optional reading whenever the architectural intuition feels fuzzy.
- **Wall-clock training time on M4 Pro.** Still untested. `d_model=128`, `n_layers=4-6`, `T=256` should be tractable in tens of minutes; benchmark when stage 9 lands.
- **jaxtyping `"L" not defined` warning** still ignored — cosmetic, will accumulate as more modules are annotated.

## Code state
- `README.md` — 15 stages listed; setup commands; M4 Pro / MPS notes. ✓
- `CLAUDE.md` — tutoring rules + session-bootstrap pointer to PROGRESS.md. ✓
- `.gitignore`, `requirements.txt` — set up. ✓
- `.venv/` — Python 3.13.5, MPS available and verified. ✓
- `data/input.txt` — tiny-shakespeare, 1,115,394 bytes, gitignored. ✓
- `src/__init__.py` — package marker. ✓
- `src/data.py` — Stage 1: `load_corpus()`, `Tokenizer`, `TokenizedDataset.get_batch`. ✓
- `src/model.py` — Stage 2: `TokenEmbedding(V, d_model)`, `LearnedPositionalEmbedding(T_max, d_model)`. ✓
- `src/attention.py` — Stage 3: `Attention(T_max, d_k, d_v, d_model)`. Fused QKV projection, register_buffer'd causal mask, single-head scaled dot-product attention with output projection. ✓
- `tests/conftest.py` — shared `text` and `tok` fixtures (module-scoped). ✓
- `tests/test_data.py` — 7 tests covering Stage 1. ✓
- `tests/test_model.py` — 9 tests covering Stage 2. ✓
- `tests/test_attention.py` — 5 tests covering Stage 3 (shape/dtype, qkv param count, total param count, causality, mask non-parameter). ✓
- `derivations/` — directory exists, empty. Stage 3's `√d_k` derivation lived inline in conversation; could be moved to `derivations/sqrt_d_scaling.md` if formalized, but not required.
- `notes/PROGRESS.md` — this file.
- `notes/stage_1_*.md` (3 files: tokenization_concepts, batch_sampling, summary).
- `notes/stage_2_*.md` (3 files: embedding_concepts, embedding_modules, summary).
- `notes/stage_3_*.md` (2 files: attention_implementation, summary).
- **21 tests passing total.** Smoke tests in each `__main__` block also work.
- **No multi-head, no transformer block, no full GPT, no training loop yet.** Stages 4-15 not begun.

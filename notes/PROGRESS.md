# Progress

## Where we are
- **Stage:** 5 — LayerNormalization. **DONE.** Summary at `notes/stage_5_summary.md`.
- **Sub-step:** Stage 6 not yet started — pointwise FFN / MLP with GELU.
- **Last completed:** Workflow fix — extended `/stage-done` to refresh PROGRESS.md as a final step, eliminating the drift pattern that left this file stale across stages 4-5. Stage 5 itself finished earlier with all three criteria met (3 tests, brief walkthrough, shape-only toy prediction).

## Resume here
Begin Stage 6: pointwise FFN / MLP with GELU. Conceptually: two `nn.Linear` layers with a nonlinearity between them, applied independently to each token — no position-mixing (attention did that in stages 3/4). Standard architecture: `x → Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model)` with `d_ff = 4 · d_model` by Vaswani/GPT convention. Conceptual probe first (why MLP at all; why 4× expansion; GELU vs ReLU vs SwiGLU; bias terms), then implementation, then tests + walkthrough + toy prediction, then `/stage-done 6`. CLAUDE.md anti-shortcut rule applies: build from `nn.Linear` and `F.gelu` (or manual GELU); no convenience wrappers. After stage 6, stage 7 finally assembles attention + MLP + LayerNorm into the transformer block — the first end-to-end composition.

## Open conceptual debts
- **Recurring `super().__init__()` wording imprecision.** Across stages 2-5 line-by-line walkthroughs, the parent-init was described as "inheriting parent's parameters/methods". Accurate framing: `super().__init__()` initializes parent *instance state* (the `_parameters`, `_modules`, `_buffers` OrderedDicts), not parameters or methods. Worth pinning before stage 7 where multiple submodules are composed.
- **Device placement strategy.** Encoded corpus on CPU; batches need `.to(device)` before the model at training time (stage 9). Exact pattern (per-batch transfer vs preallocated buffers) is open. Tests and smoke tests run on CPU; first MPS run will probably happen at stage 9.
- **MPS-fallback awareness.** `PYTORCH_ENABLE_MPS_FALLBACK=1` documented in README; not yet exercised. May first fire on GELU (historical MPS coverage gaps for some activations) in stage 6, or in the training loop at stage 9. Watch for fallback warnings.
- **Shape-arithmetic and broadcast-shape vs actual-shape slips.** Recurring pattern across stages 3-5: toy predictions had arithmetic errors (`2·2+2=8` in stage 3; mask shape `(1,1,T,T)` vs actual `(T,T)` in stage 4). Defensive habit: write shape arithmetic on paper, distinguish `.shape` (actual) from broadcast pattern.
- **Mechanistic-interpretability framing carried forward.** Residual stream as shared communication channel, QK/OV circuit decomposition (stage 3-4), coordinate-vs-direction "feature" distinction (stage 5). Denis hasn't deeply engaged with the literature yet; the framing pays off in stages 7-8 (assembled GPT) and any later interpretability work. "A Mathematical Framework for Transformer Circuits" (Elhage et al., Anthropic) is the foundational paper.
- **Pre-norm decision.** Resolved at stage 5 — will use pre-norm at stage 7. The LayerNorm class itself is placement-agnostic; the choice is structural at the transformer-block level.
- **Boredom risk at later stages.** Stage 5 hit a protocol-fatigue point; calibrated to "3 tests + brief walkthrough + shape-only toy prediction" rather than full intensity. Worked for stage 5 because LayerNorm is genuinely simple. Stage 7+ (transformer block, full GPT, training loop) needs full protocol intensity — name it explicitly if Denis tries to skip when the substance is harder.
- **jaxtyping `"L not defined"` warning.** Cosmetic, ignored. Will keep accumulating.
- **Wall-clock training time on M4 Pro.** Untested. Becomes relevant at stage 9.

## Code state
- `README.md` — 15 stages listed, setup, M4 Pro / MPS notes. ✓
- `CLAUDE.md` — tutoring rules, session-bootstrap pointer to PROGRESS.md, documentation-command output paths. ✓
- `.gitignore`, `requirements.txt` — set up. ✓
- `.venv/` — Python 3.13.5, MPS verified. ✓
- `data/input.txt` — tiny-shakespeare, 1,115,394 bytes, gitignored. ✓
- `src/__init__.py` — package marker. ✓
- `src/data.py` — Stage 1: `load_corpus()`, `Tokenizer`, `TokenizedDataset.get_batch`. ✓
- `src/model.py` — Stage 2: `TokenEmbedding(V, d_model)`, `LearnedPositionalEmbedding(T_max, d_model)`. ✓
- `src/attention.py` — Stages 3-4: `Attention(T_max, d_k, d_v, d_model)` single-head, `MultiHeadAttention(T_max, n_heads, d_model)`. ✓
- `src/normalization.py` — Stage 5: `LayerNormalization(d_model, eps=1e-5, bias=True)`. ✓
- `tests/conftest.py` — shared `text` and `tok` fixtures (module-scoped). ✓
- `tests/test_data.py` — 7 tests (Stage 1). ✓
- `tests/test_model.py` — 9 tests (Stage 2). ✓
- `tests/test_attention.py` — 10 tests (Stages 3-4; 5 single-head + 5 multi-head, mirrored). ✓
- `tests/test_normalization.py` — 3 tests (Stage 5). ✓
- `derivations/` — directory exists, empty. Stage 3's √d_k derivation and stage 5's softmax-Jacobian-at-saturation derivation lived inline.
- `notes/PROGRESS.md` — this file.
- `notes/stage_1_*.md` (3 files: tokenization_concepts, batch_sampling, summary).
- `notes/stage_2_*.md` (3 files: embedding_concepts, embedding_modules, summary).
- `notes/stage_3_*.md` (2 files: attention_implementation, summary).
- `notes/stage_4_*.md` (2 files: multihead_implementation, summary).
- `notes/stage_5_*.md` (2 files: summary, workflow_fix).
- `.claude/commands/stage-done.md` — updated to refresh PROGRESS.md after writing the summary.
- **29 tests passing total** (7 data + 9 model + 10 attention + 3 normalization). Smoke tests in each `__main__` block also work.
- **No MLP, no transformer block, no full GPT, no training loop yet.** Stages 6-15 not begun.

## Workflow updates
- `/stage-done` now updates PROGRESS.md as a final step (added 2026-05-14). Previously only `/checkpoint` touched it, which led to stale-state drift across stages 4-5. `/checkpoint` remains available for explicit full rewrites and mid-stage pauses; the new behavior closes the most common drift case.

# Progress

## Where we are
- **Stage:** 6 — Pointwise FFN / MLP with GELU. **DONE.** Summary at `notes/stage_6_summary.md`; substantive note at `notes/stage_6_mlp.md`.
- **Sub-step:** Stage 7 not yet started — transformer block (pre-norm residual composition).
- **Last completed:** Stage 6 finished with all three criteria met (12 tests passing across 3 functions: shape/dtype, parametrized parameter count over 10 (d_ff × bias) combinations, GELU-applied homogeneity-violation check; brief inline-comment walkthrough; toy shape prediction through up_proj → GELU → down_proj). `/stage-done 6` produced summary + PROGRESS.md refresh.

## Resume here
Begin Stage 7: transformer block. This is the first stage that *composes* multiple previous stages into a single coherent unit. Pre-norm residual structure (resolved at stage 5; standard since GPT-2): `x = x + attn(LN_1(x))` followed by `x = x + mlp(LN_2(x))`. The block has two `LayerNormalization` instances (each with its own γ, β), one `MultiHeadAttention` instance, one `MLP` instance, and two residual connections. Conceptual probe should hit: pre-norm vs post-norm gradient flow (covered in stage 5), residual-connection mechanics, dropout placement (modern models often skip it; need to decide), whether to share LayerNorm across the two phases (no — each phase has its own γ, β with independent learned scales). **Full protocol intensity returns** — no calibrating to lightweight. Stage 7's design choices compose into the model architecture, and a wrong choice here silently breaks training in stages 9+ rather than surfacing immediately. After stage 7, stage 8 stacks `n_layers` blocks plus embeddings plus final LayerNorm + unembedding head into the full GPT.

## Open conceptual debts
- **Recurring `super().__init__()` wording imprecision.** In stage-2, -3, -4, -5, -6 line-by-line walkthroughs, the parent-init was described as "inheriting parent's parameters/methods". Accurate framing: `super().__init__()` initializes parent *instance state* (the `_parameters`, `_modules`, `_buffers` OrderedDicts), not parameters or methods. Five stages of the same minor imprecision — still not internalized. Worth flagging in stage 7's walkthrough since transformer block has more submodules to register.
- **`bias=True` vs `bias=False` inconsistency** between stages 3-4 (attention, bias=True from `nn.Linear` default) and stage 6 (MLP, bias=False explicitly). Cosmetic inconsistency. Modern convention is bias=False throughout. Could be made uniform during stage 7 work — or revisited as a quick fix later. Not urgent.
- **Device placement strategy.** Encoded corpus on CPU; batches need `.to(device)` before the model at training time (stage 9). Exact pattern (per-batch transfer vs preallocated buffers) is open. Tests and smoke tests run on CPU; first MPS run will likely happen at stage 9.
- **MPS-fallback awareness.** `PYTORCH_ENABLE_MPS_FALLBACK=1` documented in README; not yet exercised. Might first fire on GELU (historical MPS coverage gaps for some activations) once stage 7 runs the full block, or in the training loop at stage 9. Watch for fallback warnings.
- **Shape-arithmetic and broadcast-shape vs actual-shape slips.** Recurring pattern across stages 3-5: toy predictions had arithmetic errors (`2·2+2=8` in stage 3; mask shape `(1,1,T,T)` vs actual `(T,T)` in stage 4). Defensive habit: write shape arithmetic on paper; distinguish `.shape` (actual) from broadcast pattern. Stage 6 didn't have this since shapes were trivial; stage 7+ will revive the pattern (block composition involves more intermediate tensors).
- **Mechanistic-interpretability framing.** Residual stream as shared communication channel, QK/OV circuit decomposition (stage 3-4), coordinate-vs-direction "feature" distinction (stage 5), MLP-as-key-value-memory (stage 6, Geva et al. 2021). Denis hasn't deeply engaged with the literature yet; framing pays off in stages 7-8 (assembled GPT) and any later interpretability work. "A Mathematical Framework for Transformer Circuits" (Elhage et al., Anthropic) is the foundational paper to read before stage 8.
- **Pre-norm decision** — resolved at stage 5, will use at stage 7. The LayerNorm class itself is placement-agnostic; the choice is structural at the block level.
- **Boredom risk** — calibrated to "3 tests + brief walkthrough + shape-only prediction" at stages 5 and 6. Worked because both modules are simple. **Stage 7+ requires full protocol intensity** — composition introduces design choices (residual placement, dropout, normalization sharing) that lightweight protocol won't catch. Name it explicitly if Denis tries to skip.
- **jaxtyping `"L not defined"` warning.** Cosmetic, ignored. Will keep accumulating across stages.
- **Wall-clock training time on M4 Pro.** Untested. Becomes relevant at stage 9.
- **SwiGLU at stage 14** — forward reference saved at `notes/stage_14_swiglu_reference.md` during stage 6. Re-verify the sigma-pi-vs-sigma framing when revisiting; the "result solid, mechanistic theory fuzzy" disclaimer about Shazeer 2020 is worth a fresh look against any 2025+ work that settles the theory.

## Code state
- `README.md` — 15 stages listed, setup, M4 Pro / MPS notes. ✓
- `CLAUDE.md` — tutoring rules, session-bootstrap pointer to PROGRESS.md, documentation-command output paths. ✓
- `.gitignore`, `requirements.txt` — set up. ✓
- `.venv/` — Python 3.13.5, MPS verified. ✓
- `data/input.txt` — tiny-shakespeare, 1,115,394 bytes, gitignored. ✓
- `src/__init__.py` — package marker. ✓
- `src/data.py` — Stage 1: `load_corpus()`, `Tokenizer`, `TokenizedDataset.get_batch`. ✓
- `src/model.py` — Stage 2: `TokenEmbedding(V, d_model)`, `LearnedPositionalEmbedding(T_max, d_model)`. ✓
- `src/attention.py` — Stages 3-4: `Attention(T_max, d_k, d_v, d_model)` single-head, `MultiHeadAttention(T_max, n_heads, d_model)`. Both `bias=True` (nn.Linear default). ✓
- `src/normalization.py` — Stage 5: `LayerNormalization(d_model, eps=1e-5, bias=True)`. ✓
- `src/mlp.py` — Stage 6: `MLP(d_model, d_ff=None, bias=False)`. `d_ff` defaults to `4·d_model`. ✓
- `tests/conftest.py` — shared `text` and `tok` fixtures (module-scoped). ✓
- `tests/test_data.py` — 7 tests (Stage 1). ✓
- `tests/test_model.py` — 9 tests (Stage 2). ✓
- `tests/test_attention.py` — 10 tests (Stages 3-4; 5 single-head + 5 multi-head, mirrored). ✓
- `tests/test_normalization.py` — 3 tests (Stage 5). ✓
- `tests/test_mlp.py` — 12 tests (Stage 6; 1 shape + 10 parametrized param counts + 1 nonlinearity-applied). ✓
- `derivations/` — directory exists, empty. Stage 3's √d_k derivation and stage 5's softmax-Jacobian-at-saturation derivation lived inline in conversation.
- `notes/PROGRESS.md` — this file.
- `notes/stage_1_*.md` (3 files: tokenization_concepts, batch_sampling, summary).
- `notes/stage_2_*.md` (3 files: embedding_concepts, embedding_modules, summary).
- `notes/stage_3_*.md` (2 files: attention_implementation, summary).
- `notes/stage_4_*.md` (2 files: multihead_implementation, summary).
- `notes/stage_5_*.md` (3 files: summary, workflow_fix, layernorm).
- `notes/stage_6_*.md` (2 files: summary, mlp).
- `notes/stage_14_swiglu_reference.md` — forward reference for the optional SwiGLU migration.
- `.claude/commands/stage-done.md` — updated 2026-05-14 to refresh PROGRESS.md after writing the summary. `.claude/commands/checkpoint.md` and `note.md` unchanged.
- **41 tests passing total** (7 data + 9 model + 10 attention + 3 normalization + 12 mlp). Smoke tests in each `__main__` block also work.
- **No transformer block, no full GPT, no training loop yet.** Stages 7-15 not begun.

## Workflow updates
- `/stage-done` now updates PROGRESS.md as a final step (added 2026-05-14). `/checkpoint` remains the full-rewrite alternative for explicit end-of-session or mid-stage snapshots.

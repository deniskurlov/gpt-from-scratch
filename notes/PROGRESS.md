# Progress

## Where we are
- **Stage:** 8 — Full GPT. **DONE.** Summary at `notes/stage_8_summary.md`.
- **Sub-step:** Stage 9 not yet started — training loop (AdamW + cross-entropy + LR schedule + gradient clipping + eval/checkpoint hooks).
- **Last completed:** Stage 8 finished with all three "done" criteria met (5 stage-8 tests passing: shape/dtype, return-path with `targets=None` and `targets=y` paths plus backward-fills-grad on all parameters, initial-loss in `[0.8·log V, 1.2·log V]`, end-to-end causality with `eval()` and an element-wise `any(...)` reroll, and parameter count parametrized over 4×4 = 16 `(d_ff, n_layers)` combinations — 21 collected, 66 in suite total). Walkthrough completed with several precision corrections (ModuleList registers in `_modules` not the autograd graph; weight tying held as operation-level transpose + semantic-level same-vectors + storage-level same-tensor; tying saves memory not compute; log V re-derived from small-init → uniform softmax → `-log(1/V)`). Toy prediction: shape (B,T,V); logits ≈ 0 with std ~0.2 from the tied embedding's std=0.02 init; loss ≈ log V ≈ 4.17; B-uniform under eval mode (deterministic, identical inputs), T-varying (different positional embeddings + different causal-attention prefixes). `/stage-done 8` produced summary + this PROGRESS.md refresh.

## Resume here
Begin Stage 9: training loop. Build `src/train.py` with: `AdamW(model.parameters(), lr=...)`, cross-entropy loss via the `(logits, loss)` path of `GPT.forward`, an LR schedule (linear warmup followed by cosine decay), gradient clipping (`torch.nn.utils.clip_grad_norm_`), and a basic eval loop on held-out tokens. First device transfer: `model.to(device)` + per-batch `x.to(device)`, `y.to(device)`. First MPS run — watch for op-fallback warnings (`PYTORCH_ENABLE_MPS_FALLBACK=1` is documented in README). Initial probe: can the model overfit a single batch (loss → ≈0)? This is the canonical smoke test for "training loop is wired correctly." After that: small-scale training on tiny-shakespeare with eval-loss tracking.

## Open conceptual debts
- **Walkthrough-imprecision pattern.** Three stages now have shown the same kind of precision slip during the line-by-line: stage-7 "super().__init__() inherits parent's parameters/methods" (initializes parent *instance state*); stage-8 "ModuleList registers in the grad graph" (registers in `_modules` for `.parameters()`/`.to()`/`.train()/.eval()` visibility — autograd is operation-level and orthogonal); stage-8 "tied weights speed up training" (saves *memory* — params + Adam state — at the same FLOP count). Pattern: collapses a precise mechanism into a vaguely-correct slogan. Stage-9 walkthrough on AdamW and gradient clipping is a fresh opportunity for this to recur — push hard on "what exactly does AdamW track per parameter" and "what does clip_grad_norm_ mutate, in place or returned new tensor".
- **CLAUDE.md push-back trigger fired** at stage-8 criterion 3. Denis tried to skip the toy prediction ("seems too much for a mental exercise") after only doing the shape question. Refused per protocol; simplified the questions to order-of-magnitude + B-vs-T uniformity; he then answered correctly. Pattern to watch: criterion 3 (predict-then-check) is the criterion most likely to get skipped because it's the most uncomfortable. Stage 9's analog is "predict the trajectory of training loss in the first 100 steps without running" — keep insisting.
- **`bias=True` (stages 3-4 attention) vs `bias=False` (stage 6 MLP, stage 8 lm_head) inconsistency.** Still cosmetic; modern convention is `bias=False` throughout for LLaMA-style models. Could be unified at any time; the parameter-count test in stage 8 would need a small adjustment. Not urgent.
- **Device placement strategy.** Encoded corpus on CPU; batches need `.to(device)` before the model. Exact pattern (per-batch transfer vs. preallocated buffers) is open. First exercised in stage 9.
- **MPS-fallback awareness.** `PYTORCH_ENABLE_MPS_FALLBACK=1` documented in README; not yet exercised. Likely first fires on some op during stage 9 (perhaps GELU or layernorm-variance in eval mode). Watch for warnings.
- **Shape-arithmetic slips.** Stage 8 surfaced two: missing `T_max · d_model` for `pos_emb` in the param-count formula, and misapplied "zero this term" for tied weights (zeroed `final_ln`, should have been `lm_head`). The "walk through every `self.X = ...`" exercise during the param-count test was what caught both. Defensive habit to keep.
- **Mechanistic-interpretability framing.** Residual stream as shared communication channel; QK/OV circuit decomposition; coordinate-vs-direction "feature" distinction; MLP-as-key-value-memory. Still un-engaged with the literature ("A Mathematical Framework for Transformer Circuits", Elhage et al.). Stage 8 closed the assembled architecture; stage 9 starts learning dynamics. Reading the framework paper before stage 11 (KV cache; the QK/OV decomposition is foundational to understanding what gets cached) is the natural deadline.
- **jaxtyping `"L not defined"` warning.** Cosmetic, ignored. Will keep accumulating.
- **Wall-clock training time on M4 Pro.** Becomes real in stage 9. The first end-to-end batch on MPS is the wall-clock probe.
- **SwiGLU at stage 14** — forward reference at `notes/stage_14_swiglu_reference.md`. Re-verify the sigma-pi-vs-sigma framing against any post-2024 work that may have settled the theory.

## Code state
- `README.md` — 15 stages listed, setup, M4 Pro / MPS notes. ✓
- `CLAUDE.md` — tutoring rules, session-bootstrap pointer to PROGRESS.md, documentation-command output paths. ✓
- `.gitignore`, `requirements.txt` — set up. ✓
- `.venv/` — Python 3.13.5, MPS verified. ✓
- `data/input.txt` — tiny-shakespeare, 1,115,394 bytes, gitignored. ✓
- `src/__init__.py` — package marker. ✓
- `src/data.py` — Stage 1: `load_corpus()`, `Tokenizer`, `TokenizedDataset.get_batch`. ✓
- `src/embedding.py` — Stage 2 (relocated from `src/model.py` during stage 8 refactor): `TokenEmbedding(V, d_model)`, `LearnedPositionalEmbedding(T_max, d_model)`. ✓
- `src/attention.py` — Stages 3-4: `Attention(T_max, d_k, d_v, d_model)` single-head, `MultiHeadAttention(T_max, n_heads, d_model)`. Both `bias=True` (nn.Linear default). ✓
- `src/normalization.py` — Stage 5: `LayerNormalization(d_model, eps=1e-5, bias=True)`. ✓
- `src/mlp.py` — Stage 6: `MLP(d_model, d_ff=None, bias=False)`. `d_ff` defaults to `4·d_model`. ✓
- `src/model.py` — Stage 7: `Block(T_max, n_heads, d_model, d_ff=None, dropout=0.1)`. Stage 8: `GPT(V, T_max, n_heads, d_model, n_layers, d_ff=None, dropout=0.1)` — tok_emb + pos_emb + ModuleList of Blocks + final_ln + tied lm_head with N(0, 0.02) re-init; two-path forward with `(logits, loss)` when `targets` passed. `self.V` stored on the instance. ✓
- `tests/conftest.py` — shared `text` and `tok` fixtures (module-scoped). ✓
- `tests/test_data.py` — 7 tests (Stage 1). ✓
- `tests/test_embedding.py` — 9 tests (Stage 2, relocated from `tests/test_model.py` during stage-8 refactor). ✓
- `tests/test_attention.py` — 10 tests (Stages 3-4). ✓
- `tests/test_normalization.py` — 3 tests (Stage 5). ✓
- `tests/test_mlp.py` — 12 tests (Stage 6). ✓
- `tests/test_model.py` — 5 tests (Stage 7 Block) + 5 tests (Stage 8 GPT; the parametrized param-count test expands to 16, so 21 collected from this file). ✓
- `derivations/` — directory exists, empty. Stage 3's √d_k derivation, stage 5's softmax-Jacobian-at-saturation, and stage 8's "logit std → log V" derivation lived inline in conversation.
- `notes/PROGRESS.md` — this file.
- `notes/stage_1_*.md` (3 files).
- `notes/stage_2_*.md` (3 files).
- `notes/stage_3_*.md` (2 files).
- `notes/stage_4_*.md` (2 files).
- `notes/stage_5_*.md` (3 files).
- `notes/stage_6_*.md` (2 files).
- `notes/stage_7_*.md` (2 files).
- `notes/stage_8_summary.md` — new; written by this `/stage-done 8` invocation.
- `notes/stage_14_swiglu_reference.md` — forward reference for SwiGLU.
- `.claude/commands/stage-done.md` — updated 2026-05-14 to refresh PROGRESS.md after writing the summary. `.claude/commands/checkpoint.md` and `note.md` unchanged.
- **66 tests passing total** (7 data + 9 embedding + 21 model [5 stage-7 + 5 stage-8 expanded by parametrize] + 10 attention + 3 normalization + 12 mlp). Smoke tests in each `__main__` block also work.
- **Full GPT assembled and tested. No training loop yet.** Stages 9-15 not begun.

## Workflow updates
- `/stage-done` updates PROGRESS.md as a final step (in place since 2026-05-14). `/checkpoint` remains the full-rewrite alternative for explicit end-of-session or mid-stage snapshots.

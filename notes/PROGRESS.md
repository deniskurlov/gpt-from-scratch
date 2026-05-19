# Progress

## Where we are
- **Stage:** 9 — Training loop. **DONE.** Summary at `notes/stage_9_summary.md`.
- **Sub-step:** Stage 10 not yet started — actual training on TinyShakespeare with held-out eval split, eval-loss tracking, and qualitative sample inspection.
- **Last completed:** Stage 9 finished with the canonical AdamW + cross-entropy + LR warmup + cosine decay machinery in `src/train.py`. End-to-end TinyShakespeare run: 5000 steps, B=64, T=256, lr=1e-3 (peak), warmup=100, cosine to 10% of peak. Training loss 4.20 (= log V) → 1.59 over ~10m24s wall-clock on M4 Pro / MPS, ~80M training tokens, ~70 dataset cycles. Trajectory hit every shape predicted: warmup learning of marginal distribution (4.20 → 3.27), plateau at unigram baseline (steps 100-450), sharp phase-transition at step ~500 (3.25 → 2.83) as attention circuits formed, smooth descent through cosine decay (2.83 → 1.59). All AdamW math derived and reviewed (moment EMAs, bias correction, decoupled weight decay vs L2-in-loss, ε for numerical stability). Warmup motivation grounded in variance-of-`v̂_t` + Jensen's-inequality on `1/sqrt(x)`. Cosine motivation grounded in flat-endpoint geometry (η'(0) = η'(T) = 0). `/stage-done 9` produced summary + this PROGRESS.md refresh.

## Resume here
Begin Stage 10: actual training + held-out eval + qualitative sample inspection. First concrete step: add a train/val split inside `TokenizedDataset` (e.g., 90/10 at corpus level — split the encoded tensor before constructing the dataset, or hold two `TokenizedDataset` instances). Then modify the training loop to periodically (every `eval_interval` steps, e.g., 250) put the model in `eval()` mode under `torch.no_grad()`, sample several batches from the val split, average the loss, and print alongside the training loss. Track train-vs-eval divergence — the gap is the diagnostic. Don't add sampling yet (that's stage 11); the qualitative check at stage 10 is just "does the eval loss decrease too, or are we memorizing?" Predict the eval loss trajectory before running.

## Open conceptual debts
- **Predict-then-check chronically skipped at stage 9.** Denis ran first and described after on multiple experiments (LR sweep at T=4, T=256; B=2 vs B=64; first 5000-step run). Tutor pushed back each time but didn't refuse `/stage-done 9` because the substantive work was solid. **Stage 10 is the exact place where this matters most**: eval loss will diverge from train loss in ways you need to learn to anticipate, and only by predicting the gap in advance can you tell signal from noise. Flag aggressively if it recurs at stage 10. CLAUDE.md is explicit: predict-then-check is the criterion for "I understand what should happen."
- **Slogan-replaces-mechanism pattern.** Four stages of recurrence now: (1) "super().__init__() inherits parent's parameters/methods" stages 2-7 (initializes parent *instance state*); (2) "ModuleList registers in the grad graph" stage 8 (registers in `_modules` for `.parameters()`/`.to()`/`.train()/.eval()` visibility); (3) "tied weights speed up training" stage 8 (saves *memory*, not compute); (4) "cosine is smooth from 1 to 0" stage 9 (smoothness is the wrong load-bearing property; the distinguishing feature is η'(0) = η'(T) = 0). Hypothesis: Denis reaches for the closest-fitting slogan from training-data-shaped intuition instead of mechanistically tracing each component. Stage 10/11 watch-items: "model.eval() turns off dropout" (precise: also turns off LayerNorm's running-stats mode for `nn.LayerNorm` — N/A for our hand-rolled `LayerNormalization`, but the mental model is incomplete), "no_grad() saves memory" (precise: disables autograd graph construction, which both saves memory AND speeds the forward; the two effects compound), "argmax sampling is deterministic" (precise: deterministic given the same model state and same input, but the next-token distribution itself is what you'd want to inspect).
- **Eval-vs-train loss distinction**, newly relevant for stage 10. Final training loss = 1.59 with the model having seen the dataset ~70 times. Heavy memorization expected. Eval loss on a held-out split could be 1.8 to 2.5 depending on overfit. The gap is the diagnostic: small gap = well-regularized; large gap = memorizing without generalizing. Stage 10's main pedagogical content.
- **`bias=True` (stages 3-4 attention) vs `bias=False` (stage 6 MLP, stage 8 lm_head) inconsistency.** Still cosmetic; could be made uniform. Not urgent.
- **AdamW + selective weight decay** by parameter group. Standard recipe is `weight_decay=0` for LayerNorm γ/β and biases, `weight_decay=0.1` for matrix params. Stage 9 used `weight_decay=0` globally as the minimal setup. Worth implementing during stage 10 or 11 if overfitting (visible as growing train-eval gap) becomes the bottleneck.
- **MPS-fallback awareness.** `PYTORCH_ENABLE_MPS_FALLBACK=1` documented in README; not yet observed firing during stage 9's 5000-step run. Either no ops triggered fallback, or it happened silently. Watch for warnings during stage 10 (longer runs may surface rare-op fallbacks).
- **Wall-clock training time on M4 Pro.** Stage 9 measured: ~10m24s for 5000 steps with B=64, T=256, d_model=128, 6 layers. Scaling: roughly ~120ms/step at this configuration. Stage 10 won't need significantly longer unless model size or step count changes.
- **Mechanistic-interpretability framing** (residual stream, QK/OV circuits, etc.). Still un-engaged. The phase transition at step ~500 in stage 9 is a clean live example of "circuit formation" — Denis observed it but didn't yet connect to the interpretability literature. "A Mathematical Framework for Transformer Circuits" (Elhage et al.) is still the natural pre-stage-11 (KV cache) read, where the QK/OV decomposition is foundational.
- **jaxtyping `"L not defined"` warning.** Cosmetic, ignored. Will keep accumulating.
- **SwiGLU at stage 14** — forward reference at `notes/stage_14_swiglu_reference.md`. Re-verify against any post-2024 work that may have settled the theory.

## Code state
- `README.md` — 15 stages listed, setup, M4 Pro / MPS notes. ✓
- `CLAUDE.md` — tutoring rules, session-bootstrap pointer to PROGRESS.md, documentation-command output paths. ✓
- `.gitignore`, `requirements.txt` — set up. ✓
- `.venv/` — Python 3.13.5, MPS verified. ✓
- `data/input.txt` — tiny-shakespeare, 1,115,394 bytes, gitignored. ✓
- `src/__init__.py` — package marker. ✓
- `src/data.py` — Stage 1: `load_corpus()`, `Tokenizer`, `TokenizedDataset.get_batch`. ✓
- `src/embedding.py` — Stage 2: `TokenEmbedding(V, d_model)`, `LearnedPositionalEmbedding(T_max, d_model)`. ✓
- `src/attention.py` — Stages 3-4: `Attention(T_max, d_k, d_v, d_model)` single-head, `MultiHeadAttention(T_max, n_heads, d_model)`. Both `bias=True` (nn.Linear default). ✓
- `src/normalization.py` — Stage 5: `LayerNormalization(d_model, eps=1e-5, bias=True)`. ✓
- `src/mlp.py` — Stage 6: `MLP(d_model, d_ff=None, bias=False)`. `d_ff` defaults to `4·d_model`. ✓
- `src/model.py` — Stage 7: `Block(T_max, n_heads, d_model, d_ff=None, dropout=0.1)`. Stage 8: `GPT(V, T_max, n_heads, d_model, n_layers, d_ff=None, dropout=0.1)` with tied lm_head and N(0, 0.02) init. ✓
- `src/train.py` — Stage 9: device pick (MPS / CPU), corpus load, tokenizer + dataset, model construction + `.to(device)`, AdamW (lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0), `LambdaLR` with linear-warmup + cosine-decay closure via `make_lr_lambda` factory, six-operation training step inside 5000-iter loop with per-batch `get_batch` + `.to(device)`, periodic step/lr/loss print. Final training loss 1.59 on TinyShakespeare. ✓
- `tests/conftest.py` — shared `text` and `tok` fixtures (module-scoped). ✓
- `tests/test_data.py` — 7 tests (Stage 1). ✓
- `tests/test_embedding.py` — 9 tests (Stage 2). ✓
- `tests/test_attention.py` — 10 tests (Stages 3-4). ✓
- `tests/test_normalization.py` — 3 tests (Stage 5). ✓
- `tests/test_mlp.py` — 12 tests (Stage 6). ✓
- `tests/test_model.py` — 5 tests (Stage 7 Block) + 5 tests (Stage 8 GPT; param-count parametrize expands to 16 → 21 collected from this file). ✓
- **No tests for `src/train.py`** — it's a script, not a library module. Integration test is the training run producing decreasing loss.
- `derivations/` — directory exists, empty. Stage 3's √d_k derivation, stage 5's softmax-Jacobian-at-saturation, stage 8's "logit std → log V", and stage 9's AdamW math (bias correction, v̂_t variance, cosine endpoint analysis) all lived inline in conversation.
- `notes/PROGRESS.md` — this file.
- `notes/stage_1_*.md` (3 files).
- `notes/stage_2_*.md` (3 files).
- `notes/stage_3_*.md` (2 files).
- `notes/stage_4_*.md` (2 files).
- `notes/stage_5_*.md` (3 files).
- `notes/stage_6_*.md` (2 files).
- `notes/stage_7_*.md` (2 files).
- `notes/stage_8_*.md` (2 files: summary, walkthrough_corrections).
- `notes/stage_9_summary.md` — new; written by this `/stage-done 9` invocation.
- `notes/stage_14_swiglu_reference.md` — forward reference.
- `.claude/commands/stage-done.md` — updated 2026-05-14 to refresh PROGRESS.md after writing the summary. `.claude/commands/checkpoint.md` and `note.md` unchanged.
- **66 tests passing total** (no new tests this stage; stage 9 is integration-tested by the training run itself). Smoke tests in each `__main__` block work; `src/train.py`'s `__main__` is the training script.
- **Full GPT trained end-to-end on TinyShakespeare to training loss 1.59.** No held-out eval yet (stage 10); no sampling yet (stage 11); no KV cache (stage 12); no RoPE (stage 13).

## Workflow updates
- `/stage-done` updates PROGRESS.md as final step (in place since 2026-05-14). `/checkpoint` remains the full-rewrite alternative for explicit end-of-session or mid-stage snapshots.

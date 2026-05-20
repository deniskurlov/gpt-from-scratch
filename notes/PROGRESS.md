# Progress

## Where we are
- **Stage:** 10 — TinyShakespeare training with held-out eval. **DONE.**
- **Sub-step:** Stage 11 not yet started (sampling: greedy → temperature → top-k → top-p).
- **Last completed:** Stage 10's `eval_loss` function + 90/10 sequential split + periodic (every 200 steps) and final eval integrated into the training loop. 5000-step run on TinyShakespeare (B=64, T=256, lr=1e-3, warmup=100, cosine to 10%, weight_decay=0.1, dropout=0.1): train 4.20 → 1.55, val 4.20 → 1.69, eval-min at step 4600 (val=1.6834). Train-val gap stabilizes at ~0.12-0.14 from step ~2400 onward; no U-curve in 5000 steps. Stage 10 summary + eval-function bumps note written.

## Resume here
Open a new `src/sample.py`. Write a greedy decode function: encode prompt → loop `n_new_tokens` times calling `gpt(x)` (the `targets=None` bare-tensor path of `GPT.forward`) → take `argmax` over logits at the last position → append the chosen token to `x` → repeat. **First sub-step**: stage 10 didn't save the model — add `torch.save(model.state_dict(), 'checkpoints/model.pt')` after the training loop in `src/train.py` (create the `checkpoints/` dir, add it to `.gitignore`), re-run training, then load in `sample.py`. After greedy works and the output looks plausibly Shakespeare-like, generalize: temperature (divide logits by `T` before softmax), top-k (mask all-but-top-k logits to `-inf`), top-p / nucleus (mask logits outside the smallest cumulative-prob ≥ p set). Predict the qualitative effect of each before running.

## Open conceptual debts
- **Slogan-vs-mechanism pattern**, recurring across 5+ stages now. Stage 10 added three new instances: `model.eval()` framed as "full power" instead of "deterministic forward"; "restore grad to zero" as eval cleanup instead of `model.train()`; cosine described as "smooth from 1 to 0" instead of "flat-derivative endpoints." The slogan is always close enough to the truth to fool the speaker but misses the specific mechanism. **Stage 11 watch-items**: "temperature controls randomness" (precise: scales logits before softmax; `T<1` sharpens, `T>1` flattens, `T→0` recovers argmax), "top-k truncates the distribution" (precise: masks logits to `-inf` then renormalizes via softmax), "argmax is deterministic" (precise: deterministic given identical logits, which depend on the prompt and model state). Bet: at least one will get slogan'd.
- **Predict-then-check, improving**. Stage 10 broke the chronic skip-pattern from stage 9 — Denis predicted before running and all three predictions hit. Keep enforcing in stage 11. Sample quality is harder to predict numerically than loss, but the discipline ("write your expectation in advance, even if rough") generalizes.
- **Best-eval-checkpoint / early-stopping**: not implemented. Stage 10's val-minimum at step 4600 was 0.008 below the final. Negligible here; can matter at longer runs. Pattern: save `model.state_dict()` at each new eval-min; load the best-eval checkpoint after training.
- **Per-param-group weight decay**: standard nanoGPT recipe (LN γ/β + biases excluded from decay; matrix params included). Not implemented; stage 10 used global `weight_decay=0.1`. Add if overfitting becomes the bottleneck in longer runs.
- **Checkpointing infrastructure**: nothing saved yet. Stage 11 needs a trained checkpoint to sample from. First concrete sub-step of stage 11.
- **`bias=True` (attention) vs `bias=False` (MLP, lm_head) inconsistency**: cosmetic, persistent across 5 stages. Could be uniformly bias=False; not urgent.
- **MPS fallback awareness**: not observed firing in stage 10's 5000-step training. Watch in stage 11 — autoregressive single-token forward passes have a different access pattern from full-sequence training; may surface different op fallbacks.
- **Mechanistic-interpretability framing**: still un-engaged. The clean experimental hooks accumulated (phase transition at step ~500, train-val divergence at step ~2400, the consistent log V at init) are exactly the kind of data the interpretability literature contextualizes. **Action**: read Elhage et al. "A Mathematical Framework for Transformer Circuits" before stage 12 (KV cache; the QK/OV decomposition is foundational to understanding what gets cached).
- **SwiGLU at stage 14** — forward reference queued at `notes/stage_14_swiglu_reference.md`.
- **jaxtyping `"L not defined"` warning** — cosmetic, persistent.
- **Wall-clock training time**: stage 10's 5000-step run took 6m17s on M4 Pro / MPS (vs stage 9's 10m24s same config; system-load variance). ~75ms/step. Inference benchmark for stage 12 (KV cache speedup measurement) will need a baseline; record during stage 11 sampling.

## Code state

**Source (`src/`)**
- `__init__.py` — package marker.
- `data.py` — `load_corpus`, `Tokenizer`, `TokenizedDataset.get_batch`. Stage 1. Tested.
- `embedding.py` — `TokenEmbedding(V, d_model)`, `LearnedPositionalEmbedding(T_max, d_model)`. Stage 2. Tested.
- `attention.py` — `Attention(T_max, d_k, d_v, d_model)`, `MultiHeadAttention(T_max, n_heads, d_model)`. Both `bias=True` (default). Stages 3-4. Tested.
- `normalization.py` — `LayerNormalization(d_model, eps=1e-5, bias=True)`. Stage 5. Tested.
- `mlp.py` — `MLP(d_model, d_ff=None, bias=False)`. Stage 6. Tested.
- `model.py` — `Block(T_max, n_heads, d_model, d_ff=None, dropout=0.1)` (Stage 7) + `GPT(V, T_max, n_heads, d_model, n_layers, d_ff=None, dropout=0.1)` with tied lm_head and `N(0, 0.02)` init (Stage 8). Tested.
- `train.py` — Stages 9-10. AdamW (lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1), `LambdaLR` with linear-warmup + cosine-decay closure, six-operation training step inside 5000-iter loop, periodic + final eval via `eval_loss(model, ds_val, B, T, eval_iters=20, device)`, 90/10 sequential split. Script (not library); integration-tested by training run. Final train 1.55, val 1.69.

**Tests (`tests/`)** — 66 passing total.
- `conftest.py` (shared fixtures), `test_data.py` (7), `test_embedding.py` (9), `test_attention.py` (10), `test_normalization.py` (3), `test_mlp.py` (12), `test_model.py` (21 = 5 Block + 5 GPT parametrized to 16).
- No tests for `src/train.py` — script, not library module.

**Notes (`notes/`)** — PROGRESS.md (this file) + 10 stage summaries (`stage_1_summary.md` through `stage_10_summary.md`) + 13 working-notes files (per-stage topic notes from `/note` invocations) + `stage_14_swiglu_reference.md` (forward reference).

**Other**
- `data/input.txt` — TinyShakespeare, 1,115,394 bytes, gitignored.
- `.gitignore`, `requirements.txt`, `README.md`, `CLAUDE.md` — all in place.
- `.venv/` — Python 3.13.5, MPS verified.
- `derivations/` — exists, empty. All derivations remain inline in conversation history.
- `checkpoints/` — **does not exist yet.** First sub-step of stage 11.
- `.claude/commands/` — `stage-done.md` (updated 2026-05-14 to refresh PROGRESS.md), `checkpoint.md` (this command), `note.md`.

**What does not exist yet**: sampling code, KV cache, RoPE, SwiGLU. No saved model checkpoint. Stage 11+ all open.

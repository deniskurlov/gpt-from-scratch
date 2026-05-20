# Stage 10: TinyShakespeare Training with Held-Out Eval

## Summary
Extended stage 9's training loop into the canonical "train + periodic eval" pattern. Added a 90/10 sequential split on the encoded corpus (no `TokenizedDataset` changes — sliced the tensor in `src/train.py`, constructed two dataset instances), wrote an `eval_loss(model, ds_val, B, T, eval_iters, device)` function using `model.eval()` + `torch.no_grad()` + averaging over `eval_iters=20` val batches with `model.train()` restoration, and integrated periodic eval (every 200 steps) into the loop. End-to-end run on TinyShakespeare with d_model=128, n_layers=6, lr=1e-3, warmup=100, total=5000, weight_decay=0.1, dropout=0.1: train loss 4.20 → 1.55, val loss 4.20 → **1.69 (final)**, eval-loss minimum at step 4600 (val=1.6834). Train-val gap stabilized at ~0.12-0.14 from step ~2400 onward — moderate memorization, no catastrophic overfit. This is the first stage where we measure *what the model can generalize* rather than what it can memorize, and the first concrete data point for "more training ≠ better eval loss" past the minimum.

## The math

**Eval loss as Monte Carlo estimate.** The eval loss is an empirical estimate of `E_{(x,y) ∼ val}[cross_entropy(model(x), y)]`. Single-batch estimates have variance equal to the per-batch variance of cross-entropy; averaging over `eval_iters=K` independent batches reduces variance by a factor of `K`:

```
Var[(1/K) Σ_{i=1}^K loss(x_i, y_i)] = Var[loss] / K
```

For K=20, that's ~20× variance reduction. The number was chosen to be large enough that step-to-step eval-loss noise stays below the trend signal (eval loss decreased by ~0.04 between adjacent eval points around step 4000; single-batch noise was ~0.05-0.1). Standard nanoGPT default.

**Train-val gap diagnostic.** With B=64, T=256, 5000 steps = ~80M training tokens, the model has seen TinyShakespeare's 1M unique tokens ~70× over. The train-eval gap measures how much of the capacity has gone into batch-specific memorization vs generalizable structure:
- Steps 0-1800: gap ≈ 0 (within ±0.03). Both losses tracking; model learning generalizable signal.
- Steps 2000-2400: divergence begins. Train pulls slightly ahead.
- Steps 2600-final: gap stabilizes at 0.12-0.14. Train keeps dropping (1.72 → 1.55) while val plateaus (1.79 → 1.69).

The plateau in val with continued train descent is the textbook "memorizing without generalizing" signature. Here it's mild — gap < 0.15 — because `weight_decay=0.1` + `dropout=0.1` provides moderate regularization.

**Eval-loss minimum vs final.** Min val=1.6834 at step 4600; final val=1.6910 at step 5000. Difference 0.0076 in 400 steps. For longer runs (10K+) the gap between minimum and final can exceed 0.2, making the early-stopping checkpoint substantially better than the final-step checkpoint. This is the practical reason early-stopping logic exists in production training code.

**Initial-loss sanity check.** Step 0 train=4.2006, val=4.2002. Both ≈ log(V) = log(65) ≈ 4.17 (within ε from the std=0.02 tied-embedding init's small logit perturbations). Confirms the model is at uniform-distribution prediction at init; stage-8 prediction lands live at stage-10.

## The code
- `src/train.py` — extended. Added `eval_loss(model, ds_val, B, T, eval_iters, device) -> float` function (eval mode + no_grad + per-iter val batch + average + restore train mode + `.item()` return). Refactored into `main()`. Added 90/10 sequential split via `int(0.9 * len(encoded_text))` cutoff and two `TokenizedDataset` instances. Combined train+val loss into a single print per eval interval. Final eval pass after the training loop.
- `src/data.py` — unchanged. The split happens at the corpus level in `train.py`; the dataset class remains agnostic.

No new tests added. `src/train.py` is a script, not a library module; the integration test is the training run producing decreasing train AND eval loss.

## Design choices and why

- **90/10 sequential split**, not random-window. The corpus is one long character stream — no document boundaries. Splitting at the 90% mark gives two *disjoint* index pools; sampling random offsets *within each pool* cannot leak. Random-window splits with possibly-overlapping samples would risk train sequences and val sequences sharing characters at the boundaries. For TinyShakespeare at T=256 and corpus length 1.1M, the seam-overlap risk is ~0.02% of the corpus — actually negligible — but the principle matters for larger T or smaller corpora.

- **Slice the encoded tensor in `train.py`, construct two `TokenizedDataset` instances.** Option (ii) of three candidates. Reasons:
  - **Separation of concerns.** Dataset class is "hold an encoded tensor + sample batches"; it doesn't need to know about train/val distinction.
  - **Zero change to existing code.** `TokenizedDataset.__init__` and `get_batch` are untouched.
  - **Extensible.** Want a test split later? Construct a third instance. No method explosion.
  - Contrasts with: (i) `split` flag in `__init__` (conflates which-slice with construction); (iii) `get_val_batch` method (couples class to train/val abstraction). nanoGPT uses effectively (ii).

- **Eval idiom: `model.eval()` → `with torch.no_grad():` → average over `eval_iters` val batches → `model.train()`.** Four pieces:
  - **`model.eval()`** makes the forward *deterministic* (dropout off → same input gives same output across calls). Without it, eval loss has dropout-induced noise that's confounded with real learning signal.
  - **`torch.no_grad()`** disables autograd graph construction. Two benefits: no `.grad` populated (no accidental gradient updates), and saves the memory/compute that graph construction would consume.
  - **Average over `eval_iters=20` batches.** Single-batch estimate is high-variance; averaging reduces variance ~20×. Picked to keep eval cost ~8% of training cost (20 forwards × eval_interval=200 → 1 eval forward per 10 training forwards).
  - **`model.train()` before return.** Critical and easy to miss: if you leave the model in eval mode, the rest of training runs with dropout disabled — silently. Regularization just turns off. The bug is invisible until the train-val gap diverges from expectations.

- **Eval at `step % 200 == 0`.** Modulo trigger inside the loop. eval-cost trade-off: every step would be wasteful (each eval is 20 forwards = ~20× per-step training cost), every 1000 steps misses the U-curve features. 200 is nanoGPT's default and gives ~25 eval points over 5000 steps — enough to see the trajectory shape, cheap enough to not bottleneck training.

- **`torch.zeros(eval_iters, device=device)` for the accumulator**, not a Python list. Both work. The `device=device` variant keeps everything on-device through the loop; `.mean().item()` does a single MPS → CPU sync at the end. A Python list with `.item()` per iteration pays a sync per loop iteration. For `eval_iters=20` the savings are microseconds, but the pattern generalizes to larger `eval_iters` where it matters.

- **Combined train+val print per interval**, not two separate prints. Output `step: 200  lr: 9.99e-04  train_loss: 3.32  val_loss: 3.35` is more readable than two interleaved lines per interval.

- **Final eval pass after the training loop.** Captures the end-of-training state. Without it, the last 200 steps after the final mid-loop eval are unmeasured.

- **Eval *after* the training step in the loop body** (not before). Convention choice. NanoGPT evals *before* each interval's training step; we eval *after*. Consequence: step-0 eval reflects the model *after one training update*, not the raw initial state. The "log V ≈ 4.17" baseline is therefore slightly inexact (we observed train=4.2006, val=4.2002 — close enough that you'd never notice in practice). Substantively equivalent.

## Errors and corrections

- **`math.ceiling` doesn't exist.** Python's `math` module has `math.ceil`, not `math.ceiling`. Typo from autocomplete or guess. Surfaced as `AttributeError` immediately on import-and-call.

- **Unnecessary `+ 1` gap in the split**. Initial implementation:
  ```python
  ds_train = TokenizedDataset(encoded_text[:math.floor(0.9 * len(encoded_text))])
  ds_val = TokenizedDataset(encoded_text[math.ceil(0.9 * len(encoded_text)) + 1:])
  ```
  Created a 2-character gap. Defensive but not needed: `[:N]` and `[N:]` with `N = int(0.9 * L)` already give disjoint pools (train uses indices `[0, N)`, val uses `[N, L)`). Simpler is correct.

- **Eval function entirely missing on first "done" pass.** Reported "done" after setting up the split but before implementing `eval_loss`. The loop still printed train loss only, no val loss. `ds_val` was created and never used. Lesson: "done" should mean "the feature works end-to-end," not "I made the structural changes." The tutor caught it by reading the loop body.

- **`weight_decay` changed silently from stage 9's 0.0 to 0.1**, between editing sessions. Stage 9's training-loss curve was generated with WD=0.0; stage 10's was with WD=0.1. The two runs are not strictly comparable on train-loss because the configs differ. Lesson: config drift is the easiest way to make experiments incomparable; if you change a hyperparameter, change it deliberately and document.

- **Critical eval-function bug**: sampling the val batch *outside* the loop, then re-running the same `(x, y)` through the model `eval_iters` times:
  ```python
  x, y = ds_val.get_batch(B, T)  # OUTSIDE loop
  for i in range(eval_iters):
      _, loss = model(x, targets=y)  # same batch every iteration
      losses[i] = loss               # all losses identical
  ```
  Defeats the purpose of `eval_iters` — variance reduction by averaging *different* val batches. With `model.eval()` (dropout off), the forward is deterministic, so `losses[0] == losses[1] == ... == losses[19]`. Paying 20× compute for 1× information. Fix: move `get_batch` *inside* the loop.

- **Type annotation slips**:
  - `eval_loss -> float` but `return losses.mean()` returns a 0-dim tensor. Fix: `.item()`.
  - `make_lr_lambda -> float` (from stage 9) was already wrong (returns a closure). Then changed to `-> Float32`, which is doubly wrong: (a) wrong target type (closure, not tensor), (b) invalid jaxtyping syntax (`Float32` needs `[Tensor, "..."]`). Correct: `-> Callable[[int], float]` from `typing`.
  - Tutor mistake: I named the variance-reduction parameter `B_eval` initially, which read as "eval batch size" but meant "number of eval batches." Confusing. Renamed to `eval_iters` (nanoGPT convention). Unforced naming error on my part.

- **Imprecision in "why `model.eval()`"**: initial framing was "removes dropout so we have full power of the model." The "full power" framing is slogan-shaped — not wrong in spirit but misses the load-bearing reason. The substantive answer: makes the forward *deterministic* (same input → same output across calls), so loss values are reproducible and reflect actual model state, not dropout noise. The next-level imprecision: "dropout zeros params" was wrong — dropout zeros *activations* during forward, not parameters.

- **"Restore grad to zero" for eval cleanup** — wrong. `optimizer.zero_grad()` is part of the next training step, not the eval cleanup. The thing to restore after `model.eval()` is `model.train()`. Missing this would silently disable dropout for the remainder of training — invisible until the train-val gap drifts unexpectedly. Critical fix.

- **Slogan-vs-mechanism pattern, again.** Stage 10 surfaced two more instances:
  - `model.eval()` framed as "full power" instead of "deterministic." Slogan was close enough to fool the speaker but missed the actual mechanism.
  - "Cosine is smooth from 1 to 0" (carried over from stage 9) as the cosine-distinguishing property. Smoothness is necessary but not sufficient — what matters is `η'(0) = η'(T) = 0` (flat endpoints). Smoothness alone is shared with linear, exponential, polynomial decays.

- **Predict-then-check, finally engaged.** Stage 9's chronic weakness. Stage 10 broke the pattern: Denis predicted in writing before running ("step 0 gap ≈ 0, step 5000 small gap, no U yet"), then ran, and **all three predictions hit**. The shift wasn't pedagogical victory — the predictions were relatively easy to make given stage 9's grounding. But it does establish the discipline.

## Self-quiz

1. **Sequential vs random-window split.** Why is the sequential split preferred for character-level corpora with no document boundaries? At what corpus length / T_max would the seam overlap matter? Construct a case where a random-window split would silently leak.

2. **The four pieces of the eval idiom.** State precisely what `model.eval()`, `torch.no_grad()`, "average over `eval_iters` batches", and `model.train()` (after eval) each accomplish. For each, name a bug that arises if you omit it.

3. **Variance of the eval-loss estimate.** Derive how `Var[avg loss]` scales with `eval_iters`. What governs the *per-batch* variance? Why is `eval_iters=20` a reasonable choice for `B=64, T=256` on this corpus?

4. **The train-val gap as a diagnostic.** Your gap stabilized at 0.12-0.14 from step ~2600. Interpret this number. What's a gap that would alarm you, and at what gap would you tighten regularization (and by what mechanism)?

5. **Eval-loss minimum vs final.** Your minimum val was 1.6834 at step 4600; final was 1.6910 at step 5000. Explain why the minimum and final can differ. In what scenarios does the difference become large (>0.2)? Write the early-stopping logic in pseudocode (which checkpoint do you save, when do you stop training?).

6. **Why no U-curve in 5000 steps.** Given your config (`weight_decay=0.1`, `dropout=0.1`, B=64, T=256), predict at what step count or under what regularization changes you'd expect to see val loss start rising. Connect to data size, parameter count, and effective capacity.

7. **Initial-loss sanity check.** Step 0 train=4.2006, val=4.2002 ≈ log V. What would you conclude if step-0 val were significantly lower than log V (e.g., 3.0)? What if it were significantly higher (e.g., 7.0)? Each scenario implicates a specific bug in either the init or the data pipeline — name them.

8. **The chronic-vs-acute distinction in eval cleanup.** Missing `model.train()` after eval is a subtle bug because the symptom (gap drift) is delayed by hundreds of training steps. Compare to missing `model.eval()` before eval (acute: noisy eval loss). Which class of bug is harder to catch and why?

## What this enables

- **Stage 11 (sampling: greedy → temperature → top-k → top-p).** You now have a trained model whose qualitative output you can inspect. Greedy decode is the trivial autoregressive loop over `gpt(x)` (the `targets=None` path of forward); temperature/top-k/top-p generalize from there. The eval loss alone tells you the model's predictions are good; sampling tells you whether the predictions translate to coherent text.

- **Stage 12 (KV cache).** Needs a trained model for benchmarking. Will use the stage-10 training run as the baseline for "speedup vs naive autoregressive generation" measurements.

- **Stage 13 (RoPE).** Replaces the absolute positional embedding with rotary. Retraining will use exactly stage-10's loop, with the embedding swap as the only architectural change. The train-vs-val curve shapes from stage 10 are the comparison baseline.

- **Stages 14/15 (SwiGLU, GQA, optional).** Local swaps inside MLP/attention. Same training loop. Compare against stage-10's `(1.55, 1.69)` train/val numbers as the baseline.

- **Methodologically.** Stage 10 establishes the *measurement* infrastructure for all subsequent comparisons: hyperparameter sweeps, architectural ablations, optimizer variants. From here on, every claim of "configuration X is better than Y" is grounded in eval loss, not training loss. The discipline introduced here — predict the eval curve before running; save the eval-minimum checkpoint; track train-val gap as a diagnostic — is the difference between "running training runs" and "doing ML experimentation."

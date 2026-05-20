# Stage 10 — Eval Function Implementation

## 2026-05-20

## What I worked on
Implementing the eval loop for stage 10: 90/10 sequential split on the encoded tensor, an `eval_loss(...)` function, and its integration into the training loop with periodic eval every 200 steps.

## Key concepts
- **Sequential split** at `int(0.9 * len(encoded_text))`. Two `TokenizedDataset` instances from disjoint tensor slices `[:N]` and `[N:]`. No leakage by construction. No `math.floor`/`ceil` tricks needed.
- **Eval idiom** (four pieces): `model.eval()` → `with torch.no_grad():` → loop `eval_iters` times sampling fresh val batches → `model.train()` to restore.
- **`model.eval()`** disables dropout → forward is *deterministic*. Without this, loss measurements include dropout noise confounded with real signal.
- **`torch.no_grad()`** disables autograd graph construction. Saves memory (no graph stored) AND compute (no graph-building ops during forward).
- **`model.train()` restoration** is critical. Forgetting it leaves dropout disabled for the remainder of training — silent bug, gap drifts.

## What I got wrong
- **`math.ceiling`** doesn't exist. Python's `math` module has `math.ceil`. Surfaced as `AttributeError` immediately. Typo or autocomplete guess.
- **Unnecessary `+1` gap** in the split. Wrote `encoded_text[math.ceil(0.9*L) + 1:]` for val, creating a 2-character no-man's-land. Defensive but redundant — `[:N]` and `[N:]` already give disjoint index pools because the val pool's first index is exactly the train pool's *next* index. Simpler is correct.
- **Reported "done" after the split was set up but before writing the eval function**. The loop still printed only train loss. `ds_val` was created and never used anywhere in the file. Tutor caught it. "Done" should mean "the feature works end-to-end" — structural changes alone aren't enough.
- **Critical eval-function bug**: sampled the val batch *outside* the loop, then re-ran the *same* (x, y) through the model `eval_iters=20` times. With `model.eval()` (dropout off), the forward is deterministic, so `losses[0] == losses[1] == ... == losses[19]`. The "average" was the loss on one batch, paid for 20× the compute. Defeats the whole point of `eval_iters` (variance reduction by averaging *different* batches). Tutor caught it by reading the loop body and asking "what changes between iterations?"
- **Type annotation lie**: `eval_loss -> float` but I returned `losses.mean()`, a 0-dim tensor. Tutor flagged. Fix: `.item()` on the return. The annotation should match what's actually returned.
- **"Fixed" `make_lr_lambda` instead of `eval_loss`**: misread the tutor's pointer. Changed `make_lr_lambda -> float` to `-> Float32`. Doubly wrong: (a) `make_lr_lambda` returns a *closure*, not a tensor; correct annotation is `Callable[[int], float]` from `typing`; (b) `Float32` alone (without `[Tensor, "..."]`) is not valid jaxtyping syntax — jaxtyping types require a tensor type and shape spec. The tutor was talking about `eval_loss`; I touched the wrong function.
- **Imprecise "why `model.eval()`"**: said "removes dropout so we have full power of the model." The "full power" framing is the slogan version. Substantive reason: *makes the forward deterministic*. Same input → same output across calls → reproducible loss measurement. The correction also surfaced an adjacent imprecision: "dropout zeros params" was wrong; dropout zeros *activations* during forward, not parameters.
- **"Restore grad to zero" as eval cleanup** — wrong. `zero_grad()` is part of the next training step, not eval cleanup. The thing to restore after `model.eval()` is `model.train()`. Forgetting this is the worst class of bug because it's silent — the next training iteration runs with dropout disabled, gradients keep flowing, loss keeps decreasing — but regularization just turned off. The symptom (gap drift) is delayed by hundreds of steps from the cause.
- **Silent `weight_decay` change** from 0.0 (stage 9) to 0.1 (stage 10) between editing sessions. Stage 9's curve was generated with WD=0.0; stage 10's with WD=0.1. The two runs aren't directly comparable on train loss anymore. Config drift is the easiest way to make experiments incomparable.

## Why this works
- **`model.eval()` makes the forward deterministic** because dropout is the only stochastic operation in our model (no BatchNorm). Deterministic forward → same input gives same output → loss measurements are reproducible, not contaminated by dropout-mask noise. The point is *measurement*, not "full power."
- **`no_grad()` saves both memory and compute**, not just one. Memory: no autograd graph stored. Compute: graph-building ops during forward are skipped. Both compound; this is why eval is dramatically faster than training of the same input.
- **`model.train()` restoration matters silently**. If you forget, training continues with dropout off → less regularization → gap widens → looks like overfitting but is actually a config bug. Train loss might even drop *faster* without dropout, which makes the bug look like a success at first glance. This is the worst kind of bug: silent, delayed-symptom, and looks superficially fine.
- **Variance reduction by averaging**: `Var[mean of K i.i.d. samples] = Var[single sample] / K`. For K=20, ~20× variance reduction on the eval-loss estimate. Each batch is a Monte Carlo sample of `E[loss | val distribution]`; averaging trades compute for precision in the estimate.

## Open questions
None right now — stage 10 done. Stage 11 (sampling) is next. Watch-items: will the slogan-vs-mechanism pattern recur on "temperature controls randomness" (precise: scales logits before softmax; `T<1` sharpens, `T>1` flattens), "top-k truncates the distribution" (precise: masks all-but-top-k logits to `-inf`), and "argmax is deterministic" (precise: deterministic given the same logits)? Bet: at least one of these will get slogan'd.

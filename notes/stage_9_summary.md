# Stage 9: Training Loop

## Summary
Built the end-to-end training machinery in `src/train.py`: AdamW optimizer with decoupled weight decay, cross-entropy loss via the `(logits, loss)` path of `GPT.forward`, a `LambdaLR` scheduler implementing linear warmup followed by cosine decay, gradient clipping, and a six-operation training step (`zero_grad` → forward → backward → `clip_grad_norm_` → `optimizer.step` → `scheduler.step`) looping over fresh mini-batches from `TokenizedDataset.get_batch`. Verified end-to-end on TinyShakespeare with B=64, T=256, lr=1e-3, warmup=100, total=5000, cosine to 10% — training loss dropped from log V ≈ 4.20 to **1.59** over 5000 steps (~10m24s wall-clock on M4 Pro / MPS, ~80M training tokens). This is the machinery; stage 10 is the actual training experiment with held-out evaluation.

## The math

**AdamW update** (the key derivation of this stage). Per parameter θ, with gradient `g_t = ∂L/∂θ` at step t, EMAs of the first and second moments:

```
m_t = β₁·m_{t-1} + (1-β₁)·g_t                    (first moment, m_0 = 0)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²                    (second moment, v_0 = 0)
```

**Bias correction.** With `m_0 = v_0 = 0`, the raw EMAs are biased toward zero at small t. Unrolling and using `E[g_t] = μ`:

```
E[m_t] = (1-β₁) · μ · Σ_{i=1}^t β₁^{t-i} = (1 - β₁^t) · μ
```

So the unbiased estimator is `m̂_t = m_t / (1 - β₁^t)` (and analogously `v̂_t = v_t / (1 - β₂^t)`). At t=1: `m̂_1 = g_1`, `v̂_1 = g_1²` — the unbiased one-sample estimates. As t → ∞, the correction factor → 1 and `m̂_t → m_t`.

**Variance of v̂_t** (the warmup justification). Write `v̂_t = Σ w_i · g_i²` with `Σ w_i = 1` after bias correction. For i.i.d. `g_i²` with variance V:

```
Var[v̂_t] = V · Σ w_i² ≈ V · (1-β₂) / (1+β₂)  ≈  V · (1-β₂) / 2  for β₂ ≈ 1
```

Effective sample size: `N_eff = 2/(1-β_2)`. For β₂=0.999 → N_eff ≈ 2000. So `v̂_t` becomes unbiased-but-still-noisy at small t (Var[v̂_1] = V, full single-sample noise); converges to low-variance over ~2000 steps.

**Why noise in v̂_t causes instability** (Jensen). Adam's step magnitude is `1/sqrt(v̂_t)`. The function `1/sqrt(x)` is convex and steep near zero, so `E[1/sqrt(v̂_t)] ≥ 1/sqrt(E[v̂_t])` — expected step magnitude is *upward-biased* in proportion to Var[v̂_t]. Concretely, with true `E[g²]=1`, a noisy estimate `v̂ = 0.01` gives step magnitude 10× intended; an estimate `v̂ = 100` gives 0.1× intended. Downside unbounded, upside bounded. Warmup keeps η small during the high-variance regime so that `η · spike` stays modest.

**AdamW update** (the W is decoupled decay):

```
θ_{t+1} = θ_t − η · m̂_t / (sqrt(v̂_t) + ε) − η · λ · θ_t
```

The `−η·λ·θ_t` term is *outside* the preconditioner — every parameter decays at the same rate λ. In plain Adam with L2-in-loss, the decay term enters the gradient and gets divided by `sqrt(v̂_t)`: small-`v_t` parameters (LN γ, β) decay more, large-`v_t` parameters (attention QKV) decay less. Backwards from what you want; AdamW fixes it.

**Cosine schedule.** Multiplier (relative to base LR):

```
linear:  step / warmup_steps                                              (0 ≤ step < warmup_steps)
cosine:  min_lr_ratio + ½(1 − min_lr_ratio)(1 + cos(π · progress))         (step ≥ warmup_steps)
         where progress = (step − warmup_steps) / (total_steps − warmup_steps)
```

Continuous at the handoff: linear ends at 1; cosine at progress=0 starts at min_lr_ratio + (1 − min_lr_ratio) = 1. Cosine ends at progress=1: min_lr_ratio.

Three structural features of cosine vs alternatives: **flat at start** (η'(0)=0, preserves peak-LR window), **flat at end** (η'(T)=0, long fine-tuning tail), monotonic on [0, T]. Linear, exponential, polynomial decays are all smooth but lack one or both flat endpoints — they erode the exploration window from step 1 or end abruptly.

**Initial loss prediction = log V.** Tied embeddings at N(0, 0.02) → tiny logits → softmax ≈ uniform → cross-entropy on correct class ≈ `-log(1/V) = log V ≈ 4.17` for V=65. Empirically: 4.20. ✓ (Predicted at stage 8, observed live at stage 9 step 0.)

## The code
- `src/train.py` — new file. Loads tokenizer + dataset, builds GPT on MPS, instantiates `torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)`, constructs the warmup-then-cosine `LambdaLR` via a `make_lr_lambda(warmup_steps, total_steps, min_lr_ratio)` factory closure, then runs the six-operation training step inside a 5000-iteration loop with per-batch `ds.get_batch(B, T)` and `.to(device)`. Periodic print of step / LR / loss every 100 iterations.

No tests added. `src/train.py` is a script, not a library module; the integration test is the training run producing decreasing loss.

## Design choices and why

- **AdamW over plain Adam** (decoupled weight decay). Loshchilov & Hutter 2019. Plain Adam with L2-in-loss has the decay term `λθ` enter the gradient, which then gets divided by `sqrt(v̂_t)`. Result: parameters with large recent gradients (attention QKV) decay *less* than parameters with small recent gradients (LN γ, β) — backwards from what's wanted. AdamW applies decay outside the preconditioner so every parameter decays at the same rate. Empirically beats plain-Adam-with-L2 on transformers consistently.

- **Linear warmup + cosine decay schedule.** Two motivations stacked:
  - **Warmup**: `v̂_t` has high variance at small t (single-sample estimate of E[g²]). High Var[v̂_t] + convex `1/sqrt(x)` (Jensen) → expected step magnitude upward-biased → occasional huge updates → instability. Warmup keeps η small until v̂_t stabilizes (effective settling time ~`1/(1-β₂)` ≈ 1000 steps; warmup of 100 is the conservative end). Without warmup, peak LR has to be much lower (we observed `lr=1e-1` diverging at one batch).
  - **Cosine decay (vs linear, step, exponential)**: cosine is the unique simple smooth function on [0, T] with both endpoints flat (η'(0) = η'(T) = 0). Flat top → exploration phase gets the full peak-LR budget for ~10-20% of total steps before decay erodes it. Flat bottom → long fine-tuning tail at small LR for precise basin convergence. Linear erodes both ends; step decay has sharp kinks that destabilize the optimizer; exponential is steepest at start (opposite of what cosine offers).

- **Gradient clipping at `max_norm=1.0`.** Bounds the global L2 norm of all gradients. Without it, occasional large gradients (from rare tokens, attention saturation, etc.) can blow the model out of a useful region of parameter space — the kind of single-step disaster that no amount of moment-averaging can recover from. Must be applied *after* backward (gradients exist) and *before* `optimizer.step()` (so the step uses clipped values). Putting it after step is a no-op.

- **Six-operation training step**: `zero_grad → forward → backward → clip → optimizer.step → scheduler.step`. Each operation's role:
  - `zero_grad`: clears `.grad` (PyTorch accumulates by default — legacy of RNN BPTT). Position in the loop doesn't matter; *calling it at all* matters. Omitting it would accumulate gradients across iterations → unbounded gradient growth.
  - `forward`: builds the autograd graph; produces loss as a scalar tensor with `grad_fn`.
  - `backward`: walks the autograd graph, populates `.grad` on every parameter.
  - `clip_grad_norm_`: rescales `.grad` in place if global L2 norm exceeds threshold.
  - `optimizer.step`: applies AdamW update using `.grad`; reads `param_groups[i]['lr']` for η.
  - `scheduler.step`: advances `param_groups[i]['lr']` for the next iteration.

- **Hyperparameters** (after empirical sweep):
  - **lr=1e-3** (peak), **5000 total steps**, **warmup=100**, **cosine to 10% of peak**. The 3e-4 + 1000 steps initial guess plateaued at unigram baseline (loss ~3.30) because the model never escaped letter-frequency learning; bumping to 1e-3 + 5000 steps unlocked attention circuits (phase transition at step ~500) and brought training loss to 1.59.
  - **B=64, T=256**. Earlier B=2 produced very noisy gradient (loss oscillating in 0.3 range step-to-step); B=64 smoothed the curve dramatically (oscillation < 0.05).
  - **weight_decay=0.0**. Tentative — typical transformer training uses 0.1, with separate param groups excluding LN γ/β and biases from decay. Skipped for stage 9 to keep things minimal; revisit in stage 10 if overfitting becomes the bottleneck.
  - **betas=(0.9, 0.999), eps=1e-8**. PyTorch defaults; matched to the math we derived.

- **Factory closure for the scheduler** (`make_lr_lambda`). Wraps the linear+cosine logic in a closure that captures `warmup_steps`, `total_steps`, `min_lr_ratio` by argument (not by lexical scope of `__main__`). Idiomatic in ML code; keeps the dependencies of the closure explicit and testable in isolation. The alternative — defining the consts in `__main__` and letting the closure capture them lexically — works for a single-script run but breaks down for reusability.

- **LR communication channel between scheduler and optimizer.** The scheduler mutates `optimizer.param_groups[i]['lr']` *in place* at each `scheduler.step()`. The optimizer's `step()` reads the same `'lr'` key when computing the AdamW update. One dict, two consumers; the scheduler writes, the optimizer reads. (Verified during debugging by printing `optimizer.param_groups[0]['lr']` inside the loop.)

## Errors and corrections

- **`from sched import scheduler`** — bogus stdlib import from IDE autocomplete. Python's `sched` module is a generic event scheduler unrelated to PyTorch. Was kept around for several iterations because the corresponding `scheduler.step()` call later wouldn't have errored until the device mismatch above it cleared. Removed.

- **Device mismatch**: model moved to MPS via `model.to(device)`, but `x, y = ds.get_batch(B, T)` returned CPU tensors. The forward pass through `tok_emb(ids)` tried to index a MPS weight tensor with CPU indices → `RuntimeError: Placeholder storage has not been allocated on MPS device!`. Fix: `x, y = x.to(device), y.to(device)` after `get_batch`. Lesson: model + batch must travel together; the embedding's gather op is where the mismatch surfaces because that's the first op consuming `ids`.

- **Weight decay sign error in derivation**: wrote `θ_{t+1} = θ_t − η · adam_step + η · λ · θ` (plus sign), which would amplify θ at every step rather than decay it. Caught and corrected to `− η · λ · θ`. Lesson: the L2 penalty has gradient `+λθ`, which enters the parameter update as `−η · (+λθ) = −η · λ · θ`. The minus sign comes from gradient *descent*.

- **AdamW vs Adam form-mismatch in derivation**: verbally claimed "Adam applies decay directly into the gradient" but wrote a formula where decay was added *outside* the Adam preconditioner. Verbal and formula disagreed. The correct phrasing: (A) plain Adam with L2-in-loss puts `λθ` *inside* the moment accumulators → divided by `sqrt(v̂_t)` → differential decay; (B) AdamW applies decay *outside* the preconditioner → uniform decay per parameter. Three messages to converge on the clean statement.

- **Cosine formula typo**: `0.5 ** (1 - min_lr_ratio)` instead of `0.5 * (1 - min_lr_ratio)`. `**` is exponentiation, not multiplication. With `min_lr_ratio=0.1` this evaluated to `0.5^0.9 ≈ 0.536` instead of `0.5 * 0.9 = 0.45`, producing a discontinuity at the warmup→cosine handoff (multiplier jumped from 1.0 to 1.17). Not catastrophic in effect — peak LR was 17% higher than configured — but a real bug. Surfaced by the explicit sanity check: "at `step=warmup_steps`, what should multiplier be?" Both branches should give exactly 1; the typo broke that.

- **Scheduler instantiation oversight**: `scheduler = torch.optim.lr_scheduler.LambdaLR` assigned the *class* to the variable, not an instance. No `(...)` call, no closure passed. Subsequent `scheduler.step()` would have errored had it been called — but it wasn't called in the loop either, so the bug stayed silent. Lesson: when a class assignment doesn't crash but doesn't do what you want either, the loop body needs to actually exercise the object.

- **"Loop" with one iteration**: initial `src/train.py` ran the training step once and exited. "It works" was reported when the script didn't crash; the training-loop smoke test (overfit one batch) requires looping ≥50 times and watching loss decrease. Lesson: "doesn't crash" ≠ "works." The substantive test is the *trajectory*, not the single-step.

- **lr=3e-4 + 1000 steps + B=2 → unigram plateau**. Initial conservative config plateaued at loss ~3.30 (slightly worse than empirical unigram entropy of TinyShakespeare). Diagnosis: B=2 has very noisy gradient; 1000 steps is too few for the model to escape the unigram basin; lr=3e-4 is the canonical *streaming* LR for big models, not small-model TinyShakespeare. Fix: B=64 (10× lower noise), lr=1e-3 (3× higher), total=5000 (5× longer). Plateau broke at step ~500 with a sharp phase transition (3.25 → 2.83 in 100 steps) as attention circuits formed.

- **"It works" / "doesn't learn" / "it decreases" hand-waves.** Multiple times reported subjective summaries without numbers. Each one required the tutor to push back for the actual trajectory. Lesson: in training, *loss numbers are the primary observable*. Subjective summaries hide more than they reveal — a "decreasing" loss could be `4.2 → 4.19` (LR too low) or `4.2 → 0.01` (overfit one batch perfectly), and these have completely different implications.

- **"Maybe because it smoothly goes from 1 to 0"** for cosine schedule justification — collapsed three independent properties (smoothness, flat-at-start, flat-at-end) into one slogan. Walked back via explicit counter-examples: linear, exponential, polynomial decays are all smooth but lack flat endpoints. Cosine is distinguished by *endpoint derivatives*, not just smoothness. Recurring pattern with the stage-8 "ModuleList registers in grad graph" — collapsing a precise mechanism into a closest-fit slogan.

- **NGD framing for Adam** — mostly right but slightly imprecise. Adam's `1/sqrt(v̂_t)` is closer to RMSprop-with-momentum than to literal natural-gradient descent (NGD's preconditioner is `F^{-1}`, not `diag(F)^{-1/2}`). The diagonal Fisher approximation and the sqrt are both compromises. Useful intuition; not the literal math.

- **Forgot "why log V"** in the criterion-2 walkthrough — couldn't immediately recall the derivation despite having seen its empirical confirmation (loss=4.25 ≈ log(65)=4.17) two stages earlier. Re-derived in two lines once prompted. Embarrassing because the empirical observation never produced the conceptual lock-in.

- **Predict-then-check skipped, repeatedly.** Stage 9's chronic issue. Across multiple experiments (LR sweep at T=4, T=256; 1000-step run with B=2; 1000-step run with B=64; first 5000-step run), Denis ran first and described after. The tutor pushed back each time, sometimes accepting once and continuing, sometimes flagging the pattern as a known weakness. CLAUDE.md is explicit that predict-then-check is the criterion for "I understand what should happen" — skipping it means flying blind. The pattern was named going into stage 10, where eval-vs-train loss divergence will *only* be diagnostic if you've predicted what the gap should be.

## Self-quiz

1. **AdamW vs Adam.** Derive the difference between (A) "L2 penalty in the loss" and (B) "decoupled weight decay applied after the Adam update." For a parameter with large `v̂_t` (e.g., attention QKV) and one with small `v̂_t` (e.g., LayerNorm γ/β), what's the *effective* decay rate under each formulation? Why does this matter empirically?

2. **Bias correction.** Starting from `m_0 = 0` and `m_t = β_1·m_{t-1} + (1-β_1)·g_t`, derive `E[m_t]` assuming `E[g_t] = μ` and i.i.d. gradients. Show the unbiased estimator is `m̂_t = m_t / (1 - β_1^t)`. At what step does the correction factor reach 99% for β_1=0.9? For β_2=0.999?

3. **Warmup justification.** Why does Var[`v̂_t`] matter for stability, not just E[`v̂_t`]? Use Jensen's inequality on `1/sqrt(x)` to argue that high Var[`v̂_t`] biases the expected step magnitude *upward*. Derive the effective sample size of `v̂_t` for β_2=0.999.

4. **Cosine schedule justification.** Among smooth monotone-decreasing functions from η_max to η_min on [0, T], what specifically distinguishes cosine? Why are flat endpoints (η'(0) = η'(T) = 0) load-bearing rather than incidental? What does "smooth" alone fail to give you that cosine does?

5. **Training step structure.** Write the six operations in order. For each, state (a) its purpose, (b) what depends on it being in this position. Where does `optimizer.zero_grad()` go and why doesn't its position-in-loop matter as long as it's called? What happens if `clip_grad_norm_` comes after `optimizer.step()`?

6. **Scheduler ↔ optimizer communication.** When you call `scheduler.step()`, what does it mutate? When `optimizer.step()` runs the AdamW update, where does it read the current LR from? Why is the right channel `optimizer.param_groups[i]['lr']` and not, say, an attribute on the scheduler object?

7. **The unigram plateau.** Why did `lr=3e-4` + 1000 steps + B=2 plateau at loss ~3.30 instead of descending further? What is loss 3.30 approximately on a character-level model — connect to the empirical unigram entropy of TinyShakespeare. Why did `lr=1e-3` + 5000 steps unlock further descent (loss 1.59), and what's the mechanism behind the "phase transition" at step ~500?

8. **Predicting train vs eval loss.** Your final training loss was 1.59 after the model had cycled through TinyShakespeare ~70 times (B=64 × T=256 × 5000 steps ≈ 80M training tokens, corpus has ~1.1M unique tokens). Predict eval loss on a held-out split. Why is the gap diagnostic of memorization vs generalization? What signals would tell you the model is overfitting?

## What this enables

- **Stage 10 (TinyShakespeare actual run + eval split + qualitative samples).** Adds a train/val split to `TokenizedDataset` (e.g., 90/10), tracks eval loss separately at each `eval_interval` (model.eval() + no_grad), and inspects sample quality. The training machinery from stage 9 is the input; the only new code is the eval hook and a sampling probe. The big learning is *eval-vs-train divergence* — exactly what the predict-then-check discipline (now flagged as weak) is for.
- **Stage 11 (sampling: greedy → temperature → top-k → top-p).** Wraps `gpt(x)` (the `targets=None` path of forward) in an autoregressive loop. Uses the `(logits, _)` half of the two-path interface that stage 8 built. Standalone module; doesn't modify training.
- **Stage 12 (KV cache).** Mutates the attention forward inside `Block` to accept and update a per-block K/V cache. The `GPT` and `Block` outer interfaces stay unchanged; only attention's internals are touched.
- **Stage 13 (RoPE).** Replaces `LearnedPositionalEmbedding` with rotary embedding applied inside attention. Training loop from stage 9 doesn't need to change; the swap is local to the model.
- **Stages 14, 15 (SwiGLU, GQA).** Local swaps inside MLP / attention. Training loop unchanged.

Stage 9 is the longest-living piece of code in the project: every subsequent stage either runs this training loop or replaces a piece of the model it trains.

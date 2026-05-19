# Stage 10 — Hyperparameter Sweeps and Stopping Criteria

## 2026-05-20

## What I worked on
Pre-stage-10 conceptual prep: asked the tutor how to do hyperparameter sweeps properly, how low loss can go, and when to stop training. No code yet — this is the framework for stage 10's eval-loss tracking.

## Key concepts
- **Sweep methods, in order of sophistication:** grid (cartesian product, cheap setup, explodes combinatorially); random search (Bergstra & Bengio 2012 — beats grid at fixed budget because some hyperparameters matter much more than others); Bayesian optimization (Optuna, Ax — model the eval-loss surface with GP/TPE, propose by expected improvement); LR Range Test (Smith 2017 — single short run with LR linearly increasing, plot loss vs LR, pick the steepest-descent point); population-based training (DeepMind 2017 — copy + perturb online).
- **Shannon entropy floor ≈ 0.69 nats/char** for English (Shannon 1951, ~1 bit/char). Absolute lower bound on cross-entropy for *any* model on natural-language character prediction. Beneath this is memorization noise.
- **U-shaped eval-loss curve.** Train loss → 0 with memorization; eval loss bottoms out when capacity matches data, then rises as overfitting takes over. The minimum is the right checkpoint to save.
- **Chinchilla scaling** (Hoffmann et al. 2022). At fixed total compute, eval loss is minimized by scaling tokens and parameters together (~roughly 20 tokens per param, or 1:1 in some characterizations). My setup (~1.5M params, 1.1M unique tokens) is near data-token-to-param ratio 1, close to that regime for this model size.
- **Practical floor for my config** (d_model=128, n_layers=6): eval loss ~1.7-2.0 on TinyShakespeare. Stronger models (nanoGPT's d_model=384) reach ~1.4-1.5. Data is the bottleneck: 1.1M tokens of single-author archaic English limits everyone.

## What I got wrong
- **Implicit "more training = better" framing in the 24-hour question.** Correction: past the eval-loss minimum, longer training is *strictly worse* on what matters (eval loss). Pure compute → train loss going to zero (memorization). Eval loss is U-shaped, not monotone-decreasing. The whole reason early-stopping exists is that "minimize loss" and "train longer" are not equivalent in the presence of finite data.
- **"How low can loss go?" treated as having a single answer.** Correction: train loss and eval loss have different answers, and they diverge dramatically with overfitting. Train loss → 0 is achievable by memorization; eval loss has a floor determined by *data entropy + model bias*, not by training duration. The question only makes sense if I specify *which* loss.
- **Implicit assumption that hyperparameter sweeps mean grid search.** Correction: grid is the worst common choice for >3 hyperparameters. Random search is provably better at fixed compute. LR Range Test is the cheapest single-run technique and probably the highest-leverage thing to do before stage 10's main training.

## Why this works
- **Random search beats grid** because the joint hyperparameter space is high-dimensional but the loss surface depends strongly on only a few dimensions (mostly LR). Grid wastes runs exploring product structure that the loss surface doesn't have. Random allocates the same budget more uniformly to whichever dimensions actually matter.
- **U-shaped eval loss** is mechanical. Early in training, the model learns the marginal distribution → both train and eval drop. Mid-training, the model learns context-dependent structure that *generalizes* → both keep dropping, eval lagging by a constant gap. Late training, the model memorizes training-specific noise → train loss keeps falling, but eval loss has nothing left to learn that helps on held-out data, so it stalls or rises. The eval-loss minimum is the point where generalizable signal has been extracted but memorization hasn't started dominating.
- **Chinchilla scaling** result: for a fixed compute budget `C ≈ 6·N·D` (N parameters × D tokens × ~6 FLOPs/param/token), eval loss is minimized at `N ≈ D / 20` (or thereabouts depending on derivation). Doubling N or D alone is strictly suboptimal; both should grow together. Practical consequence: scaling models without scaling data wastes capacity on memorization; scaling data without scaling models wastes data on a model that can't absorb it.

## Open questions
- **What is the actual eval-loss U-curve shape for my model on TinyShakespeare?** Predicted floor ~1.7-2.0; predicted minimum location around steps 10K-50K. Need to verify in stage 10. Will be the first chance to apply predict-then-check rigorously to a non-trivial trajectory.
- **Should I do an LR Range Test before stage 10's main training?** Smith 2017's method seems like 10-15 minutes of compute and would directly justify the peak-LR choice rather than relying on "nanoGPT uses 1e-3." Maybe yes; revisit when starting stage 10.
- **Per-param-group weight decay** — LN γ/β and biases excluded (`weight_decay=0`), matrix params decayed (`weight_decay=0.1`). Standard transformer recipe. Worth adding in stage 10 if the train-eval gap grows large.
- **What's the practical wall-clock cost of a 10-config sweep on M4 Pro?** At ~10m24s per 5000-step run, a 10-config sweep is ~2 hours. Affordable but not trivial.

# Stage 8 — Walkthrough Corrections

## 2026-05-17

## What I worked on
Line-by-line walkthrough of `GPT.__init__` and `GPT.forward` during `/stage-done 8`, plus the toy-input prediction that I tried to skip.

## Key concepts
- **`nn.ModuleList` vs Python list.** ModuleList registers children into the parent's `_modules` dict, making them visible to `.parameters()`, `.to(device)`, `.train()/.eval()`, `state_dict`. Autograd is operation-level — it builds the graph from tensor ops during forward, independent of module registration. A plain list still allows gradients to compute (the ops are still there); what breaks is that `model.parameters()` doesn't include them, so the optimizer never updates them. Silent.
- **Weight tying — three complementary views.**
  - Operation: embedding does `v @ W`, lm_head does `x @ W.T`. The matrices in those two operations are transposes.
  - Semantic: rows of `W` are per-token embedding vectors. Embedding picks row `i`; lm_head dots `x` against every row, scoring "how aligned is `x` with each token's embedding." Same vectors, two roles.
  - Storage: one tensor in memory. The `.T` in `nn.Linear.forward` is a view (no copy, no FLOPs) because PyTorch stores `weight` as `(out, in)` = `(V, d_model)`, matching the embedding's `(V, d_model)`.
- **Tying saves memory, not compute.** Same forward and backward FLOPs; saves `V·d_model` params + Adam's `2·V·d_model` moment floats. For GPT-2 small that's ~460MB. Step-time speedup only indirect (larger batch, no OOM).
- **Initial loss derivation.** Tied weights at N(0, 0.02) → logits with std ≈ √(d_model · 0.02²) ≈ 0.23 → softmax ≈ uniform → `-log(1/V) = log V`.

## What I got wrong
- **"ModuleList registers in the grad graph."** Wrong axis. Grad graph is operation-level; ModuleList registration is module-level (`_modules`). Conflated "autograd needs to see all the gradient flow" with "modules need to be registered for autograd to see them." Autograd doesn't need module registration — it follows tensor ops. What ModuleList enables is `.parameters()` *discovering* the children, which is what the optimizer (not autograd) needs.
- **"Weight tying — the modules become transposes of each other."** Incomplete, not wrong. Operation-level true. But the load-bearing point is semantic (rows of W *are* the per-token vectors used both as representations and as classification templates) and the cheapness is storage-level (one Parameter, `.T` is a view). I collapsed three levels into one slogan.
- **"Tied weights speed up training."** Wrong. Same FLOPs forward and backward. Saves memory (params + Adam state). The intuition "fewer params → faster" applies when fewer params means *smaller operations* (smaller d_model). With tying, the operations are unchanged; only the count of distinct Parameter objects changes. Step-time speedup is indirect via memory.
- **Forgot why log V.** Couldn't recall the derivation when asked, even though I'd seen its empirical confirmation (loss=4.25 ≈ log(65)=4.17) two debugging steps earlier. Math: small init → softmax(near-zero noise) ≈ uniform → cross-entropy on correct class = `-log(1/V) = log V`. Two lines. Empirically observed without internalizing the math.
- **`.view(V, -1)` in the explanation** when the code is `.view(-1, V)`. The former gives `(V, B·T)`, wrong for cross-entropy's `(N, C)`. Paraphrase typo only — code was right — but reveals that "first dim, second dim" isn't yet automatic in my head.
- **Tried to skip criterion 3 toy prediction** ("seems too much for a mental exercise") after answering only the shape question. CLAUDE.md push-back trigger fired; tutor refused, simplified the remaining questions, made me answer. Pattern: criterion 3 is the most uncomfortable because it requires *predicting* the math, not reciting it post-hoc.

## Why this works
- **Three-view weight-tying framing is load-bearing.** Each level answers a different question: operation level — what does it compute; semantic level — what does it mean (inductive bias: token vectors as classification templates); storage level — what does it cost (free, because of PyTorch's `(out, in)` convention). A single-level framing ("it's a transpose") loses two-thirds of the picture.
- **Compute-vs-memory distinction is general.** Any "optimization" claim should specify which axis it touches. FLOPs are set by operation sizes, not parameter count per se. Same principle applies to LoRA (saves memory + speeds optimizer step, not forward), quantization (memory + bandwidth, not necessarily compute on hardware without int kernels), etc. Worth carrying forward.
- **Push-back at criterion 3.** The prediction probe is the test of whether the math has been internalized. Skipping because "it's too much" means it hasn't been — exactly what CLAUDE.md's "let me just move on" trigger exists to catch. The tutor not letting me skip is the protocol working as designed.

## Open questions
- **Recurring "vague-correct slogan replaces precise mechanism" pattern.** Three stages now: stages 2-7 "super().__init__() inherits parent's parameters/methods" (initializes parent *instance state*: `_parameters`, `_modules`, `_buffers` dicts); stage 8 "ModuleList registers in the grad graph" (registers in `_modules`, not the autograd graph); stage 8 "tied weights speed up training" (saves memory, not compute). Hypothesis: I reach for the closest-fitting slogan from training-data-shaped intuition instead of mechanistically tracing what each component actually does. Watch in stage 9 for "AdamW tracks running averages" (precisely: first and second moment estimates with bias correction at step *t*) and "clip_grad_norm_ limits the gradient" (precisely: rescales gradients in-place when global L2 norm exceeds threshold).

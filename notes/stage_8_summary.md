# Stage 8: Full GPT

## Summary
Assembled the full decoder-only GPT model in `src/model.py`: token + learned-positional embeddings at input, a stack of `n_layers` pre-norm `Block` instances (stage 7), a final `LayerNormalization` before the head, and a linear `lm_head` whose weight is tied to the token embedding. Includes the cross-entropy loss path conditional on `targets`. This is the canonical decoder-only architecture from GPT-2 / GPT-3; everything stages 9-15 will modify or wrap is this model (training loop, sampling, KV cache, RoPE migration).

## The math

**Forward**:
```
x = tok_emb(ids) + pos_emb(positions)           # (B, T, d_model)
for block in blocks: x = block(x)                # pre-norm two-phase residual update per block
x = final_ln(x)                                  # normalize residual stream before head
logits = lm_head(x)                              # (B, T, V), where lm_head.weight is tied to tok_emb's
```

**Initial loss derivation** (the load-bearing sanity-check for stage 8):
- Tied embedding init at N(0, 0.02). At initialization, `lm_head.weight = tok_emb.weight` has small entries.
- For any input `x` to the head, `logits = x @ W.T` where `W` is (V, d_model). Variance per logit ≈ `d_model · (1 · 0.02²) ≈ 0.05`, std ≈ 0.23. So logits are tiny perturbations of 0.
- `softmax(tiny noise)` ≈ uniform over V classes: `P(class i) ≈ 1/V` for every i.
- Cross-entropy on the correct class: `-log P(correct) ≈ -log(1/V) = log V`.
- For V=65: log(65) ≈ 4.17. Empirically: 4.25. ✓

Without the std=0.02 init, default N(0, 1) on `nn.Embedding.weight` propagates to `lm_head.weight` via tying. Logit variance becomes `d_model · 1² = 128`, std ≈ 11.3. Initial loss ~70 (the bug that surfaced before the fix).

**Weight tying** as a mathematical/operational/storage triple:
- **Operation**: embedding does `v @ W` (one-hot `v`); lm_head does `x @ W.T`. The matrices in these two operations are transposes of each other.
- **Semantic**: rows of `W` are per-token embedding vectors. Embedding picks row `i`; lm_head dots the hidden state against every row, scoring "how aligned is `x` with each token's embedding." Same vectors, two roles.
- **Storage**: `id(lm_head.weight) == id(tok_emb.tok_emb.weight)` — one tensor in memory. The `.T` in `nn.Linear.forward` is a view (stride manipulation, no copy, no compute). PyTorch's `(out, in)` storage convention for `nn.Linear.weight` happens to match the embedding's `(V, d_model)`, so tying is free.

## The code
- `src/model.py` — `GPT(V, T_max, n_heads, d_model, n_layers, d_ff=None, dropout=0.1)` class added below the stage-7 `Block`. Stores `self.V`, holds `tok_emb`, `pos_emb`, `nn.ModuleList` of blocks, `final_ln`, `lm_head`. Tying + N(0, 0.02) re-init in `__init__`. Two-path forward (logits-only vs (logits, loss)) in `forward`.
- `tests/test_model.py` — five tests for stage 8: shape/dtype; return-path with parametrize over `return_loss` covering both `targets=None` and `targets=y` (including a backward + `p.grad is not None` check on all parameters); initial loss within `[0.8 log V, 1.2 log V]`; end-to-end causality at the GPT level with `eval()` and an `any(...)` reroll loop; parameter count with stacked parametrize over `d_ff ∈ {128, 256, 512, 1024}` and `n_layers ∈ {4, 5, 6, 8}` (16 combos). 66 tests in the suite total.

## Design choices and why

- **Pre-norm + final LN before the head.** Pre-norm leaves the residual stream itself never normalized between blocks; only the inputs to attention/MLP are normalized inside each block. The residual stream's magnitude can drift upward with depth. Without `final_ln`, the head sees a high-variance stream → logits with large magnitude → softmax saturation → flat gradients in many directions during training, and degenerate (essentially argmax-only) distributions at inference even with temperature scaling. `final_ln` restores unit scale per coordinate before the head.

- **Weight tying** between `tok_emb.tok_emb.weight` and `lm_head.weight`. Imposes the inductive bias "a token's embedding vector IS its classification template" — strong, semantically coherent constraint. The cheap side: saves `V · d_model` params + Adam's `2 · V · d_model` moment floats (memory savings, **not compute** — same FLOPs forward and backward; gradients accumulate into the shared tensor). For GPT-2 small (V=50k, d_model=768) that's ~460MB; for our toy (V=65, d_model=128) negligible. Speed-up only comes indirectly via larger viable batch size or unblocking OOM.

- **N(0, 0.02) re-init on the tied embedding** after the tying assignment. The fix order matters: PyTorch's default init on `nn.Embedding` is N(0, 1); after tying, that variance propagates to the lm_head. Re-init with std=0.02 keeps initial logits at unit-or-sub-unit scale, so the initial loss is `log V`, not `~70`. GPT-2's empirical choice.

- **`nn.ModuleList` over plain Python `list`** for `self.blocks`. ModuleList registers children into the parent's `_modules` dict, making them visible to `.parameters()`, `.to(device)`, `.train()/.eval()`, and `state_dict`. With a plain list, gradients would still compute during backward (autograd doesn't care about module registration — it follows tensor operations), but the optimizer wouldn't see those parameters via `model.parameters()`, so they'd never update. Silent bug: model "runs" but never learns the block layers.

- **`bias=False` on `lm_head = nn.Linear(d_model, V, bias=False)`**. Convention, NOT a requirement for tying — tying touches `weight`, not `bias`. Could keep bias and still tie. Reasons to drop: (a) pre-norm's LN-β absorbs much of bias's role; (b) empirically no benefit at scale (LLaMA/Mistral/Qwen convention); (c) cosmetic uniformity with the MLP (also bias=False).

- **`device=ids.device` in `torch.arange(T, ...)`** when constructing position indices. Without it, `torch.arange` defaults to CPU. If `ids` is on MPS/CUDA, `tok_emb(ids) + pos_emb(positions)` would error on device mismatch. Cheap defensive idiom.

- **Two-path forward**: `(logits,)` when `targets is None`; `(logits, loss)` otherwise. Cleaner than always returning loss (sampling/inference paths don't need it) or always returning logits (training paths would need to recompute loss externally). Test the path-discrimination assertion via shape check on the bare-call branch (catches accidental tuple returns through `AttributeError`).

- **`F.cross_entropy(logits.view(-1, self.V), targets.view(-1))`** flattens (B, T) sample dims because cross-entropy's primary signature is `(N, C)` — N samples, C classes. Each `(b, t)` is one classification problem; flatten gives `N = B · T`. (Cross-entropy also accepts higher-rank inputs of shape `(N, C, *)` for spatial classification, so `logits.transpose(1, 2)` against unflattened `targets` of shape `(B, T)` would also work — flatten is just the standard NLP idiom.)

- **`self.V = V` stored on the instance** (rather than reading from `logits.size(-1)`). Either works; the stored attribute is more explicit and survives refactors of `lm_head` that might change the inferred dimension. The bug that surfaced this: smoke test in `__main__` had `V = tok.vocab_size` as a module-global that the bare-V reference accidentally resolved against; tests in a different import context (no `__main__` block run) hit `NameError`. Fix was to store on self.

## Errors and corrections

- **`V` NameError in tests**: bare `V` reference in `forward` resolved to a module-global set by `__main__`'s smoke test, masking the bug locally. Tests import in a different context where that global doesn't exist. Lesson: smoke tests verify "runs in dev environment"; tests verify "runs in any import context." Fix: `self.V = V` in `__init__`, reference `self.V` in forward.

- **Loss = 0.0 with `targets=x`** (passing the input as its own target). Caused by weight tying + small init + pre-norm residual stream: at init the model is approximately identity through the residual stream, then the tied lm_head reconstructs the input embedding via row dot product. Model "perfectly predicts" the input. Fix: target the next token (`targets=y`). Educational about how tying + residual interact.

- **Loss = 70.93** with PyTorch's default N(0,1) embedding init propagating through tying. Logits had std ≈ √d_model. Fix: explicit `nn.init.normal_(..., std=0.02)` after the tying assignment.

- **Test `else` branch was vacuous**: `logits = gpt(x)` with no assertion would silently pass even if `gpt` returned a tuple. Fix: `assert gpt(x).shape == (B, T, V)` — `.shape` on a tuple raises `AttributeError`, simultaneously catching wrong-return-type and wrong-shape bugs.

- **`grad_fn is not None` is a weaker test than backward+grad-populated.** `grad_fn` proves the tensor is in *some* autograd graph; it doesn't prove the graph reaches every model parameter. A detached layer, a frozen submodule, or a broken weight tie (e.g., typo `.data = ...` instead of `.weight = ...`) all leave `grad_fn` non-None but break gradient flow. Strictly stronger: `loss.backward()` then `all(p.grad is not None for p in model.parameters())`.

- **Causality test failed under default training mode** (again — same as stage 7's Block-level test). New `nn.Module` defaults to `training=True`; dropout is active; two forward passes with identical inputs give different outputs. Fix: `gpt.eval()`. Stage 7 even called this out in its notes; it recurred in stage 8.

- **Token IDs from `100 * torch.randn(B)`** in the causality test perturbation. Conflated "random perturbation" (continuous, MLP-style) with "random token" (discrete index in `[0, V)`). Embedding errored with `IndexError: index out of range`. Fix: `torch.randint(low=0, high=V, size=(B,))` produces valid integer IDs.

- **`torch.randint(high=V-1, ...)`**: `high` is *exclusive* (Python range convention). With `high=V-1`, token `V-1` is never drawn. Fix: `high=V`.

- **Random perturbation token can equal the original**: with `randint`, ~1/V chance per batch element of drawing the same token, making the perturbation a no-op and the test vacuous. Fix: reroll-while-loop using `(x_modified[:, -1] == x[:, -1]).any()` to guarantee *every* batch element differs. (Deterministic alternatives like `(x[:, -1] + 1) % V` also valid; the loop was chosen for explicitness.)

- **`torch.equal` vs element-wise `.any()`**: `torch.equal` returns True only when shapes and all elements match; using it in the while condition with `not` semantics would exit when *any* element differed, not when *all* elements differed. Required a re-think of the predicate to use `any(...)` (element-wise) for element-wise rigor.

- **`@` missing on `pytest.mark.parametrize` decorators**: bare function calls produced Mark objects discarded into the void; pytest never saw the parametrize. Recurrence of stage 6's parametrize confusion (different flavor: that was list-of-lists vs list-of-tuples). Fix: prepend `@`.

- **Parameter count formula** had two bugs:
  - Missing the positional embedding's `T_max · d_model` term.
  - Zeroed `final_ln_param_count` (which has `2·d_model` params) with a comment about tied embeddings — the zeroing should have been applied to `lm_head_param_count` instead. Right intuition ("something is zero from tying"), wrong line. After fix: 16 parametrized configs all pass.

- **`d_ff` not passed to the GPT constructor** initially in `test_model_param_count`; constructor used MLP's default `4·d_model`, while the expected formula scaled with the parametrize value. Fix: pass `d_ff=d_ff`.

- **Walkthrough imprecisions** that needed correction:
  - "ModuleList registers in the grad graph" → wrong; autograd graph is operation-level, not module-level. Correction: ModuleList registers in `_modules` for `.parameters()`/`.to()`/`.train()/.eval()`/`state_dict` visibility.
  - "Weight tying = transpose of each other" → incomplete. Correct as one of three complementary views: operation level (transpose, fair), semantic level (same vectors in two roles), storage level (one tensor, `.T` is a view). All three should be held simultaneously.
  - "Tied weights speed up training" → wrong on FLOPs, right on memory. Same forward/backward compute; saves param + Adam state memory. Step-time speedup only comes indirectly via larger batch or avoided OOM.
  - "log V" derivation forgotten momentarily; re-derived from softmax-on-tiny-noise → uniform → cross-entropy.
  - `.view(V, -1)` written in explanation (would produce `(V, B·T)`); code is `.view(-1, V)` producing `(B·T, V)`. Paraphrase typo only.

- **Push-back trigger fired** during criterion-3 toy prediction: Denis tried to skip the prediction questions ("seems like too much for a mental exercise") after answering only shape. Per CLAUDE.md, refused. Simplified the questions (dropped the full variance trace through 6 blocks; kept order-of-magnitude + the B-vs-T uniformity probe), but did not let the criterion go uncovered. Denis then answered both follow-ups correctly.

- **Stray import** of `test_token_embedding_parameter_count` from `tests/test_embedding.py` into `tests/test_model.py`. Pytest would have re-discovered and re-run that test in the new file's context. Tests should be isolated. Deleted.

## Self-quiz

1. **Three-level view of weight tying.** Explain at the operation level, the semantic level, and the storage level. Why does PyTorch's `(out_features, in_features)` storage convention for `nn.Linear.weight` make tying *free* (zero copy, zero extra FLOPs)? What would change if `nn.Linear` had instead stored `(in, out)`?

2. **Initial loss arithmetic.** Without running anything, predict the initial cross-entropy loss for the model with default `nn.Embedding` init (no `std=0.02` re-init) at `V=65, d_model=128`. Show the chain: weight std → logit std → softmax temperature regime → expected `-log P(correct)`. The empirical answer was ~70. Is that consistent?

3. **Final LN: what pathology emerges without it?** Trace what happens to the residual stream's per-coordinate variance through 6 pre-norm blocks. Walk through what the softmax does to high-variance logits and what that implies for both training gradients and inference diversity.

4. **`bias=False` on `lm_head` — required for tying?** Defend with a counter-construction: write down (in words) a module that ties `weight` and keeps `bias`. What goes wrong, if anything? Why is the convention to drop bias anyway?

5. **`F.cross_entropy` shape variants.** The standard idiom is `cross_entropy(logits.view(-1, V), targets.view(-1))`. State an alternative that avoids the flatten by exploiting cross-entropy's higher-rank input contract. Are the two numerically identical?

6. **`grad_fn is not None` is weaker than what?** Name three concrete bugs the weaker test misses but the stronger test (`backward()` + `all(p.grad is not None)`) catches.

7. **ModuleList vs list.** A colleague replaces `self.blocks = nn.ModuleList([...])` with `self.blocks = [...]`. The forward pass still runs. The loss still calls `.backward()` without error. Loss goes down... for the embedding and lm_head only, not the blocks. Why? Trace it through `model.parameters()` → optimizer → which parameters get updates.

8. **Parameter count.** Compute (don't run) the parameter count of `GPT(V=50257, T_max=1024, n_heads=12, d_model=768, n_layers=12, d_ff=3072)` — GPT-2 small. Itemize: token emb, pos emb, per-block (attn + MLP + 2×LN), final LN, lm_head (with tying!). Compare against the canonical 124M / 125M figure and explain any delta (bias terms, gain/beta counts).

## What this enables

- **Stage 9 (training loop)**: AdamW over `model.parameters()`, cross-entropy via the `(logits, loss)` path, LR schedule (warmup + cosine decay), gradient clipping, eval/checkpoint hooks. First real device transfer (`model.to(device)` and per-batch `.to(device)`); first MPS run. Tied weights are transparent to the optimizer because `model.parameters()` deduplicates by tensor identity.
- **Stage 10 (sampling)**: wraps `gpt(x)` in an autoregressive loop with temperature, top-k, top-p. Uses the `targets=None` path that returns bare logits.
- **Stage 11 (KV cache)**: mutates the attention forward inside `Block` to accept and update a per-block K/V cache for autoregressive inference at length > 1. The `GPT` and `Block` outer interfaces stay unchanged.
- **Stage 12 / 13 / 14 (RoPE, GQA, SwiGLU)**: surgical replacements inside attention and MLP. The outer `GPT.__init__` and `forward` are the stable interface that lets these swaps happen behind it.

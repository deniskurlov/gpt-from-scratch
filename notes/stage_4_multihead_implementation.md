# Stage 4 — Multi-head Implementation

## 2026-05-13

## What I worked on
Implementing `MultiHeadAttention` in `src/attention.py` as a rank-+1 generalization of stage 3's `Attention`, plus 5 new tests mirrored from the single-head suite.

## Key concepts
- Multi-head is **a tensor-reshape on top of single-head attention**: insert one `n_heads` dim via `view + transpose`, do the standard scaled dot-product attention with one extra leading dim, then collapse heads back via `transpose + reshape`.
- **Per-head scaling factor `√head_dim`** (not `√d_model`). Same variance argument as stage 3 applied per-head: each head's Q, K are `head_dim`-dim, so `Var(Q_h · K_h) = head_dim`.
- **`W_O` is structural in multi-head** (load-bearing), vestigial in single-head. With multiple heads, each writes to a head_dim-wide subspace by default; `W_O` is the parametrization of "which direction in the residual stream does each head write to". Without `W_O`, heads are locked into their initial subspace assignments forever.
- **The mask is unchanged.** `(T, T)` bool, broadcasts implicitly against `(B, n_heads, T, T)` scores via the right-aligned-leading-dim rule. Same mask for every head and every batch element — causal structure is a per-position property, not a per-head one.
- **`view` vs `reshape` discipline**: `view` requires contiguous (works fine on `nn.Linear` output); `reshape` falls back to a copy if needed (correct choice after a transpose that produces non-contiguous data).

## What I got wrong
- **"Why multi-head" intuition gap (substantive).** Knew the "different heads learn different patterns" line but didn't *feel* why 4-8 heads is enough or why a single bigger head can't do the same. Sharpened: a single head computes ONE attention distribution per position → mathematically *forced* to weighted-average all gathered values together → outputs are smeared, downstream layers can't separately read "noun info" vs "verb info". Multi-head writes to separate subspaces, then `W_O` mixes — structurally separable. The structural separation is the inductive bias for specialization; the "feel" comes from inspecting trained attention patterns, not from theory alone.
- **`(B, T, d_model).split([n_heads, head_dim], dim=-1)`** — confused `split` (cut a dim into separate tensors) with `view`/`reshape` (refactor one dim into multiple factor-dims). The right tool was `view(B, T, n_heads, head_dim)`. Same naming-adjacency error class as `chunk` vs `split` from stage 3.
- **`x.shape[:1]` instead of `x.shape[:2]`** for `B, T = ...`. `[:1]` gives a 1-tuple; `[:2]` gives `(B, T)`. Simple slice typo that would have crashed at unpacking.
- **Forgot `self.d_model = d_model`** in `__init__` while using it in `forward`. AttributeError at first call. Caught on review.
- **`assert` before `super().__init__()`** — works functionally but breaks convention. `super()` should be the first line of every `nn.Module.__init__` so parent state initializes before any custom setup.
- **`view` confused as in-place.** `.view()` returns a new tensor (with shared storage) — not in-place. PyTorch convention: `_` suffix means in-place (`add_`, `transpose_`); no suffix means functional. `view` has no suffix → functional.
- **Recurring imprecision: `super().__init__()` "inherits parent's parameters"** in the line-by-line walkthrough — same wrong phrasing as stage 3. Correct framing: `super().__init__()` initializes parent *instance state* (the `_parameters`, `_modules`, `_buffers` OrderedDicts), not "parent parameters". `nn.Module` has no parameters of its own.
- **Toy prediction: `self.mask[:T, :T]` shape predicted as `(1, 1, T, T)`.** Actual shape is `(T, T)`. Confused the *actual* shape (what `.shape` returns) with the *broadcast-imputed* shape (how PyTorch logically treats it during arithmetic). The `(1, 1, T, T)` only exists as a broadcast pattern when interacting with a higher-rank tensor; the tensor itself is `(T, T)`.
- **Toy prediction: `scores.masked_fill(...)` output as `(B, n_heads, T, head_dim)`.** Actual is `(B, n_heads, T, T)`. Same error as the "softmax drops a dim" mistake from stage 3 — `masked_fill` is shape-preserving (only replaces values), and so is softmax.
- **`test_mha_total_parameter_count` tested only `qkv_proj.parameters()`**, not `mha.parameters()`. Name said "total" but assertion was partial. Fixed to `mha.parameters()` and the formula `4 * d_model * (d_model + 1)` (qkv_proj + out_proj).

## Why this works
- **Per-head scaling `√head_dim`.** Each head's `Q_h, K_h` are `head_dim`-dim, so `Var(Q_h · K_h) = head_dim` by the same iid argument as stage 3. Pre-softmax scores at unit variance keep softmax inputs in the non-saturated regime at initialization — gradients flow through softmax, training works. Using `√d_model` would over-scale by `√n_heads`, making softmax inputs too small and (initially) too uniform.
- **`W_O` as per-head write-direction.** `W_O` decomposes into `n_heads` `(d_model, head_dim)` blocks `W_O^h`. Block `h` paired with `W_V^h` forms head h's OV circuit: a rank-`head_dim` `(d_model, d_model)` linear map from input to written output. Without `W_O`, heads can only write to their initial concat-assigned subspace; with `W_O`, each head can learn to write to any direction in the residual stream. This is the parametrization downstream layers see when reading head h's contribution.
- **Reshape pattern preserves total elements.** `(B, T, d_model)` ↔ `(B, T, n_heads, head_dim)` is just a re-factoring of the last dim — same memory, same total count (`d_model = n_heads · head_dim`). Then transposing `n_heads` to the leading-after-B position aligns the tensor for per-head attention with the same matmul/softmax/etc. machinery, which now operates on the new leading dim independently per head.
- **Mask broadcast works because causal structure is position-dependent, not head-dependent.** Every head must obey the same "no peeking at the future" constraint. Sharing one `(T, T)` mask across heads via broadcasting is structurally correct, not a shortcut.

## Open questions
- Whether to provide a `_split_heads` / `_merge_heads` helper method on the class for testability and clarity. Currently the reshape logic is inline in `forward`; pulling it into helpers would let you unit-test the reshape independently. Skipped for stage 4; revisit if stage 7+ introduces reshape ambiguity.
- Once stage 9 trains the model, inspect per-head attention patterns to actually *see* specialization. Bookmark TransformerLens for this; current implementation is plain PyTorch and doesn't expose per-head intermediates, but a quick attention-pattern visualization on a forward hook would work.
- Memory pattern of the `transpose + reshape` reverse-merge: O(N) copy is unavoidable in pure PyTorch. Worth confirming the copy cost is negligible at training scale (B=64, T=256, d_model=128) when stage 9 lands — likely yes, but worth measuring.

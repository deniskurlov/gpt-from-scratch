# Stage 3 — Attention Implementation

## 2026-05-13

## What I worked on
Implementing single-head causal `Attention` in `src/attention.py`, hitting and fixing multiple bugs along the way, then writing 5 unit tests including the causality structural test.

## Key concepts
- **Fused QKV projection**: one `nn.Linear(d_model, 2*d_k + d_v)`, then `qkv.split([d_k, d_k, d_v], dim=-1)`. Same parameter count as three separate projections; better memory locality, one kernel call.
- **`chunk(n: int, dim)` vs `split(sizes: list, dim)`**: chunk takes a *count*; split takes *sizes*. Different APIs for similar tasks. Use `split` when widths might differ.
- **`register_buffer('mask', tensor)`**: the canonical PyTorch idiom for non-parameter constants on a Module. The buffer moves with `.to(device)`, is saved/loaded in `state_dict()`, but is excluded from `parameters()`.
- **`transpose(-1, -2)`** over `transpose(1, 2)`: negative-dim indices abstract over the leading-dim count. Same expression works for 3-D `(B, T, d_k)` (stage 3) and 4-D `(B, n_heads, T, head_dim)` (stage 4).
- **`masked_fill(bool, value)` fills *where* bool is True.** The tril-of-ones mask is True at *allowed* (past) positions, so `.logical_not()` is required before `masked_fill` to flip polarity to "mask out the future".
- **Causality test**: modify input at position `T-1`; earlier outputs must be **bit-exact identical**. The only test that catches mask-polarity, mask-shape, mask-axis, or missing-mask bugs.

## What I got wrong
- **`super.__init__()` without parens.** `super` is the class; `super()` is the bound instance — same trap as `dict` vs `dict()`. Second occurrence of this exact bug in the project.
- **`qkv.chunk([d_k, d_k, d_v], dim=-1)`** — pasted `split` arguments into `chunk`. The mnemonic ("chunk takes count, split takes sizes") didn't stick the first time I read it.
- **Mask polarity bug.** Wrote `scores.masked_fill(self.mask[:T, :T], -inf)`. The tril-of-ones bool mask is True at past positions, so `masked_fill` zeroed *past* positions instead of *future*. Caught by tracing the polarity on a 3×3 example by hand. Fix: `.logical_not()` before `masked_fill`.
- **`from _pytest.monkeypatch import V`** — Cursor autocomplete artifact, **second** occurrence in the project. The bogus import silently let `attn @ V` (capital) typecheck instead of NameError-ing on the lowercase-v typo I'd made. **Two coupled bugs hidden by one bad import.** Recurring pattern: always re-read imports after Cursor edits.
- **`out_proj_bias_num = d_v`** in the parameter-count test. Test passed because `d_v == d_model == 128` numerically in the test config — passed for the wrong reason. Fix: read the formula off the `nn.Linear` declaration (`nn.Linear(d_v, d_model)` → bias has `d_model` entries). Lesson: tests with default values that happen to coincide give false green lights. Parametrize over non-equal configs to break the coincidence.
- **`x_modified = x` in the causality test.** Not a copy — modifies `x` in place. Happened to not break this specific test (since `out` was already computed before mutation), but it's a foot-gun that would break analogous patterns. Fix: `x.clone()`.

## Why this works
- **`super().__init__()` is about state, not methods.** Methods are inherited automatically via subclassing. `super().__init__()` initializes parent *instance state*: the `_parameters`, `_modules`, `_buffers` OrderedDicts that PyTorch uses to track submodules. Without it, `self.qkv_proj = nn.Linear(...)` silently fails to register; `parameters()` returns empty; the optimizer trains nothing; training looks "fine" but loss never decreases.
- **`√d_k` scaling**: `Var(Σ_i Q_i K_i) = d_k`, so std = √d_k. Without scaling, softmax saturates as `d_k` grows (one input dominates → softmax outputs near one-hot → **all** entries of `∂softmax_j / ∂x_i` evaluate to 0 at saturation → no gradient flow through softmax → training stalls). The `√d_k` keeps softmax inputs at unit variance at initialization; the model can then learn arbitrary attention sharpness post-init via the QK projections themselves.
- **`register_buffer` over plain attribute**: enrolls the tensor in PyTorch's state-tracking system. Plain `self.mask = torch.tril(...)` doesn't move with `.to('mps')` (stays on CPU → device mismatch crash on first forward) and doesn't survive a checkpoint roundtrip.
- **Mask before softmax with `-inf`**: `-inf + score = -inf` → `exp(-inf) = 0` → softmax denominator sums only allowed positions → rows sum to 1 over the past. Mask-after-softmax would zero future positions but require renormalization; same result, twice the operations.

## Open questions
- Whether to add assertions in `__init__` (positive ints, divisibility for multi-head). Skipped for now; revisit if stage-4 errors become confusing.
- Cursor autocomplete keeps inserting bogus imports despite Cursor Tab being off. Worth investigating the autocomplete settings more carefully before stage 4 brings more files to autocomplete into.
- Pattern noticed: arithmetic slips appear more in shape computations than in math derivations. Possibly because shape arithmetic feels "trivial" so I don't write it down, whereas derivations get the paper-and-pen treatment. Defensive move: write shape arithmetic on paper too, even when it looks like it can be done in head.

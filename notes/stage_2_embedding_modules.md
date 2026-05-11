# Stage 2 — Embedding Modules (implementation)

## 2026-05-11

## What I worked on
Implementing `TokenEmbedding` and `LearnedPositionalEmbedding` in `src/model.py`, wiring up `src/` as a package, setting up shared pytest fixtures in `tests/conftest.py`, and writing 9 unit tests for the two modules.

## Key concepts
- `nn.Module` subclass: `super().__init__()` first, then `self.<sub> = nn.Embedding(...)`. The `super()` call initializes `_parameters` / `_modules` / `_buffers` dicts that `nn.Module.__setattr__` writes into for parameter tracking. Skip it → `model.parameters()` returns empty → optimizer trains nothing → silent failure.
- `nn.Embedding(N, d_model)` is a **Module wrapping** a learnable `(N, d_model)` `.weight` tensor — not the tensor itself. Calling `nn.Embedding(N, d)(ids)` performs indexed lookup, returning `S + (d_model,)` for any input shape `S`.
- Call modules via `module(x)`, not `module.forward(x)` — `__call__` runs forward pre/post-hooks (needed for `torch.compile`, gradient checkpointing, mechanistic-interp tools).
- Broadcasting `(B, T, d_model) + (T, d_model) → (B, T, d_model)`: missing leading dims are treated as size 1 and stretched. No copy made; stride manipulation.
- pytest workflow: `tests/conftest.py` for fixtures (auto-loaded, no imports); fixtures must appear in the test's parameter list to be injected; `@pytest.mark.parametrize("T", [...])` for granular tests; `with pytest.raises(IndexError):` for error tests.

## What I got wrong
- **`__inti__` typo, twice in a row.** Once in commented-out code, once after explicitly flagged. Python doesn't complain — `__inti__` is just a regular method name; the default `__init__` from `nn.Module` runs instead. Surfaces as `AttributeError` on first call. **Magic dunder names are strings Python looks up at specific moments; typos turn them into dead code.**
- **`super.__init__()` without parens.** `super` is the class; `super()` is the bound instance. Same trap as `dict` vs `dict()`. Raises at runtime.
- **`TokenEmbedding(V=tok.vocab, d_model=128)`** — passed the vocab *list*, not `tok.vocab_size`. `nn.Embedding` expects int; PyTorch's error message buried the cause under "invalid combination of arguments".
- **Claimed `token_emb` has `B · T · d_model` parameters.** Confused activation shape with parameter count. Parameters are the entries of the `(V, d_model)` table — independent of batch. Activations `(B, T, d_model)` are per-forward-pass and transient. Parameters vs activations is a foundational ML distinction I had not internalized.
- **Called self-attention "permutation-symmetric".** Correct term: **equivariant** (`f(π(x)) = π(f(x))`). Invariance (`f(π(x)) = f(x)`) is strictly weaker and not what attention has. Equation right, name wrong.
- **Proposed `d_model = 12-16` citing Johnson-Lindenstrauss.** JL gives the floor for distinguishing V categorical tokens, but `d_model` is the **residual-stream bandwidth**, not just an embedding dim. Every block reads/writes the same `d_model`-dim vector, so the model needs capacity for syntactic role, position info, accumulated attention outputs, etc. across layers — not just for token identity. Anthropic / Olah residual-stream framing is the right intuition.
- **Vague on sum vs concat.** Caught "wider is slower" but missed (1) parameters quadruple downstream (QKV+MLP scale with residual-stream width), and (2) the linear-equivalence punchline: `[t; p] @ W = t @ W_t + p @ W_p`, so concat+linear ≡ sum with the embedding matrices acting as `W_t, W_p`. Sum is strictly cheaper for the same operation class.
- **Two walkthrough imprecisions.** Said `super().__init__()` is "needed for methods to be inherited" — methods are inherited automatically; what `super().__init__()` does is initialize *state*. Called `nn.Embedding(V, d_model)` "the tensor of embedding weights" — it's a Module wrapping the weight tensor, not the tensor.
- **Test bug: `def test_token_embedding_shape(tok):` referenced `text` in body without declaring it as parameter.** Pytest injects fixtures only when they're in the parameter list; bare references resolve to the fixture-function-definition object, which isn't iterable. `'FixtureFunctionDefinition' object is not iterable` — diagnostic.

## Why this works
- **`super().__init__()` for parameter tracking.** `nn.Module.__init__` initializes `_parameters`, `_modules`, `_buffers` OrderedDicts. `nn.Module.__setattr__` (invoked by `self.tok_emb = nn.Embedding(...)`) inspects the assigned value, recognizes a Module, and registers it in `_modules`. Without these dicts initialized, registration silently fails. End state: `model.parameters()` empty, optimizer no-op.
- **Sum ≡ concat-then-linear.** Block-decompose `W ∈ R^{2d × d}` as `[W_t; W_p]`. Then `[t; p] @ W = t @ W_t + p @ W_p` — the sum of two linearly-transformed copies. By summing at the input with embedding matrices as the (learnable) projections, concat+linear is collapsed into one step. Concatenation downstream bloats every linear layer's input dim from `d` to `2d`, quadrupling QKV+MLP parameters for zero expressiveness gain.
- **`d_model` as residual-stream bandwidth.** Each block reads `(B, T, d_model)`, adds its computation, writes back. Token identity occupies a small fraction of that bandwidth; the rest carries the running representation (partial parses, attention sums, MLP transformations). Choking `d_model = 16` starves the channel, not just the embedding.

## Open questions
- `n_heads` for stage 4. With `d_model = 128`, candidates: 4 (head_dim=32) or 8 (head_dim=16). Defer to attention implementation.
- Device placement. Encoded corpus stays on CPU; batches will need `.to(device)` before the model at stage 9. Exact pattern (per-batch transfer vs prealloc buffers) is open.
- Wall-clock training time on M4 Pro. Haven't benchmarked. `d_model = 128`, `n_layers = 4-6`, `T = 256` should be tractable; verify when stage 9 lands.

# Stage 1 — Batch Sampling (`get_batch` implementation)

## 2026-05-10

## What I worked on
Implementing `TokenizedDataset.get_batch(B, T) → (x, y)` with vectorized random offset sampling, advanced indexing, and shift-by-1 target alignment, plus tests and a toy by-hand prediction on `corpus="abcab"`.

## Key concepts
- `torch.randint(low, high, size)` samples from `[low, high)` — **exclusive** high. To get an inclusive max `M`, the call site value is `M + 1`.
- Vectorized index construction: `offsets[:, None] + torch.arange(T)[None, :]` broadcasts `(B, 1) + (1, T) → (B, T)`. Row `k` is the contiguous T-window starting at `offsets[k]`.
- Advanced indexing: `self.encoded[idx]` where `idx` is `(B, T)` int64 produces a `(B, T)` int64 output, gathered from the 1-D corpus. Same semantics as NumPy.
- Shift-by-1 supervision: `y = self.encoded[idx + 1]` shares offsets with `x` and yields `B·T` parallel training examples per forward pass.
- Tokenizer/Dataset separation: `Dataset(tokenizer.encode_to_tensor(text))`. Tokenizer is reusable across texts; Dataset just holds a tensor and samples.
- Python is an *interface layer* over PyTorch's C++/CUDA/Metal kernels. Vectorize to stay in C; iterating over a tensor in Python breaks out of the kernel, holds the GIL, pays per-element dispatch — ~100× slower than the equivalent tensor op.

## What I got wrong
- **Off-by-one in `torch.randint`.** Wrote `torch.randint(0, L-T-1, (B,))` for offsets when the correct call is `torch.randint(0, L-T, (B,))`. I had the inclusive max (`L-T-1`) right but pasted it directly into the exclusive-high slot. The discipline I missed: derive the inclusive bound on paper, *then* translate to `inclusive_max + 1` at the call site — never in your head. Especially galling because we'd just spent multiple turns hammering the half-open-interval convention.
- **"Indices start from one"** typo while reasoning about the offset bound. Late-night cosmetic.
- **`set[str](text)`** as a runtime constructor. Works by accident through `__class_getitem__`. Should just be `set(text)`; the parameterized-generic form is for type annotations.
- **`dict[str: int]`** — colon instead of comma. Slice syntax in a generic-parameter context. Static type checkers reject; runtime is permissive but produces a junk type alias.
- **Spurious `from torch._dynamo.utils import V` import.** Cursor autocomplete artifact — `V` was never used. Read your imports.
- **Skipped predicting before showing `torch.randint` output.** The predict-then-check protocol re-engages whenever a tensor is constructed or transformed (shape/dtype) — not for stdlib ops with deterministic semantics, but yes for any tensor manipulation.
- **`Long` doesn't exist in jaxtyping** — that's PyTorch's name. Cross-library naming inconsistency: PyTorch uses Lua/Torch7-era names (`Long`, `Float`); NumPy/JAX/jaxtyping use numpy-style bit-width names (`Int64`, `Float32`).

## Why this works
- **Offset upper bound.** `y = encoded[i+1 : i+T+1]` reads up to index `i+T`. For validity, `i + T ≤ L - 1`, so `i ≤ L - T - 1`. The `x` slice's constraint (`i ≤ L - T`) is *less* restrictive, so it's `y` that binds. Translating the inclusive bound to `torch.randint`'s exclusive-high API: `torch.randint(0, L-T, (B,))`.
- **Advanced indexing.** `self.encoded[idx]` is the sparse-matmul shortcut that mathematically equals `OneHot(idx) @ encoded`. Implementation exploits the 1-of-V sparsity and stays O(d_model · B · T) compute, ~25,000× less memory than materializing the one-hot.
- **Broadcasting `(B, 1) + (1, T) → (B, T)`** is the same primitive that's about to dominate stage 3 (causal masks, pairwise QK^T scores) and stage 13 (RoPE rotation indices). Building the muscle now pays off.
- **`B·T` parallel training examples** come from the shift-by-1 alignment in `y`, *not* from the causal mask. The causal mask (stage 3) is what makes those examples *honest* (no peek at future tokens); without it, the alignment degenerates to a copy task.
- **Tokenizer/Dataset separation = dependency injection.** Caller wires `Dataset(tokenizer.encode_to_tensor(text))`. Each class has one responsibility, both are independently testable, and one Tokenizer can feed multiple Datasets (training, validation, prompts).

## Open questions
- `d_model` selection — open since stage 1 concepts. Becomes live the moment we instantiate `nn.Embedding(V, d_model)` in stage 2.
- Token + positional embeddings: standard practice is elementwise sum. Why not concatenation? What changes if we did?
- When does `PYTORCH_ENABLE_MPS_FALLBACK=1` actually fire? Probable first hit in stage 3 (attention) or stage 13 (RoPE).

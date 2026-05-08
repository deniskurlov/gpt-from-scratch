# Progress

## Where we are
- **Stage:** 1 — Data loading + character-level tokenization
- **Sub-step:** Code phase open. Conceptual probe complete; no source code written yet.
- **Last completed:** Conceptual probe of the full data pipeline (`corpus.txt → str(L) → encoded LongTensor (L,) → (B,T) batches → embedding → (B,T,d_model) → first nn.Linear`), the vocab-vs-encoded-corpus distinction, the geometric argument for integer-indexed embeddings, the one-hot/matmul ↔ indexed-lookup equivalence, and the char-level vs BPE structural contrast. Documented in `notes/stage_1_tokenization_concepts.md`.

## Resume here
Next session: write the actual Stage 1 code. In one Python module, implement (in roughly this order) corpus load from `data/input.txt`, vocab construction via `sorted(set(text))`, the `stoi: Dict[str,int]` and `itos: Dict[int,str]` mappings, `encode(str) → List[int]` and `decode(List[int]) → str`, the encoded corpus as a 1-D `LongTensor` of shape `(L,)`, and a `get_batch(B, T)` sampler that returns `(x, y)` pairs of shape `(B, T)` int64 where `y` is `x` shifted by one token. Then write a tests file covering: encode/decode round-trip on the full corpus, encoded-tensor shape and dtype, vocab size matches `|set(text)|`, batch shapes, and the `y == x_shifted_by_1` invariant. Stage 1 isn't done until those tests pass and Denis can predict by hand what `encode("abcab")` returns under vocab `['a','b','c']` and what one valid `(x, y)` batch looks like for a tiny corpus and small `(B, T)`.

## Open conceptual debts
- **`L` vs `V` discipline.** Denis conflated corpus length with vocab size repeatedly during the probe. They stay distinct in code (`(L,)` for the encoded corpus, `(V, d_model)` for the embedding matrix, `(V,)` for the vocab list), so any shape-print or test that mixes them is a red flag.
- **BPE mechanism articulation.** First answer was hand-wavy ("you can keep merging further and further"). Tightened to "each merge step adds exactly one entry; user picks number of merges (= target vocab size)" only after pushback. Should be re-articulated unprompted before any later "implement BPE" attempt.
- **Next-token `(x, y)` target structure.** Mentioned in passing but not engaged with. The `y` tensor of shape `(B, T)` is `x` shifted by one position, so a single `(B, T)` batch yields `B·T` next-token-prediction examples in parallel. Needs to land cleanly when writing `get_batch`.
- **`d_model` selection.** Open question parked for later — no heuristic given yet for choosing it for char-level on tiny-shakespeare. Will become live at Stage 2 when `nn.Embedding(V, d_model)` is actually instantiated.
- **MPS-fallback awareness.** `PYTORCH_ENABLE_MPS_FALLBACK=1` mentioned in README but Denis hasn't yet hit a missing-op case. Open until first surprise.

## Code state
- `README.md` — written, lists 15 stages and success criteria. ✓
- `.gitignore` — covers `.venv/`, `__pycache__/`, `*.pyc`, `data/`, `.DS_Store`, `.claude/sessions/`. ✓
- `requirements.txt` — `pip freeze` output; direct deps `torch==2.11.0`, `numpy==2.4.4`, plus torch transitives. ✓
- `.venv/` — Python 3.13.5, MPS available and verified (`torch.backends.mps.is_available() == True`). ✓
- `data/input.txt` — tiny-shakespeare, 1,115,394 bytes, gitignored. ✓
- `derivations/` — directory exists, empty. No Stage 1 derivations needed yet.
- `notes/stage_1_tokenization_concepts.md` — Stage 1 conceptual notes with explicit error log. ✓
- **No source code yet.** No `tokenizer.py` / `dataset.py` / equivalent. No tests. No `stage_1_summary.md` (stage-done refused — criteria not met).

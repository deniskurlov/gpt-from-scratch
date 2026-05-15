# Stage 14 reference — SwiGLU

> Saved 2026-05-15 during stage 6 (GELU MLP) for forward reference. Stage 14 is the optional GELU → SwiGLU migration listed in README. Content below is the substantive explanation of *why* SwiGLU outperforms GELU FFN at matched compute. Not from Denis's own derivation — saved as a reference; verify or rederive when stage 14 arrives.

---

## Matched-compute comparison — the core point

SwiGLU's advantage over standard GELU FFN is **not** that it has more parameters. Parameter count is held fixed; the win is **structural**: multiplicative gating.

**Standard FFN**: one pointwise nonlinearity sandwiched between two linear maps,

$$y = W_2 \, \sigma(W_1 x)$$

Every hidden unit $h_i = \sigma((W_1 x)_i)$ is a function of a single learned scalar projection of $x$. The nonlinearity is applied coordinate-wise on one feature.

**SwiGLU FFN**: two parallel projections of $x$, multiplied element-wise,

$$y = W_2 \, \big(\sigma(W_1 x) \odot (W_3 x)\big)$$

Now hidden unit $h_i = \sigma((W_1 x)_i) \cdot (W_3 x)_i$. Each output couples **two independently-learned linear features** of $x$ via a product. The gate $\sigma(W_1 x)$ modulates the value path $W_3 x$. That's a **multiplicative interaction** — the network can compactly express things like "feature B is relevant only when feature A is on", which a coordinate-wise nonlinearity cannot represent without much more width or depth.

## ★ Insight ─────────────────────────────────────

- **Param count is held fixed, not increased.** Standard FFN with `d_ff = 4d` has $2 \cdot d \cdot 4d = 8d^2$ params in the two linear maps. SwiGLU has three linear maps, so to match $8d^2$ you set $d_{ff} \approx (8/3) \cdot d \approx 2.67 d$. Same compute, same params as a vanilla 4d GELU FFN, but observably lower loss. The win is **per-parameter expressivity**, not raw capacity.

- **Why multiplicative coupling matters in theory.** A coordinate-wise nonlinearity acts on a 1-D projection — call it a "sigma" unit. A product of two linear projections is a "sigma-pi" unit (sum-of-products in classical neural-net taxonomy). **Sigma-pi units are strictly more expressive per parameter** for any function class that involves conditional or AND-like structure. Attention itself is multiplicative for the same reason — $\text{softmax}(QK^T)$ is a product of two learned projections of the tokens. SwiGLU brings that same flavor of interaction inside the FFN.

- **What we don't fully understand.** "Gated activations beat pointwise activations at matched compute" is empirical (Shazeer 2020, *GLU Variants Improve Transformer*). The mechanistic theory is unsettled — people argue about whether the gate is doing something analogous to attention, or just providing a nicer optimization landscape, or capturing higher-order moments of $x$. Don't let anyone tell you the explanation is closed; the **result is solid, the story is fuzzy**.

─────────────────────────────────────────────────

## Concrete contrast to test the intuition

Suppose your input has two features and you want the output to be $x_1 \cdot x_2$ (a product). How many ReLU/GELU layers do you need to approximate that?

**At least two** — because one pointwise nonlinearity cannot produce a product. SwiGLU can do it in one block exactly: set $W_1$ to project out $x_1$, $W_3$ to [project out $x_2$, and the product $\sigma(x_1) \cdot x_2$ — or with $\sigma$ initialized near identity, just $x_1 \cdot x_2$ directly. Standard FFN must compose nonlinearities to approximate the product, with more depth and/or width than the analytical solution requires.]

> Bracketed completion above reconstructs the truncated tail of the original paste; verify against original source when revisiting at stage 14.

---

## When to revisit

When stage 14 begins (after the full GPT trains end-to-end on Shakespeare and you're looking for ablations), re-read this and verify:

1. The matched-compute formula $d_{ff} \approx (8/3) \cdot d$.
2. The sigma-pi / multiplicative-gating framing.
3. Shazeer 2020 as the empirical citation.
4. Whether more recent papers (2025+) have settled the mechanistic theory the original "we don't fully understand" disclaimer flagged as open.

---
description: Mark a stage complete with a comprehensive summary doc
argument-hint: [stage number]
---

You are documenting completed learning, not tutoring. Switch role for this command only.

Stage to summarize: $ARGUMENTS. If not provided, ask Denis which stage and stop.

Before writing anything, verify the CLAUDE.md "done" criteria:
1. Code passes its unit test
2. Denis can verbally explain why each line is the way it is
3. Denis can predict a forward-pass output on a toy input without running it

Ask Denis to confirm each, briefly. If any criterion isn't met, refuse to
write the summary and state which one is missing. Do not soften this —
premature stage-done docs are a form of self-deception.

If all three are met, write `notes/stage_<N>_summary.md`:

# Stage <N>: <Title>

## Summary
One paragraph. What this stage built and why it matters in the transformer.

## The math
Equations and derivations referenced or done. Inline if short,
link to `derivations/<name>.md` if longer.

## The code
Files added/modified, one line each. Don't paste code — link to it.

## Design choices and why
Each non-obvious choice with its mechanistic justification.

## Errors and corrections
Synthesize from `notes/stage_<N>_*.md` files. Don't duplicate verbatim.
The errors are the most useful part for future review.

## Self-quiz
5-8 pointed questions Denis should be able to answer cold six months from
now after just rereading this doc. Frontier-lab-interviewer level. No softballs.

## What this enables
Brief note on what stages N+1, N+2 build on this.

After writing the summary, also update `notes/PROGRESS.md` to reflect the
completed stage. PROGRESS.md is the single source of truth for resume-state;
leaving it stale defeats the session-bootstrap mechanism in CLAUDE.md.
The update is a minimal refresh, not a full rewrite — same sections as the
existing PROGRESS.md (which /checkpoint produces in full):

- Update **Where we are**: stage N marked DONE, sub-step set to "stage N+1 not yet started",
  last-completed line updated.
- Update **Resume here**: 2-3 sentences pointing at stage N+1's first concrete step.
- Refresh **Open conceptual debts**: close items resolved in stage N; add items surfaced.
- Refresh **Code state**: add any new files (source, tests, notes) created in stage N.

When done, print both file paths (the summary and the updated PROGRESS.md). No narrative.

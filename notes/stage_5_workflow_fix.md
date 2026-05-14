# Stage 5 — Workflow Fix (PROGRESS.md drift)

## 2026-05-14

## What I worked on
Fixing the workflow gap where `/stage-done` didn't refresh `notes/PROGRESS.md`, which led PROGRESS.md to go stale after stage 3 even as stages 4 and 5 finished.

## Key concepts
- `/stage-done` writes `notes/stage_<N>_summary.md` but doesn't touch PROGRESS.md.
- `/checkpoint` owns PROGRESS.md (single living document, rewritten end-to-end).
- The intended workflow was `/stage-done → /checkpoint → /note` as a manual sequence, but I had stopped running `/checkpoint` after stage 3.
- After two stages without `/checkpoint`, PROGRESS.md drifted by two stages — invisible until I noticed it during stage-5 wrap-up.

## What I got wrong
- **Assumed `/stage-done` updated PROGRESS.md.** It didn't — the two slash commands have non-overlapping responsibilities (per-stage summary vs single living state file), and I conflated them. The mistake was invisible because PROGRESS.md doesn't error out when stale; it just silently misrepresents reality.
- **Stopped running `/checkpoint` after stage 3.** Habit collapsed silently. Workflow rituals that require manual invocation will drift unless either (a) the habit is conscious enough to survive boredom, or (b) the workflow automates the dependency. For me at stage 4+, (a) failed.
- **Didn't notice the drift sooner.** The session-bootstrap pointer in CLAUDE.md ("read PROGRESS.md first in every new session") assumed PROGRESS.md was current. If a new session had started between stages 4 and 5, that session would have begun with the wrong picture of project state.

## Why this works
- The fix: extend `/stage-done` to include a final step that minimally updates PROGRESS.md (refresh "Where we are", "Resume here", "Open conceptual debts", "Code state"). `/checkpoint` remains available for explicit full rewrites and mid-stage pauses.
- Trade-off: `/stage-done` is now slightly less focused — writes two files instead of one. Worth it: the single source of truth for resume-state has to actually stay current, and tying the update to stage completion (the natural milestone where PROGRESS.md should change) is the right binding.
- The broader principle: when a multi-step workflow has a step that everyone forgets, either rebuild the step into a more reliable trigger or accept that the workflow has a known failure mode. Choosing the former here because PROGRESS.md drift is bad: it silently corrupts the bootstrap mechanism for new sessions.

## Open questions
- Whether `/checkpoint` is now redundant. Probably not — it's still useful for mid-session checkpoints when no stage has completed. But verify by checking whether I ever invoke `/checkpoint` outside of stage-end over the next few stages. If I don't, consider folding it into `/stage-done` entirely.
- Whether a settings.json hook should run something at the end of `/stage-done` invocations as a backstop. Probably overkill — the slash command edit should be enough. Revisit if PROGRESS.md drifts again despite the fix.

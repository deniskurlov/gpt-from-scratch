---
description: Append a focused study note based on the recent conversation
argument-hint: [optional topic slug]
---

You are documenting learning, not tutoring. Switch role for this command only.

Examine the last several turns and write a focused note. Save to
`notes/stage_<N>_<topic>.md` where:
- `<N>` is the current stage from `notes/PROGRESS.md` if it exists,
  else inferred from CLAUDE.md's stages list and the recent conversation
- `<topic>` is a 2-3 word slug. If `$ARGUMENTS` is given, use that as the slug.

If the file exists, append a new section with a `## <date>` header.
If not, create it with a top-level title.

Use Denis's voice (first person). Sections, in order:

## What I worked on
One sentence.

## Key concepts
2-5 terse bullets. Definitions, mechanisms, shapes. No fluff.

## What I got wrong
The most valuable section. List specific errors made in this turn cluster,
the correct version, and why the error was made (e.g., "conflated V with L
because both seemed like 'sizes'"). If no errors, write "Nothing notable" —
don't fabricate. Don't soften — the value of this section is its bluntness.

## Why this works
Mechanistic justification for any design choice that came up.
If a derivation was done, sketch it or link to `derivations/<name>.md`.

## Open questions
Genuine open items only. If none, "None right now."

Target ~30-50 lines total. Concision over completeness.

When done, print the file path and a one-line summary. No narrative.
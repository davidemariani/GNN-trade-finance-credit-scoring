# Instruction: maintaining STUDYBOOK.md

`STUDYBOOK.md` is the fast-orientation entry point — it goes stale immediately if it isn't updated in the
same breath as the decisions it tracks, so treat updating it as part of *making* a decision, not a
follow-up chore.

## When to update it

- **Any time a real design or process decision gets made** (graph design, dataset choice, architecture
  choice, tooling choice, scope change) — add a one-line entry to the decision log (most recent first),
  with a one-line "why" and a link to the full detail in `wiki/`, `specs/`, or `CONSTITUTION.md`. Don't
  duplicate the full reasoning here — that belongs in the linked page; this file is the index.
- **When a roadmap phase changes** — update "Where things stand right now" to match `specs/roadmap.md`.
- **When a key number changes** (a re-derived stat, a new baseline result) — update "Key numbers to
  remember."

## Style

- Keep the whole file scannable in under a few minutes — if the decision log gets long enough that this
  stops being true, consider archiving older entries to a `wiki/this-project/decision-history.md` and
  keeping only the last ~10-15 in `STUDYBOOK.md` itself.
- One line per decision in the log, not a paragraph — the paragraph belongs in the linked wiki/specs page.
- Don't let this file and the pages it links to disagree — if you update a decision here, make sure the
  linked page reflects the same thing (or update the link).

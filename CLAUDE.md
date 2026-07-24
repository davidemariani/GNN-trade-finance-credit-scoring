# CLAUDE.md

This file gives Claude Code instructions for working in this repository.

## Before doing anything else

**Read `STUDYBOOK.md` first** for fast orientation (what's been decided, why, and where things stand) —
then **read `CONSTITUTION.md` in full before starting any task in this project.** It defines why this
project exists, the isolation rules that must never be violated (personal GitHub identity only, never
the work GitLab remote or work email), the folder structure, and the core working principles. Do not
skip this even for a small-seeming request.

**During the task, not just before it**: any real decision, discovery, or change of direction gets
recorded immediately in both `wiki/` (full detail) and `STUDYBOOK.md` (one-line decision-log entry) — see
`CONSTITUTION.md` §2.8. This is not optional bookkeeping; it's the thing that keeps this repo's
documentation actually true instead of stale.

Also check, if relevant to the task at hand:
- `specs/mission.md`, `specs/tech-stack.md`, `specs/roadmap.md` — the full plan and its rationale.
- `specs/instructions/` — recurring workflows: adding an architecture, notebook standards, testing
  standards, maintaining the wiki and `STUDYBOOK.md`. Follow these before adding a new architecture,
  notebook, or piece of `src/graph_ml/` code.
- `wiki/` — the knowledge base. Check `wiki/original-project/` before touching anything specific to the
  original 2019 thesis (terminology, data, methodology, results), `wiki/this-project/` for decisions
  already made about this rework, and `wiki/gnn-concepts/` for GNN concepts already covered. Don't
  re-derive or guess something that's already written down here.
- `specs/roadmap.md` — the full phased plan and status of record (start with its "plan at a glance" table).
- `BACKLOG.md` — the ordered next few concrete tasks. Update it and the roadmap checkboxes as work moves.
- `USAGE.md` — how to set up the environment and run things in this repo.

## What this repo is

See `STUDYBOOK.md` (one-paragraph summary + decision log) and `specs/mission.md` (canonical mission).
This file only exists to make sure those get read first — it is not itself a source of truth.

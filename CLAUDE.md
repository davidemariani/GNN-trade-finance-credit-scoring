# CLAUDE.md

This file gives Claude Code instructions for working in this repository.

## Before doing anything else

**Read `CONSTITUTION.md` in full before starting any task in this project.** It defines why this
project exists, the isolation rules that must never be violated (personal GitHub identity only, never
the work GitLab remote or work email), the folder structure, and the core working principles. Do not
skip this even for a small-seeming request.

Also check, if relevant to the task at hand:
- `specs/mission.md`, `specs/tech-stack.md`, `specs/roadmap.md` — the full plan and its rationale.
- `specs/instructions/` — recurring workflows: adding an architecture, notebook standards, testing
  standards. Follow these before adding a new architecture, notebook, or piece of `src/graph_ml/` code.
- `BACKLOG.md` — current work items (To Do / In Progress / Done). Update it as work starts/finishes.
- `USAGE.md` — how to set up the environment and run things in this repo.

## One-line summary of what this repo is

A from-scratch graph neural network (PyTorch + PyTorch Geometric) rework of the original
`networkAnalysisForML` trade-finance transaction-network project — built as both a deep, demonstrable
GNN learning exercise and a software-engineering portfolio piece. Full detail is in `CONSTITUTION.md`
and `specs/mission.md` — treat those files, not this one, as the source of truth. This file exists only
to make sure they get read first.

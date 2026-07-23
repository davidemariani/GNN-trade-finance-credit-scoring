# graph-ml — Graph Neural Networks: study & showcase

A from-scratch graph neural network (PyTorch + PyTorch Geometric) rework of
[`networkAnalysisForML`](https://github.com/davidemariani/networkAnalysisForML), a Msc thesis project
(Birkbeck College, University of London, with Tradeteq Ltd) that modeled a buyer/seller trade-finance
transaction network with hand-engineered `networkx`/bond-graph features fed into classical ML models
(linear model, random forest, MLP, RNN) for credit and default prediction.

This repo has two goals, deliberately pursued together:

1. **Learn graph ML deeply** — from graph representation basics and the message-passing framework, through
   specific architectures (GCN, GraphSAGE, GAT, GIN), each understood well enough to explain the intuition
   and the math, not just call a library function.
2. **Showcase software engineering discipline** — specs-driven development, tested and typed code,
   reproducible notebooks, and honest reporting of results. This is one of the works in a broader
   job-application portfolio; it's built to be reviewed, not just to run.

## What's here

- **`specs/`** — the plan and its rationale: `mission.md`, `tech-stack.md`, `roadmap.md`, and
  `instructions/` for recurring workflows (adding an architecture, writing a notebook, writing a test).
- **`wiki/`** — a growing knowledge base: `original-project/` documents the 2019 thesis this reworks in
  full (terminology, data, methodology, results, and why it needs genuine modernization, not a same-methods
  port), and `gnn-concepts/` is a lookup-style GNN reference that fills in alongside the roadmap.
- **`notebooks/00_foundations/`** — GNN fundamentals (graph representations, message passing, spectral vs.
  spatial convolutions, over-smoothing), independent of any specific dataset.
- **`notebooks/01_architectures/`** — one notebook per architecture, explaining the idea and math with a
  toy example before it's used anywhere else.
- **`notebooks/02_project/`** — applying the above to the real problem: predicting trade-finance/credit
  outcomes from a buyer/seller transaction graph, with an honest comparison against a classical baseline.
- **`src/graph_ml/`** — the production-shaped package (typed, tested) that the notebooks import from.
- **`tests/`** — `pytest` coverage for everything in `src/graph_ml/`.

See [`STUDYBOOK.md`](STUDYBOOK.md) for a fast-orientation summary and decision log,
[`CONSTITUTION.md`](CONSTITUTION.md) for the full governing rationale and working principles,
[`BACKLOG.md`](BACKLOG.md) for current work, and [`USAGE.md`](USAGE.md) for setup and common commands.

## Quick start

```bash
uv sync
source .venv/bin/activate
pytest
jupyter lab notebooks/
```

## Status

Early scaffolding stage — see `specs/roadmap.md` for the phased plan and current progress.

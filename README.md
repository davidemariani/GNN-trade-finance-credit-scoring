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
- **`src/graph_ml/`** — the production-shaped package (typed, tested) that the notebooks import from,
  including a `viz/` module for reusable data/graph/results visualization.
- **`tests/`** — `pytest` coverage for everything in `src/graph_ml/`.

Visualization is a first-class part of the project (topology, architecture, and results visuals — see
`wiki/this-project/visualization.md`), with an interactive Hugo showcase dashboard planned as a later
milestone.

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

Planning/design, graph construction, fixed-origin evaluation, EDA/topology, and the first complete model
comparison are implemented. Relation-aware GraphSAGE reaches 0.305 PR-AUC at seed 42 but only
0.244 ± 0.079 across five frozen-configuration seeds, below the strong LightGBM baseline's 0.465. The
result, including seen/cold-start breakdowns and visual diagnostics, is recorded honestly rather than
tuned against test labels. A 2026-08-02 audit found no direct outcome fields in either model's inputs and
confirmed post-cutoff message isolation, but also found that final-snapshot label maturity and cutoff-wide
training histories are not fully point-in-time. These scores are therefore retrospective benchmarks; the
next milestone is an as-of event/label pipeline, time-aware LightGBM baseline, and temporal GNN. See
`specs/roadmap.md` (its "plan at a glance" table) for the phased plan and current progress, and
`STUDYBOOK.md` for a fast orientation + decision log. The first causal-time lesson and bond-artifact audit
are executable and visual in `notebooks/02_project/05_point_in_time_and_bond_audit.ipynb`.

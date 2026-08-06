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

Planning/design, graph construction, fixed-origin evaluation, EDA/topology, and two complete model
comparisons are implemented. Relation-aware static GraphSAGE reaches 0.305 PR-AUC at seed 42 but only
0.244 ± 0.079 across five frozen-configuration seeds, below the strong LightGBM baseline's 0.465. The
result, including seen/cold-start breakdowns and visual diagnostics, is recorded honestly rather than
tuned against test labels. A 2026-08-02 audit found no direct outcome fields in either model's inputs and
confirmed post-cutoff message isolation, but also found that final-snapshot label maturity and cutoff-wide
training histories are not fully point-in-time. Those impairment scores are therefore retrospective
benchmarks. The first corrected causal p90 slice is now complete: LightGBM reaches 0.079 PR-AUC overall,
while a four-channel temporal role GNN averages 0.053 ± 0.033 across five seeds. Cold-start remains near
its 0.023 prevalence for both model families. See
`specs/roadmap.md` (its "plan at a glance" table) for the phased plan and current progress, and
`STUDYBOOK.md` for a fast orientation + decision log. The first causal-time lesson and bond-artifact audit
are executable and visual in `notebooks/02_project/05_point_in_time_and_bond_audit.ipynb`.

The visual derivation and honest comparison are in
`notebooks/02_project/06_temporal_role_gnn.ipynb`; the causal p90 results are not numerically comparable
with the retrospective impairment scores above because the targets and cohorts differ.

Two pre-holdout expanding-window backtests now protect the reported 2018 period from further architecture
tuning. They reveal strong temporal drift: in the later development fold the temporal GNN is effectively
tied with LightGBM overall (0.119 vs. 0.120 mean PR-AUC), stronger on seen companies (0.194 vs. 0.157),
and weaker on cold-start (0.092 vs. 0.119), with material seed variance. The visual explanation is in
`notebooks/02_project/07_temporal_backtesting.ipynb`. The causally masked temporal graph Transformer is now
implemented and evaluated across the same five seeds. It improves fold-1 seen-company mean PR-AUC to
0.034 but trails LightGBM and the fixed-decay temporal GNN in fold 2 (0.087 overall versus 0.120 and
0.119). Notebook `08_temporal_graph_transformer.ipynb` derives the model, visualizes a real learned
attention pattern, and explains why attention weights are diagnostics rather than causal explanations.
Notebook `09_model_comparison_and_time_ablation.ipynb` separates the retrospective impairment and causal
p90 scoreboards, then shows that learned log-age, fixed 180-day decay, and no-age Transformer variants
are nearly tied on paired validation seeds. Its follow-up coverage gate has a large sign-reversing
effect—0.427 versus 0.302 in fold 1, but 0.010
versus 0.020 in fold 2—so it is retained as an ablation rather than promoted. A subsequent 2×2 experiment
finds compact models weaker; stronger dropout/weight decay produces one fold-2 outlier, not a stable gain.
The final K control finds a regime trade-off: K=2 improves the sparse fold but collapses the later fold,
while K=16 doubles K=8 memory without improving either mean. The original wide, learned-time, residual,
K=8 configuration remains frozen; the planned Transformer ablation sequence is complete.

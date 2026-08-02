# Backlog — graph-ml

The **currently-active, fine-grained task list** — "what am I building right now and next." The full
phased plan and its completion status live in `specs/roadmap.md` (the checkboxes there are the status of
record); this file does **not** duplicate them. Decisions/rationale live in `wiki/` with a one-line
pointer in `STUDYBOOK.md` (see `CONSTITUTION.md` §2.8).

Keep this short: when a "Next up" item is done, check it off in `specs/roadmap.md` and pull the next one
here. If this file starts to mirror the roadmap, prune it back.

## Now (in progress)

_(nothing actively in progress — next action is the top of "Next up")_

## Next up — point-in-time correctness, then temporal modelling

The first vertical slice is complete. These are the next evidence-building steps, not permission to tune
against the held-out test set.

1. [x] **Convert `data/` to Parquet (zstd)** — done via `src/graph_ml/data/convert.py`, verified with an
       exact value-level round-trip check. [ ] **Still open: recover/back up private data** (Parquet files
       are local-only; historical pickles and the temporal snapshot are currently absent). →
       `wiki/this-project/data-availability.md`.
2. [x] **Graph construction** (`src/graph_ml/data/`) — `HeteroData` with company + instrument node types,
       role-typed edges, company identity by **name**, company features aggregated from pre-cutoff
       instruments only, built directly from the real Parquet data. Tests use small, hand-built in-memory
       fixtures (a handful of rows with known expected output — see `testing-standards.md`), not a full
       synthetic dataset (see `wiki/this-project/data-availability.md` for why that was dropped). Done in
       `src/graph_ml/data/graph.py`; the studybook treatment is
       `notebooks/02_project/00_graph_construction.ipynb`.
3. [x] **Split + metrics v1** — inductive cutoff split, final-snapshot label-maturity filter, PR-AUC (+ ROC
       for comparability), seen vs. cold-start breakdown, and leakage-safe training/inference edge views.
       Post-T isolation is tested; point-in-time label/history remediation is items 8–10 below.
       → `evaluation.md`, `notebooks/02_project/01_temporal_split_and_metrics.ipynb`.
4. [x] **Strong baseline** — LightGBM on instrument features + pre-T company aggregates (plus trivial and
       logistic-regression reference points). Overall PR-AUC 0.465; results split by seen/cold-start in
       `results/baseline_metrics.csv`; studybook: `notebooks/02_project/02_tabular_baselines.ipynb`.
5. [x] **EDA + topology viz** (`src/graph_ml/viz/`) — imbalance, temporal volume, degree distributions,
       connected components, hybrid footprint, and anonymous static/interactive ego networks.
       → `wiki/this-project/visualization.md`, `notebooks/02_project/03_eda_and_topology.ipynb`.
6. [x] **Vertical slice** — relation-aware GraphSAGE reaches 0.305 overall PR-AUC versus LightGBM's
       0.465; the honest conclusion and visual diagnostics are in
       `notebooks/02_project/04_hetero_graphsage.ipynb`. Phase 3.5 is closed.
7. [x] **Robustness pass** — five fixed-configuration CPU seeds give overall PR-AUC 0.244 ± 0.079
       (range 0.115–0.305). No hyperparameter changed; seed 42 is the maximum, not the typical run.
       → `results/gnn_metrics.csv`, `notebooks/02_project/04_hetero_graphsage.ipynb`.
8. [ ] **Recover and audit temporal sources** — locate the historical snapshot/private backup if possible;
       establish the prediction timestamp, impairment-event availability, closure semantics, and snapshot
       window meaning. The current workspace has only the eight Parquet files. →
       `wiki/this-project/data-availability.md`.
9. [ ] **Build a point-in-time data contract** — strictly-earlier (`< t_i`) cumulative endpoint histories,
       shifted so a row cannot include itself; label-availability masks; preprocessing fitted per rolling
       training window; adversarial leakage tests. Strictly-prior histories, schema guard, event/horizon
       label availability, and rolling masks are implemented; fold-fitted preprocessing integration
       remains. → `wiki/this-project/evaluation.md`, `wiki/this-project/bond-graph-leakage-audit.md`.
10. [ ] **Rebenchmark tabular first on p90** — p90 has a defensible due-date-plus-90-day availability rule
        and viable current cohorts; p180 has zero mature test positives, while impairment event time is
        unresolved. This separates corrected time handling from graph message passing.
11. [ ] **Implement temporal GraphSAGE** — timestamped role edges, explicit age/recency weighting, causal
        company-state updates, hub-aware recent-neighbor selection, and multiple seeds; document it as a
        visual studybook. → `wiki/gnn-concepts/temporal-graphs.md`.
12. [ ] **Backfill foundations and static ablations alongside the evidence** — message passing, root-only
        neural baseline, remove pre-aggregated company histories, collapse relations, and test degree-aware
        aggregation. GAT/GIN follow after the corrected evaluation contract.

## Parked (revisit when the relevant phase starts)

- Longer hybrid-mediated contagion and temporal message passing; the two-layer v1 does not test these.
- Interactive Hugo showcase / D3 hero pieces (Phase 6).

> Completed work is recorded as checked-off items in `specs/roadmap.md` and as dated entries in
> `STUDYBOOK.md`'s decision log — not duplicated here.

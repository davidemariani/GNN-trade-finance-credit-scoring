# Backlog — graph-ml

The **currently-active, fine-grained task list** — "what am I building right now and next." The full
phased plan and its completion status live in `specs/roadmap.md` (the checkboxes there are the status of
record); this file does **not** duplicate them. Decisions/rationale live in `wiki/` with a one-line
pointer in `STUDYBOOK.md` (see `CONSTITUTION.md` §2.8).

Keep this short: when a "Next up" item is done, check it off in `specs/roadmap.md` and pull the next one
here. If this file starts to mirror the roadmap, prune it back.

## Now (in progress)

_(nothing actively in progress — next action is the top of "Next up")_

## Next up — the data pipeline toward the v1 vertical slice (roadmap Phase 3 → 3.5)

Ordered; each builds on the last. Design is already decided — see `wiki/this-project/graph-design.md`
and `evaluation.md`; these are just the build steps.

1. [x] **Convert `data/` to Parquet (zstd)** — done via `src/graph_ml/data/convert.py`, verified with an
       exact value-level round-trip check. [ ] **Still open: back it up off-GitHub** (the data is currently
       laptop-only; originals kept until this happens). → `wiki/this-project/data-availability.md`.
2. [x] **Graph construction** (`src/graph_ml/data/`) — `HeteroData` with company + instrument node types,
       role-typed edges, company identity by **name**, company features aggregated from pre-cutoff
       instruments only, built directly from the real Parquet data. Tests use small, hand-built in-memory
       fixtures (a handful of rows with known expected output — see `testing-standards.md`), not a full
       synthetic dataset (see `wiki/this-project/data-availability.md` for why that was dropped). Done in
       `src/graph_ml/data/graph.py`; the studybook treatment is
       `notebooks/02_project/00_graph_construction.ipynb`.
3. [x] **Split + metrics** — inductive temporal split, target-aware label-maturity filter, PR-AUC (+ ROC
       for comparability), seen vs. cold-start breakdown, and leakage-safe training/inference edge views.
       → `evaluation.md`, `notebooks/02_project/01_temporal_split_and_metrics.ipynb`.
4. [x] **Strong baseline** — LightGBM on instrument features + pre-T company aggregates (plus trivial and
       logistic-regression reference points). Overall PR-AUC 0.465; results split by seen/cold-start in
       `results/baseline_metrics.csv`; studybook: `notebooks/02_project/02_tabular_baselines.ipynb`.
5. [ ] **EDA + topology viz** (`src/graph_ml/viz/`) — imbalance, temporal volume, degree distributions,
       hybrid footprint, interactive company↔instrument network. → `wiki/this-project/visualization.md`.
6. [ ] **Vertical slice** — add one GNN (GCN or GraphSAGE), compare honestly to LightGBM on PR-AUC, write
       the short conclusion + results visuals. This closes roadmap Phase 3.5.

## Parked (revisit when the relevant phase starts)

- `04_network_snapshots.pkl` snapshot semantics — for the temporal phase, not v1.
- Foundations notebooks (Phase 2) — backfilled around the slice, written when each concept first bites.
- Architecture progression GCN→GraphSAGE→GAT→GIN + applied-model choice (Phase 4).
- Interactive Hugo showcase / D3 hero pieces (Phase 6).

> Completed work is recorded as checked-off items in `specs/roadmap.md` and as dated entries in
> `STUDYBOOK.md`'s decision log — not duplicated here.

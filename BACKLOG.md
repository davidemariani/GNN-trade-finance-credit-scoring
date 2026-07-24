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

1. [ ] **Convert `data/` to Parquet (zstd)** and back it up off-GitHub (the data is currently laptop-only).
       → `wiki/this-project/data-availability.md` "Storage format & policy".
2. [ ] **Synthetic data generator** (`src/graph_ml/data/synthetic.py`) — schema-faithful fake dataset
       (company + instrument, 15-ish hybrids, imbalanced labels) so the pipeline runs without the private
       data and doubles as `tests/` fixtures + CI.
3. [ ] **Graph construction** (`src/graph_ml/data/`) — `HeteroData` with company + instrument node types,
       role-typed edges, company identity by **name**, company features aggregated from pre-cutoff
       instruments only. Tests against the synthetic generator.
4. [ ] **Split + metrics** — inductive temporal split, label-maturity filter, PR-AUC (+ ROC for
       comparability); report seen vs. cold-start breakdown. → `evaluation.md`.
5. [ ] **Strong baseline** — LightGBM on instrument features + pre-T company aggregates (plus trivial and
       logistic-regression reference points).
6. [ ] **EDA + topology viz** (`src/graph_ml/viz/`) — imbalance, temporal volume, degree distributions,
       hybrid footprint, interactive company↔instrument network. → `wiki/this-project/visualization.md`.
7. [ ] **Vertical slice** — add one GNN (GCN or GraphSAGE), compare honestly to LightGBM on PR-AUC, write
       the short conclusion + results visuals. This closes roadmap Phase 3.5.

## Parked (revisit when the relevant phase starts)

- `04_network_snapshots.pkl` snapshot semantics — for the temporal phase, not v1.
- Foundations notebooks (Phase 2) — backfilled around the slice, written when each concept first bites.
- Architecture progression GCN→GraphSAGE→GAT→GIN + applied-model choice (Phase 4).
- Interactive Hugo showcase / D3 hero pieces (Phase 6).

> Completed work is recorded as checked-off items in `specs/roadmap.md` and as dated entries in
> `STUDYBOOK.md`'s decision log — not duplicated here.

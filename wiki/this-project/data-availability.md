# This project: data availability (confirmed 2026-07-23)

Unlike the default assumption in `specs/roadmap.md` Phase 1 (that the original dataset was likely
inaccessible), the owner has the **actual pipeline artifacts from the 2019 thesis** in `data/`
(gitignored — local-only, never committed; see `CONSTITUTION.md` §2). This resolves the open dataset
question: **this rework can use the real anonymized data**, not a public substitute or synthetic
generator.

## Files present, in pipeline order

| File | Shape | Role |
|---|---|---|
| `00_transactionsdf_simNames.pkl` | 163,437 × 37 | Raw anonymized transactions (simulated company names), matches `wiki/original-project/data-and-network-construction.md`. Date range 2007-02-06 to 2018-12-18. |
| `01_instrumentsdf.pkl` | 66,593 × 37 | Deduplicated/cleaned instruments, snake_case columns, `uid` assigned. |
| `02_instrumentsdf_2.pkl` | 59,820 × 57 | Adds target/status flags: `has_impairment1`, `is_pastdue`/`30`/`90`/`180`, `has_prosecution`, `is_open`, etc. |
| `03_instrumentsdf_deg1stats.pkl` | 59,820 × 110 | Adds Tier 1 "trade relationship" features (`cd_*`, `d_*`, `c_*` — see `wiki/original-project/feature-engineering.md`). |
| `04_instrumentsdf_bondgraph.pkl` | 59,820 × 130 | Adds Tier 2 bond-graph features (`imp_*`, `p90_*`, `p180_*`, `flow_shock_*`). |
| `04_instrumentsdf_bondgraph2.pkl` | 59,820 × 135 | Same as above + one-hot currency columns — **the final modeling-ready dataset** from the original pipeline. |
| `04_network_snapshots.pkl` | 59,820 × 3,217 | Per-instrument state repeated across ~247 rolling time snapshots (`sshot_0_*` ... `sshot_246_*`: payment/delay/status fields per window) — **this is the closest thing to genuine temporal/sequential data** in the original pipeline and the most directly useful artifact for a temporal-graph approach. 1.5GB, loads in ~9s. |
| `inst_buyer.pkl` | 3,234 × 9 | Per-buyer (debtor) rollup of credit-event flags. |
| `inst_seller.pkl` | 130 × 9 | Per-seller (customer) rollup of credit-event flags. |

All files load cleanly with the current environment's pandas (3.0) despite being pickled in 2019 —
no compatibility shim needed.

## Confirmed stats (re-derived directly — supersedes rounded figures in the report where they differ)

On `04_instrumentsdf_bondgraph2.pkl` (the final feature set, 59,820 rows):
- Date range: 2013-07-23 to 2018-12-18.
- 132 unique seller *IDs* (`customer_id`), 3,176 unique buyer *IDs* (`debtor_id`) — close to but not
  identical to the report's rounded "115 sellers, 3,219 buyers" (likely a slightly different filtering
  stage).
- Target positive rates (**real class imbalance to design around**): `has_impairment1` 2.06% (1,232
  positives), `is_pastdue90` 7.01% (4,195), `is_pastdue180` 6.02% (3,601).
- Heavy missingness in several lifecycle-date columns (`debt_collection_date`, `cancellation_date`,
  `discharge_*` ~95-98% missing) — expected, since most instruments never reach those lifecycle stages;
  not a data quality problem, a feature-engineering consideration (presence/absence itself is signal).

### Hybrids: match by NAME, not ID (corrected 2026-07-24)

`customer_id` and `debtor_id` are **separate ID spaces** — the same company gets a different ID in each
role — so an ID-based overlap check finds zero and is misleading. Matching on **normalized company name**
finds **15 hybrid companies** (companies acting as both seller and buyer), exactly matching the original
report. Hybrids touch **18.7%** of instruments (12,465 / 66,593) and every one appears on both sides —
they are material bridge nodes, not noise. Consequence for graph construction: **resolve company identity
by name**, giving one canonical company node per name (see `wiki/this-project/graph-design.md`).

### Split-dependent facts relevant to modelling (at cutoff T = 2018-04-30, impairment)

- **Cold-start**: ~56% of test-period companies are unseen in training; ~25.5% of test instruments involve
  a cold-start company. Drives the node-feature choice (aggregated features, not learned embeddings) — see
  `graph-design.md` / `evaluation.md`.
- **Label maturity / censoring**: ~32% of *test* instruments (and ~8% of train) are still `is_open` at the
  data's end — their impairment label may not be final. Must be handled in the evaluation rule — see
  `evaluation.md`.

## Implications for the roadmap

- Phase 3's dataset decision is resolved: **use this real data**, not a public/synthetic substitute.
- `04_network_snapshots.pkl`'s snapshot structure is a strong candidate as the basis for temporal graph
  construction (see `wiki/original-project/limitations-and-motivation-for-gnn.md` on why temporal-aware
  modeling matters here) — worth understanding its exact snapshot semantics (what time window each
  `sshot_N` corresponds to) before building on it.
- Graph construction (company + instrument nodes, role-typed edges — see `graph-design.md`) can be derived
  directly from `00_transactionsdf_simNames.pkl` or `01_instrumentsdf.pkl`, resolving company identity by
  name so hybrids unify correctly.
- Class imbalance (2-7% positive rate depending on target) needs explicit handling in whatever GNN loss/
  sampling strategy is used — the original project used class weighting (MLP) and this should carry
  forward as a design consideration, not be rediscovered as a surprise later.

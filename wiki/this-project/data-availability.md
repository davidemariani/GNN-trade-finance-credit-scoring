# This project: data availability (confirmed 2026-07-23; filesystem re-audited 2026-08-02)

The current local `data/` directory contains the eight converted Parquet artifacts from the 2019 thesis
pipeline (gitignored, local-only, never committed; see `CONSTITUTION.md` §2). All implemented modelling
uses this real anonymized data directly. The original pickle files and the large temporal snapshot file
are **not present in the current workspace as of 2026-08-02**; their backup/recovery status is unknown and
must be resolved before relying on them.

## Files present, in pipeline order

| File | Shape | Role |
|---|---|---|
| `00_transactionsdf_simNames.parquet` | 163,437 × 37 | Transaction-line level source: raw lifecycle/payment dates and amounts; multiple records may belong to one instrument. |
| `01_instrumentsdf.parquet` | 66,593 × 37 | Deduplicated/aggregated instrument level with `uid`; several transaction fields remain as ragged arrays. |
| `02_instrumentsdf_2.parquet` | 59,820 × 57 | Filtered modelling cohort plus final-snapshot target/status fields such as `has_impairment1`, past-due flags and `is_open`. This is the current v1 source. |
| `03_instrumentsdf_deg1stats.parquet` | 59,820 × 110 | Adds Tier-1 relationship and outcome aggregates (`cd_*`, `d_*`, `c_*`). Not used in v1; many require point-in-time reconstruction before reuse. |
| `04_instrumentsdf_bondgraph.parquet` | 59,820 × 130 | Adds Tier-2 bond-graph features (`imp_*`, `p90_*`, `p180_*`, `flow_shock_*`). Not used in v1; the leakage audit found a cross-target duplicate shock column. |
| `04_instrumentsdf_bondgraph2.parquet` | 59,820 × 135 | Nominally adds currency columns and p180 effort, but also mutates 14 bond columns and contains identical p90/p180 families. Not used as v1 input. |
| `inst_buyer.parquet` | 3,234 × 9 | Buyer-level final event rollup. Not point-in-time safe without temporal reconstruction. |
| `inst_seller.parquet` | 130 × 9 | Seller-level final event rollup. Not point-in-time safe without temporal reconstruction. |

Historical inspection described `04_network_snapshots.pkl` as a 59,820 × 3,217 table with roughly 247
rolling snapshots. It is absent now, so those dimensions and semantics are historical notes, not a
currently verified input. Recovery must be followed by a column-by-column timestamp/availability audit.

## Storage format & policy (decided 2026-07-24)

**Decision: convert to Parquet, keep data out of the repo, back up off-GitHub.**

- **Format → Parquet (zstd).** The 2019 `.pkl` files are bulky and unsafe (pickle executes arbitrary code
  on load, and is brittle across library versions). Parquet is columnar, compressed, safe, and faster to
  load. Measured on the real files: `00_transactions` 43.8→3.1 MB (7%), `01_instruments` 27.3→3.5 MB (13%),
  `04_..._bondgraph2` 65.0→9.0 MB (14%). This is the storage format for all working data.
- **Repo → data stays out.** `data/` remains gitignored; the real anonymized data is **not committed**,
  even though compressed it would technically fit under GitHub's limits. Rationale: it's real (if
  simulated-name) financial data, git history is permanent, and a public portfolio repo shouldn't carry
  it.
- **Runnability trade-off (accepted 2026-07-24, revising the original plan).** The original plan here was
  a synthetic-data generator so the modelling pipeline could run end-to-end from a bare public clone. That
  was built (`src/graph_ml/data/synthetic.py`) and then **removed**: a generator draws labels and features
  independently by construction, so a model trained on it has nothing real to learn — which is fine for
  "does the code run" but directly undermines the actual point of this project, testing whether real
  transaction-network structure predicts real credit outcomes. Since injecting a genuine, non-trivial
  feature→label relationship into a fake dataset without just re-deriving the real one isn't a small
  addition, the honest choice is: **no synthetic substitute; the pipeline requires the real local data and
  is not runnable end-to-end from a fresh clone alone.** Reviewers see the work through committed code,
  small hand-built test fixtures (`testing-standards.md`), notebook outputs, and results logs/visuals
  committed as artifacts — not by re-running the modelling pipeline themselves. Revisit only if a
  lower-effort way to give fake data real learnable structure turns up later.
- **Backup/recovery → open and now urgent.** The current Parquet files exist locally, but the historical
  pickles and `04_network_snapshots.pkl` do not. Locate any private backup/source copy, document it, and
  restore the snapshot only into gitignored storage. Do not infer why files disappeared without evidence.
- **The historical `04_network_snapshots.pkl`** is a candidate for temporal work, not yet a dependency.
  After recovery, retain/convert only verified columns needed for as-of event and label reconstruction.
- **Further anonymization**: revisit later if the data is ever to be shared more widely; not needed while
  it stays local-only.

> Not using Git LFS: on a *public* repo LFS blobs are publicly downloadable (doesn't help confidentiality),
> and the free tier (1 GB) is blown by the snapshot file immediately. Wrong tool here.

### Conversion done (2026-07-24); one representation quirk to know about

All 8 non-snapshot files converted via `src/graph_ml/data/convert.py` (`python -m graph_ml.data.convert
data/`), verified with an exact cell-by-cell round-trip check (not just shape/dtype) — **every value is
identical**. Sizes matched the earlier estimate: 7-17% of pickle size (e.g. raw transactions 43.8→3.1 MB,
final feature set 65.0→9.0 MB; total non-snapshot data 1.7 GB pickle → ~230 MB Parquet).

Two **cosmetic, non-lossy** representation changes downstream code must expect, both from pandas
3.0 + Arrow's normal round-trip behavior, not from anything specific to this conversion:
- Plain string columns and the index come back as pandas' `StringDtype` instead of plain `object`.
- Several columns (`posting_date`, `payment_date`, `transaction_type`, `payment_amount`, `ttype`,
  `ttypeset`) hold a **per-instrument list of values across that instrument's transaction lines** (ragged,
  ~1-30 items/row) — a raw/pre-aggregation structure that survives all the way to the final
  `04_instrumentsdf_bondgraph2` modeling table alongside the engineered scalar features. Parquet round-trips
  these as **numpy arrays of `datetime64`/plain values** instead of Python **lists of `Timestamp`/objects**.
  Code reading these columns from Parquet should expect arrays, not lists — an `isinstance(x, list)` check
  written against the old pickles would silently break.

The original `.pkl` files are **not present in this workspace** as of the latest audit. The Parquet
round-trip was previously verified, but that does not replace the missing temporal snapshot.

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

On the filtered 59,820-row modelling graph, the same 15 hybrids touch 12,464 instruments (**20.84%**).
The one-row/count and percentage difference is purely the denominator/filtering stage; topology reporting
for the implemented graph uses 20.84%. See `visualization.md`.

### Split-dependent facts relevant to modelling (at cutoff T = 2018-04-30, impairment)

- **Cold-start**: ~56% of test-period companies are unseen in training; ~25.5% of test instruments involve
  a cold-start company. Drives the node-feature choice (aggregated features, not learned embeddings) — see
  `graph-design.md` / `evaluation.md`.
- **Label maturity / censoring**: ~32% of *test* instruments (and ~8% of train) are still `is_open` at the
  data's end — their impairment label may not be final. Must be handled in the evaluation rule — see
  `evaluation.md`.

## Implications for the roadmap

- Phase 3's dataset decision is resolved: **use this real data**, not a public/synthetic substitute.
- Recover `04_network_snapshots.pkl` if possible and establish what each `sshot_N` timestamp means, when
  target events become observable, and whether values are cumulative or windowed. Until then, the existing
  transaction/lifecycle dates may support reconstruction, but their semantics must be validated.
- The 2026-08-02 leakage audit found that final-snapshot `is_open`/eventual labels are used for both sides
  of the current split. Current scores are therefore retrospective benchmarks, not certified as-of-T
  deployment estimates. See `evaluation.md`.
- The stored Tier-1/Tier-2 tables are not approved shortcuts for temporal reconstruction: outcome
  availability is unproven, bond propagation can use future topology, and the two `04` stages have
  value-level consistency anomalies. See `bond-graph-leakage-audit.md`.
- Graph construction (company + instrument nodes, role-typed edges — see `graph-design.md`) can be derived
  directly from `00_transactionsdf_simNames.pkl` or `01_instrumentsdf.pkl`, resolving company identity by
  name so hybrids unify correctly.
- Class imbalance (2-7% positive rate depending on target) needs explicit handling in whatever GNN loss/
  sampling strategy is used — the original project used class weighting (MLP) and this should carry
  forward as a design consideration, not be rediscovered as a surprise later.

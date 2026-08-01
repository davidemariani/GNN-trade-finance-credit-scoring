# This project: data availability (confirmed 2026-07-23)

Unlike the default assumption in `specs/roadmap.md` Phase 1 (that the original dataset was likely
inaccessible), the owner has the **actual pipeline artifacts from the 2019 thesis** in `data/`
(gitignored — local-only, never committed; see `CONSTITUTION.md` §2). This resolves the open dataset
question: **all modelling and pipeline development uses the real anonymized data directly** (no synthetic
substitute — see "Storage format & policy" below for why that was tried and dropped).

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
- **Backup → off-GitHub.** The data currently exists only on this laptop and is explicitly unsynced (the
  "GitHub is the single source of truth" claim in `tech-stack.md` is true for code, **not** for data). It
  should be backed up once to a private location (private Release asset / cloud storage) so a laptop
  failure doesn't lose the dataset that underpins the whole project.
- **The 1.5 GB `04_network_snapshots.pkl`** is only needed for the deferred temporal phase — convert/keep
  it only when that phase starts, and consider storing just the columns/downsample actually needed.
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

The original `.pkl` files are **kept for now** (not deleted) until the off-GitHub backup above actually
exists — deleting the only copy of the real data before a backup is confirmed would be needless risk.

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

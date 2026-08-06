# Studybook — rapid consultation

The fast-orientation doc for this project. Read this to get back up to speed in a few minutes — for depth
on any item, follow the link into `wiki/`, `specs/`, or `CONSTITUTION.md`. This is not a replacement for
those files, it's the index/cheat-sheet layer on top of them. See `specs/instructions/studybook-standards.md`
for how this file gets maintained.

## What this project is, in one paragraph

A from-scratch graph neural network (PyTorch + PyTorch Geometric) rework of `networkAnalysisForML`, a 2019
Msc thesis that predicted trade-finance credit risk (impairment / 90-day delay / 180-day delay) from a
buyer-seller transaction network using hand-engineered "bond graph" features and classical ML. This repo
is deliberately dual-purpose: a genuine graph-ML learning exercise (fundamentals through architectures,
each explained) and a software-engineering portfolio piece (specs-driven, tested, documented), tracked
among the works referenced from the job-application portfolio at `~/Desktop/studybook`. Full mission:
`specs/mission.md`.

## Terminology cheat-sheet

(Full glossary: `wiki/original-project/glossary.md`)

| Term | Meaning |
|---|---|
| imp | Impairment — instrument written off as uncollectable. Rarest, most severe event (2.06% positive). |
| p90 / p180 | Payment delayed ≥90 / ≥180 days. 7.01% / 6.02% positive. |
| Instrument | One invoice/transaction — the atomic unit and label-bearing node in this project's graph. |
| Buyer / debtor | Company that owes payment. Seller / customer | Company financed against the invoice. |
| Effort / flow / energy | Original thesis's bond-graph-theory hand-engineered features (see glossary for exact defs) — this rework aims to learn equivalent signal via message passing instead. |

## Key numbers to remember

- Original results as a **reference point** (`wiki/original-project/results.md`): impairment RF 0.954, p90
  RF 0.861, p180 MLP 0.884 — these are *ROC AUC*, which misleads at 2% positives, so they're for
  comparability, not a target to chase. This project's headline metric is **PR-AUC** (`evaluation.md`).
- Real data (`wiki/this-project/data-availability.md`): 59,820 instruments (final feature set), 132 seller
  IDs, 3,176 buyer IDs, dates 2013-07-23 to 2018-12-18. **15 hybrids** (companies that are both buyer and
  seller) — resolvable only by company *name*, since buyer/seller IDs are separate spaces; hybrids touch
  18.7% of instruments (the real network-contagion signal).
- Split facts at T=2018-04-30, A=2018-12-18 (impairment): 42,321 mature train / 9,293 mature test;
  8,206 open negatives excluded as right-censored; 2,208 mature test instruments (23.76%) are cold-start.
  Test prevalence is 5.62% overall, 4.12% seen, and 10.42% cold-start. → `evaluation.md`.
- v1 graph (`wiki/this-project/graph-design.md`): company + instrument node types, ~63k nodes / ~120k
  edges — small enough for full-batch training, no sampling infra needed yet.
- Strong baseline (`wiki/this-project/evaluation.md`): LightGBM overall PR-AUC **0.465** / ROC AUC 0.913;
  seen PR-AUC 0.432, cold-start 0.387. At a 5% review budget overall precision is 49.03% and recall 43.68%.
- Filtered graph topology (`wiki/this-project/visualization.md`): 45 components; largest contains 81.55%
  of nodes; median company degree 5 vs. maximum 5,636; 15 hybrids touch 20.84% of modelling instruments.
- Causal p90 comparison (`wiki/this-project/evaluation.md`): LightGBM PR-AUC 0.079 overall / 0.102 seen /
  0.026 cold-start; temporal role GNN five-seed mean 0.053 ± 0.033 / 0.065 ± 0.044 / 0.023 ± 0.003.
- Later pre-holdout p90 fold (`wiki/this-project/evaluation.md`): temporal GNN mean PR-AUC 0.119 overall /
  0.194 seen / 0.092 cold-start versus LightGBM 0.120 / 0.157 / 0.119; ordering varies through time.
- Temporal Transformer backtest: fold-2 mean PR-AUC 0.087 overall / 0.146 seen / 0.067 cold-start; it
  improves sparse fold-1 seen ranking but does not displace LightGBM or the fixed-decay temporal GNN.

## Decision log (most recent first — one line + why, link for detail)

- **Coverage gating has a large but regime-dependent effect (2026-08-06)**: a scalar gate conditioned on
  the current root state and four relation-coverage fractions raises fold-1 validation PR-AUC from 0.302
  to 0.427, but lowers fold 2 from 0.020 to 0.010. *Why*: the sign reversal is not a robust improvement;
  residual fusion stays default and smaller capacity/stronger regularization is next. The implementation
  also preserves the RNG stream so a dormant gate cannot silently change control dropout masks. →
  notebook 09, `results/temporal_transformer_fusion_ablation.csv`
- **Transformer time encoding is not the main bottleneck (2026-08-06)**: paired five-seed validation
  means for learned log-age, fixed 180-day attention decay, and no age are respectively 0.3016/0.3046/
  0.2968 in fold 1 and 0.0203/0.0199/0.0195 in fold 2. *Why*: differences are small relative to seed
  spread and fixed decay does not improve both origins, so learned time remains the default and the next
  ablation targets root/message fusion (now completed below). → notebook 09,
  `results/temporal_transformer_time_ablation.csv`
- **Temporal graph Transformer evaluated but not promoted (2026-08-06)**: the complete causal model,
  rolling train/select/refit wrapper, five-seed backtest, and anonymous learned-attention diagnostic are
  implemented. It improves fold-1 seen mean PR-AUC to 0.034, but fold-2 overall reaches only 0.087 versus
  LightGBM 0.120 and fixed-decay GNN 0.119; cold-start remains weak. *Why*: selective aggregation is
  plausible but is not a regime-stable improvement. Next changes stay validation-only and target time
  encoding, capacity, regularization, and root fusion. → notebook 08,
  `results/temporal_transformer_backtest_p90_metrics.csv`

- **Temporal graph Transformer core implemented behind the backtest gate (2026-08-06)**: relation and
  learned continuous-time encodings feed masked multi-head attention; all-empty histories take an exact
  root fallback, and masked padding cannot change logits. *Why*: establish attention semantics and
  trainability before adding expensive rolling experiments. →
  `src/graph_ml/models/temporal_graph_transformer.py`, notebook 08
- **Bounded causal event tensors implemented for attention (2026-08-06)**: each invoice now receives up
  to K newest strictly-prior events in four role channels, with positive age, padding mask, and auditable
  source index; future/simultaneous events are adversarially excluded. *Why*: temporal attention needs
  individual legal events rather than a pre-collapsed mean, and hubs require predictable memory. →
  `src/graph_ml/data/temporal_graph.py`, notebook 07
- **Temporal graph Transformer accepted as the next candidate, behind causal sequence construction
  (2026-08-05)**: attention may select informative events better than one fixed-decay mean, especially at
  hubs, but it cannot solve missing history and must use strict time masks, bounded role-specific events,
  five seeds, and pre-holdout selection. *Why*: backtests show relation context can help seen companies but
  is non-stationary and seed-sensitive. → `wiki/gnn-concepts/temporal-graph-transformers.md`, notebook 07
- **Pre-holdout temporal backtesting implemented (2026-08-05)**: two accepted twelve-month folds rebuild
  preprocessing/selection/refit per origin and enforce at least ten examples per class; the later fold
  finds temporal GNN 0.119 overall / 0.194 seen versus LightGBM 0.120 / 0.157, but weaker cold-start.
  *Why*: architecture choices need development-time evidence now that the final 2018 holdout has been
  reported. → `wiki/this-project/evaluation.md`, `results/temporal_backtest_p90_summary.csv`, notebook 07
- **Root-only temporal control isolates graph value and cold-start harm (2026-08-02)**: removing relation
  messages lowers five-seed mean PR-AUC from 0.053 to 0.035 overall and 0.065 to 0.038 seen, but improves
  cold-start from 0.023 to 0.033 with much lower variance. *Why*: causal graph history carries signal when
  available, while the current sparse-history gates need a better fallback before adding depth. →
  `wiki/this-project/evaluation.md`, `results/root_only_p90_metrics.csv`, notebook 06
- **First causal temporal GNN slice completed (2026-08-02)**: four role-aware, exponentially decayed
  strictly-prior channels produce mean p90 PR-AUC 0.053 ± 0.033 overall across five fixed seeds versus
  LightGBM's 0.079; one seed exceeds the tree, but the distribution is unstable and cold-start remains at
  prevalence. *Why*: time-aware message passing is now tested under the same legal information set, and
  the result points to validation-only component ablations and cold-start inputs rather than test-driven
  tuning. → `wiki/gnn-concepts/temporal-role-gnn.md`, `wiki/this-project/evaluation.md`, notebook 06

- **First causal p90 LightGBM benchmark completed (2026-08-02)**: strictly-prior role-aware endpoint
  histories and fold-fitted preprocessing produce PR-AUC 0.079 overall / 0.102 seen / 0.026 cold-start;
  overall ROC AUC is 0.819, illustrating again why ROC alone flatters rare-event performance. *Why*: this
  is the first defensible bar for a temporal GNN; cold-start companies receive almost no useful ranking
  signal and become the main design challenge. It is not numerically comparable with the old impairment
  result. → `wiki/this-project/evaluation.md`, `results/point_in_time_p90_metrics.csv`, notebook 05
- **p90 selected for the first complete causal pipeline (2026-08-02)**: explicit label-availability and
  rolling-origin code now supports event/resolution timestamps or due-date horizons. At current T/A, p90
  has 38,083 known-before-T train rows (3,041 positives) and 10,554 known-through-A test rows (222 positives);
  p180's 2,504-row test cohort has zero positives, and impairment event time remains unverified. *Why*:
  p90 lets temporal engineering continue without fabricating impairment timestamps or evaluating an
  all-negative cohort. → `wiki/this-project/evaluation.md`,
  `src/graph_ml/evaluation/point_in_time.py`
- **Original bond features are quarantined pending causal regeneration (2026-08-02)**: Tier-1 outcome
  histories are shifted by invoice order but do not prove the earlier outcomes were observable; Tier-2
  flows propagate those rates and can also see future topology. The stored artifacts additionally contain
  exact p90/p180 feature-family duplicates and unexplained stage mutations. *Why*: target-derived network
  features are only safe when built from strictly prior known events inside each fold. A tested schema
  guard and strictly-prior history primitive now start that replacement. →
  `wiki/this-project/bond-graph-leakage-audit.md`, `src/graph_ml/data/temporal.py`
- **Point-in-time leakage audit changes the next milestone (2026-08-02)**: v1 excludes lifecycle/outcome
  fields from model inputs and blocks post-T feature/message contamination, but its final-snapshot label
  maturity can admit outcomes learned after T, while cutoff-wide endpoint histories let early training
  rows see later pre-T attributes/topology. The 0.465 LightGBM and 0.305 seed-42 GraphSAGE PR-AUC scores
  remain comparable retrospective benchmarks, not certified prospective estimates. *Why*: time must be
  represented in features, labels, preprocessing, validation, and graph messages—not just the train/test
  mask. This motivated the now-completed strictly-as-of p90 histories, rolling baseline, and temporal role
  GNN; verified impairment event time remains open. → `wiki/this-project/evaluation.md`,
  `wiki/gnn-concepts/temporal-graphs.md`
- **First GNN vertical slice completed (2026-08-02)**: deterministic relation-aware GraphSAGE reaches
  PR-AUC 0.305 overall / 0.291 seen / 0.319 cold-start at seed 42, below LightGBM's
  0.465 / 0.432 / 0.387; five frozen seeds average only 0.244 ± 0.079 overall. *Why*: GraphSAGE matches
  the inductive typed graph, but two-hop mean aggregation is both weaker than a strong tree over
  post-T-isolated endpoint histories and initialization-sensitive; the negative result narrows the next
  questions without weakening the test contract. → `wiki/this-project/evaluation.md`,
  `notebooks/02_project/04_hetero_graphsage.ipynb`, `results/gnn_metrics.csv`
- **Applied notebooks adopt a visual teaching language (2026-08-01)**: tested schema, real anonymous ego,
  receptive-field, cohort, ranking-curve, baseline, and importance figures now accompany the prose.
  *Why*: each studybook should make abstract graph and evaluation objects inspectable without exposing
  company identities or reducing the notebooks to unexplained charts. → `wiki/this-project/visualization.md`
- **EDA/topology layer implemented (2026-08-01)**: tested aggregate/static/pyvis builders show severe
  imbalance, temporal shift, heavy-tailed company degree, 45 components, and an anonymous hybrid ego
  network; the filtered graph's hybrid footprint is 20.84% (not the pre-filter table's 18.7%). *Why*:
  topology affects normalization, reachability, and the contagion claim, while anonymous bounded views
  remain safe to publish. → `wiki/this-project/visualization.md`,
  `notebooks/02_project/03_eda_and_topology.ipynb`
- **Strong tabular baseline completed (2026-08-01)**: temporally validated LightGBM on instrument +
  post-T-isolated endpoint histories reaches PR-AUC 0.465 overall (0.432 seen / 0.387 cold-start), far above
  logistic's 0.074. *Why*: this is the honest bar the GNN must clear; beating a weak linear reference would
  prove little. → `wiki/this-project/evaluation.md`, `results/baseline_metrics.csv`,
  `notebooks/02_project/02_tabular_baselines.ipynb`
- **Temporal evaluation contract implemented (2026-08-01)**: T=2018-04-30, A=2018-12-18; impairment
  labels mature when positive or closed, open negatives are right-censored; PR-AUC/ROC/top-k metrics and
  seen/cold-start masks share leakage-safe training/inference graph views. *Why*: labels can be unknown,
  and a graph test mask alone does not stop post-T messages contaminating company states.
  → `wiki/this-project/evaluation.md`, `notebooks/02_project/01_temporal_split_and_metrics.ipynb`
- **Graph construction implemented (2026-07-27)**: cutoff-fitted PyG `HeteroData` with 59,820 instrument
  nodes, 3,349 name-resolved company nodes, role-typed reverse relations, origination-safe instrument
  features, and role-specific pre-cutoff company history; eventual outcome aggregates are deferred
  because their as-of-cutoff availability is not yet proven. *Why*: graph preprocessing is part of
  leakage control, not a neutral formatting step. → `wiki/this-project/graph-design.md`,
  `notebooks/02_project/00_graph_construction.ipynb`
- **Synthetic data generator removed (2026-07-24), reversing the entry directly below.** Built, then
  deleted the same day: it drew labels independently of features by construction, so a model trained on it
  has nothing real to learn. Fine for "does the code run," but that's not this project's purpose — the
  point is testing whether real transaction-network structure predicts real credit outcomes, which requires
  real linkage between features and labels. **Decision: no synthetic substitute.** The pipeline now runs
  only against the real local data; it is not runnable end-to-end from a bare public clone. Reviewers see
  the work through committed code, small hand-built test fixtures, notebook outputs, and results
  artifacts — not by re-running the pipeline themselves. *Why*: a fake dataset with real learnable
  structure isn't a small addition (it's close to re-deriving the real one), so the honest trade was to
  drop runnability rather than ship a generator that quietly misrepresents what the project tests.
  → `wiki/this-project/data-availability.md` "Runnability trade-off"
- **Synthetic data generator built (2026-07-24, superseded by the entry above the same day)**:
  `src/graph_ml/data/synthetic.py` (`generate_instruments()`) produced a fake instruments table matching
  the real `02_instrumentsdf_2` schema, with hybrid coverage and disjoint ID spaces guaranteed rather than
  left to chance, and calibrated currency/factoring/amount distributions. 8 tests. Kept here only as a
  record of the reasoning trail — the code itself is deleted.
- **Data converted to Parquet + a `.gitignore` bug fixed (2026-07-24)**: ran the conversion
  (`src/graph_ml/data/convert.py`, tested), verified **exact** cell-by-cell value equality against the
  originals (not just shape) — only cosmetic, expected representation changes (list→array, object→
  StringDtype), documented so future code doesn't assume the old shapes. While building this, found
  `.gitignore`'s unanchored `data/` rule was silently excluding **any** directory named `data` anywhere in
  the repo, not just the top-level one — it was hiding the brand-new `src/graph_ml/data/` and `tests/data/`
  source code from git. Fixed to `/data/` (anchored). *Why it matters*: caught before any commit, but would
  have silently lost source code otherwise — a reminder to `git status`-check after adding any new
  top-level-named directory. The later 2026-08-02 filesystem audit found only the converted Parquet files;
  the historical pickles and temporal snapshot are no longer present locally, so recovery is now explicit
  work. → `wiki/this-project/data-availability.md`
- **Visualization approach (2026-07-24)**: Python stack — matplotlib (static, GitHub-rendered) + Plotly
  (interactive, Hugo-ready HTML export) + pyvis (interactive network topology); Mermaid + torchview for
  architecture/message-passing diagrams. Visualization is cross-cutting (woven into every phase, reusable
  code in `src/graph_ml/viz/`), not a phase. **D3.js is deferred, showcase-only** — reserved for a few hero
  pieces on the eventual Hugo static-site dashboard (Phase 6), adapting the D3 force-graph from
  `dave_the_human`. *Why*: get interactivity from Python for free now; reserve bespoke JS for the showcase.
  → `wiki/this-project/visualization.md`
- **Data storage → Parquet, out of repo (2026-07-24)**: convert working data from 2019 `.pkl` to Parquet
  (zstd) — measured 7-14% of pickle size, safe, fast; keep `data/` gitignored (not committed even though it
  would fit), back up off-GitHub, revisit further anonymization later. *Why*: efficient + safe + confidential,
  without repo bloat. → `wiki/this-project/data-availability.md`
- **Plan review + corrections (2026-07-24)**: adversarial review of the whole plan against the data.
  Corrected the hybrid finding (15 hybrids exist, by *name* not ID — earlier "zero overlap" was wrong);
  revised the graph schema to **company + instrument** (two node types) so hybrids/contagion work;
  replaced "pure learned company embeddings" with **time-windowed aggregated company features** (the
  embedding+inductive combo was self-contradictory and 25% of test instruments are cold-start); added
  `evaluation.md` (PR-AUC co-primary since AUC misleads at 2% positives, label-maturity rule, strong
  gradient-boosting baseline). *Why*: correctness holes + honest-benchmarking. → `wiki/this-project/graph-design.md`, `wiki/this-project/evaluation.md`
- **Graph design v1** (superseded in part by the review above): heterogeneous, static, instrument-centric;
  node classification on impairment only for v1; inductive train/test split for leakage control. *Why*:
  matches label granularity and avoids clique-blowup of a homogeneous instrument graph. The later audit
  showed that cutoff isolation does not close every point-in-time leak. → `wiki/this-project/graph-design.md`
- **Confirmed real dataset available (historical entry, availability revised 2026-08-02)**: the full
  pipeline was inspected and converted; the current workspace retains all eight Parquet stages but no
  original pickles or temporal snapshot. *Why it matters*: existing v1 work uses real data, while temporal
  work now has an explicit recovery dependency. → `wiki/this-project/data-availability.md`
- **2019 methodology is a reference point, not a spec to reproduce**: the rework must genuinely modernize
  (heterogeneous + temporal graph learning), not just port bond-graph feature engineering into PyTorch.
  *Why*: graph ML has moved on since 2019; the original's own limitations section (and the fact it names
  GNNs as "the natural prosecution of the project") supports this. → `wiki/original-project/limitations-and-motivation-for-gnn.md`
- **Deep-read the original thesis in full** rather than working from the README/script skim. *Why*: needed
  precise definitions (imp/p90/p180, bond-graph terms) and the actual results to benchmark against, not
  guesses. → `wiki/original-project/`
- **`Report.pdf` kept local-only (gitignored); only derived wiki markdown is published.** *Why*: minor
  personal detail (family first names in the acknowledgements) unsuited to a public repo; no technical
  content is lost since the wiki captures everything useful.
- **`wiki/` built as plain markdown, in-repo** (not an Obsidian vault, not a static site — yet). *Why*:
  directly agent-readable, git-tracked alongside the code, renders natively on GitHub (helps the portfolio
  angle), no extra tooling dependency. → `wiki/README.md`
- **Specs-driven development structure adopted** (`specs/mission.md`, `tech-stack.md`, `roadmap.md`,
  `instructions/`), mirroring the pattern already used in `~/dave_the_human`. *Why*: consistency with an
  established personal workflow, and it's genuinely useful for a project meant to demonstrate process.
- **Reframed as a dual-purpose learning + portfolio project**, not just an applied rework. *Why*: explicit
  owner direction — this repo needs to be reviewable, not just runnable.
- **New public GitHub repo** (`davidemariani/GNN-trade-finance-credit-scoring`), not a fork/overwrite of
  the original. *Why*: clean separation; the original stays untouched as historical reference.
- **`uv`-managed venv, Python 3.12, isolated per-repo git identity** (personal GitHub, not work GitLab).
  *Why*: matches the isolation pattern already used for other personal projects on this machine.
  → `specs/tech-stack.md`, `CONSTITUTION.md` §1

## Where things stand right now

Planning/design is done (Phases 0-1 + the design decisions), the data is converted to Parquet (backup
still open), and graph construction, fixed-origin evaluation, EDA/topology, and both retrospective and
causal model comparisons are implemented, tested, and explained in ten applied notebooks. Phase 3.5 is
complete, and the first Phase 3.6 causal p90 slice now compares fold-safe LightGBM with a four-channel
temporal role GNN and a bounded temporal graph Transformer. Neither neural family wins robustly and
cold-start remains unresolved. Time treatment is nearly tied and coverage gating reverses across origins;
smaller Transformer capacity/stronger regularization is next, alongside foundations notebooks;
impairment timestamp recovery remains open in parallel. The causal audit is in notebook 05 and the visual
temporal-model derivation/results and comparison are in notebooks 06–09. See
`specs/roadmap.md` for the full phased plan and `BACKLOG.md` for the ordered next tasks.

## Map of the docs (what to read for what)

- **This file** — fast orientation, decision log.
- `CONSTITUTION.md` — the rules (isolation, principles, folder structure). Read first if you're an agent
  picking this up cold.
- `specs/` — the plan (`mission.md`, `tech-stack.md`, `roadmap.md`) and recurring workflows (`instructions/`).
- `wiki/` — the knowledge base: `original-project/` (the 2019 thesis, in depth), `this-project/`
  (`data-availability`, `graph-design`, `evaluation`, `visualization`), `gnn-concepts/` (growing GNN reference).
- `BACKLOG.md` — the ordered next few tasks (Now / Next up / Parked); roadmap holds the full plan.
- `USAGE.md` — how to actually run things.

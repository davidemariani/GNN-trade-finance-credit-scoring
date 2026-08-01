# This project: evaluation methodology (decided 2026-07-24)

The single home for *how models get measured* on this problem — metrics, the train/test split, label
maturity, cold-start handling, the baseline set, and results logging. `graph-design.md` owns the graph;
this file owns the yardstick. Both exist so no single doc has to be re-derived from chat history.

## Metrics: PR-AUC is the headline, ROC AUC is for comparability only

The original thesis reported **ROC AUC** as its primary metric. On this data that is misleading:
impairment is **2.06% positive**, and ROC AUC is well known to be over-optimistic and insensitive under
heavy class imbalance (the true-negative-dominated denominator flatters the score).

**Decision:**
- **Primary: PR-AUC (average precision)** — the standard headline for rare-event detection; sensitive to
  performance on the positive class we actually care about.
- **Secondary: ROC AUC** — reported *only* so results are comparable to the original thesis's numbers
  (`wiki/original-project/results.md`), never as the optimization target.
- Also report a **precision/recall operating point** (e.g. recall at a fixed low false-positive rate, or
  precision@k) since that's what a credit-risk user actually acts on.

**Reframing the goal**: not "beat RF's 0.954 ROC AUC." The honest, more valuable goal is *rigorously test
whether a GNN adds value over a strong tabular baseline, and report it either way* — including "it
doesn't," which is a legitimate and more mature portfolio result than a forced win. The 0.954 figure is a
reference point, not a target to chase past the point of honesty (and it is a ROC AUC, so not even the
primary metric here).

## Train/test split

- **Temporal, not random.** Split on `invoice_date` at a cutoff T; train strictly before T, test on/after.
  Random/shuffle splits are used only as a deliberately-inflated reference (as the original showed), never
  as the reported result.
- **Cutoff T**: **2018-04-30** for impairment (fixed for v1, matching the original for comparability).
  The analysis snapshot ends **2018-12-18**, giving ~8 months of test runway.
- **Inductive**: company node features are computed from pre-T instruments only; the model trains on the
  pre-T subgraph and is evaluated on post-T instrument nodes attached to it. See `graph-design.md`.

### Graph edge visibility (implemented 2026-08-01)

A node mask is insufficient for a GNN because edges allow test nodes to change representations used by
other nodes. The implementation therefore builds two node-induced views and returns their mappings to the
original graph indices:

- **Training view**: physically contains pre-T instruments and their connected companies only, preventing
  post-T tensors from influencing global operations such as batch normalization. Censored pre-T
  instruments remain as unlabelled structural context, but never enter the loss.
- **Inference view**: only pre-T instruments may send messages into companies; companies may send messages
  to all instruments observed by the analysis date. Post-T test instruments receive frozen historical
  company context but cannot update company states or contaminate one another.

This is a conservative **fixed-origin** evaluation. A later temporal model may roll history forward, but
only with strictly time-ordered message passing. Implementation:
`src/graph_ml/evaluation/split.py`; studybook: `notebooks/02_project/01_temporal_split_and_metrics.ipynb`.

## Label maturity / censoring (must handle — easy to get silently wrong)

`has_impairment1` is an *eventual* outcome. At the analysis snapshot, **~32% of test-period and ~8% of
train-period instruments are open**. An open instrument with no recorded impairment is not a confirmed
negative: it may impair after observation ends. This is right censoring.

**Decision (v1 impairment rule, implemented 2026-08-01):** a label is mature by analysis date A when the
positive event has already been observed **or** the instrument is closed:

`mature = invoice_date <= A AND (has_impairment1 OR NOT is_open)`.

This correctly retains the seven open instruments that already have a known positive impairment while
excluding 8,206 open negatives whose eventual status is unknown. No arbitrary 180-day window is imposed
on impairment: unlike p180, impairment has no fixed event horizon, while closure directly establishes a
negative. This is target-specific; p90/p180 will need due-date-plus-horizon maturity rules when added.

At T=2018-04-30 and A=2018-12-18 the resulting cohorts are:

| Cohort | Instruments | Impairments | Prevalence |
|---|---:|---:|---:|
| Mature train | 42,321 | 710 | 1.68% |
| Mature test — all | 9,293 | 522 | 5.62% |
| Mature test — seen | 7,085 | 292 | 4.12% |
| Mature test — cold-start | 2,208 | 230 | 10.42% |

The prevalence shift is itself an important result: later and especially cold-start instruments are much
riskier, so every model report must show cohort prevalence beside PR-AUC. Excluding censored negatives can
also select toward faster-resolving instruments; survival analysis is a possible later extension, but is
outside the v1 binary-classification scope.

## Cold-start companies

~56% of test-period companies are unseen in training; before maturity filtering ~25.5% of test instruments
involve one. In the implemented mature test cohort, **2,208 / 9,293 (23.76%)** are cold-start, defined as
having at least one seller or buyer absent from all pre-T instruments. This is handled by the node-feature
choice (aggregated features → new company gets an honest zero-history vector, not a random embedding — see
`graph-design.md`), **and** results are reported split by seen/cold-start so a strong aggregate number
cannot hide poor cold-start behaviour (or vice versa).

## Baselines (the comparison must be against a *strong* baseline)

A GNN beating a weak baseline proves nothing. The baseline set:
1. **Trivial**: predict base rate / majority — sanity floor.
2. **Logistic regression** on instrument raw features — linear reference (≈ the original's SGD).
3. **Gradient-boosted trees (LightGBM)** on instrument raw features **+ pre-T company aggregates** — the
   *strong* modern tabular baseline and the real bar to clear. (The original used Random Forest; gradient
   boosting is the stronger, more honest 2020s equivalent.) This is the number that matters.

All baselines use the *same* split, maturity rule, and metrics as the GNN — no comparing across different
evaluation setups.

### Baseline protocol and result (implemented 2026-08-01)

The baseline feature contract reuses the graph builder's cutoff-fitted tensors:

- logistic regression receives the 12 instrument features only;
- LightGBM receives those 12 plus the ten pre-T history features of each seller and buyer endpoint (32
  columns total). It receives no eventual-outcome aggregates, bond-graph features, or post-T statistics.

LightGBM uses fixed, conservative hyperparameters and class balancing. Its tree count is selected by
average-precision early stopping on the latest 20% of mature training instruments (validation begins
2017-12-05), then a fresh model with the selected 202 trees is refitted on all 42,321 mature training
rows. Test labels are never supplied to fitting or early stopping. Seed 42, deterministic mode, one
thread. Implementation: `src/graph_ml/baselines/tabular.py`; executed studybook:
`notebooks/02_project/02_tabular_baselines.ipynb`; committed run log: `results/baseline_metrics.csv`.

| Model | Cohort | PR-AUC | ROC AUC | Precision@top 5% | Recall@top 5% |
|---|---|---:|---:|---:|---:|
| Base rate | all | 0.056 | 0.500 | 4.95%¹ | 4.41%¹ |
| Logistic (instrument only) | all | 0.074 | 0.402 | 9.46% | 8.43% |
| **LightGBM (instrument + company)** | **all** | **0.465** | **0.913** | **49.03%** | **43.68%** |
| LightGBM | seen | 0.432 | 0.900 | 34.65% | 42.12% |
| LightGBM | cold-start | 0.387 | 0.904 | 28.83% | 13.91% |

¹All base-rate scores tie, so its top-k set follows stable row order and has no operational meaning; its
PR-AUC equals cohort prevalence and ROC AUC is 0.5, which are the actual no-ranking references.

**Interpretation:** the GNN must clear **0.465 overall PR-AUC**, not merely beat logistic regression.
Cold-start PR-AUC is lower despite high ROC AUC and a much higher 10.42% base rate, so the subgroup remains
the harder operational problem. LightGBM gain importance is dominated by endpoint history counts and
buyer-history amount/timing summaries; this supports the value of company context but does not establish
that message passing improves on fixed aggregates.

## Results logging & reproducibility

- **Results log**: a committed `results/` table (markdown or CSV) — one row per run/cohort: date, model, target,
  split/cutoff, maturity rule, metrics (PR-AUC, ROC AUC, operating point), and a pointer to the
  notebook/script that produced it. Enough to satisfy `CONSTITUTION.md` §2.3 without heavyweight tooling;
  mlflow/W&B only if the run count grows enough to justify it.
- **Seeds**: a single seed-setting utility in `src/graph_ml/` used everywhere; record the seed per run.
- **MPS caveat**: PyTorch on Apple MPS is not fully deterministic; note this next to reported numbers and,
  where exact reproducibility matters, fall back to CPU for the final reported run.

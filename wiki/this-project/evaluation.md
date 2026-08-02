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

This is a **fixed-origin, final-snapshot benchmark**, not yet a fully point-in-time deployment simulation.
The edge views prevent post-T instruments from changing historical company states, but the label and
feature-time limitations documented below still apply. A later temporal model may roll history forward,
but only with strictly time-ordered feature construction and message passing. Implementation:
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

## Leakage audit (2026-08-02)

“Temporal split” is not by itself proof of point-in-time correctness. The audit traced every v1 input,
label mask, preprocessing fit, and message-passing edge.

| Question | Finding |
|---|---|
| Do post-T feature rows enter fitted transforms or company histories? | **No.** Numeric transforms, categorical vocabularies, and company histories are fitted from pre-T instruments only. |
| Are lifecycle or eventual-outcome fields model inputs? | **No.** The 12 instrument features and 10 company-history features contain origination attributes only. LightGBM receives 32 columns formed from those same tensors. |
| Can test instruments update companies or one another during GNN inference? | **No.** Only pre-T instruments send messages into company states. Tests also flip post-T labels and verify unchanged scores. |
| Were pre-T training labels necessarily known at T? | **Not established.** The mask uses final-snapshot `has_impairment1` and `is_open` observed at A, not an event-availability state at T; lifecycle dates show material post-T activity. |
| Is every training row represented using only history earlier than that row? | **No.** All pre-T endpoint histories and the static training graph are built through T, so an early invoice can see later pre-T sibling attributes/topology. |
| Is the internal validation block point-in-time clean? | **No.** The split is chronological, but preprocessing and company histories are fitted through T, including validation-period feature distributions and topology. No validation labels enter features, but model selection is not a clean rolling-origin simulation. |

The most consequential issue is label timing. For both train and test, maturity is evaluated using final
state at A=2018-12-18. Therefore a pre-T invoice can be admitted to training because it closed or impaired
*after* T=2018-04-30. As a concrete diagnostic proxy, **3,027 / 42,321** current mature training rows
have at least one recorded lifecycle timestamp (`last_payment_date`, `discharge_date`,
`cancellation_date`, or `debt_collection_date`) on/after T; 2,932 are closed negatives and 95 positives.
These fields are incomplete and are not a definitive event-time reconstruction, so this count diagnoses
the problem rather than solving it.

Consequently, the reported LightGBM 0.465 and GraphSAGE 0.305 PR-AUC results remain useful controlled
comparisons under one shared retrospective protocol, but they must not be described as unbiased
prospective performance at T. The audit cannot certify “no leakage anywhere” until target event times and
as-of label availability are reconstructed.

The audit was also extended to the original Tier-1 and bond-graph tables. Their stored outcome histories
and propagated flows cannot be certified as-of-time, and the final bond artifacts contain cross-target
duplication/stage-mutation anomalies. They are not v1 inputs and remain prohibited unless regenerated
inside temporal folds. See `bond-graph-leakage-audit.md`.

### Required point-in-time protocol

1. Define each prediction time `t_i` (normally invoice origination/input time) and the business prediction
   horizon.
2. Supervise a row only when its outcome was knowable at that fold's training cutoff. If impairment event
   time cannot be recovered, prefer a target with an explicit due-date-plus-horizon rule (p90/p180), or
   model time-to-event/censoring directly.
3. Build endpoint histories with cumulative, shifted operations using events strictly before `t_i`; never
   include the current row or a later sibling.
4. Use rolling-origin train/validation/test folds. Fit scalers, category vocabularies, aggregates, and
   graph topology on the training window only.
5. Rebuild a time-aware LightGBM baseline and a temporal GNN from the same as-of feature/event stream.
   Only then does their comparison isolate the value of temporal message passing.

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

**Interpretation within the current retrospective protocol:** the GNN must clear **0.465 overall PR-AUC**,
not merely beat logistic regression.
Cold-start PR-AUC is lower despite high ROC AUC and a much higher 10.42% base rate, so the subgroup remains
the harder operational problem. LightGBM gain importance is dominated by endpoint history counts and
buyer-history amount/timing summaries; this supports the value of company context but does not establish
that message passing improves on fixed aggregates.

## First GNN result (2026-08-02)

The first applied model is a two-layer, relation-aware GraphSAGE with 64 hidden channels, mean
aggregation, layer normalization, ReLU, 0.2 dropout, and class-weighted binary cross-entropy. Epoch count
is selected on the same late pre-cutoff validation block used by the baseline protocol: validation
PR-AUC peaks at 0.261 at epoch 139, then a fresh seed-42 CPU model is refitted on all mature training
labels for 139 epochs. CPU plus deterministic PyTorch algorithms is used for the reported run.

| Model | Cohort | PR-AUC | ROC AUC | Precision@top 5% | Recall@top 5% |
|---|---|---:|---:|---:|---:|
| **GraphSAGE** | **all** | **0.305** | **0.818** | **42.37%** | **37.74%** |
| GraphSAGE | seen | 0.291 | 0.790 | 28.17% | 34.25% |
| GraphSAGE | cold-start | 0.319 | 0.884 | 21.62% | 10.43% |
| LightGBM | all | 0.465 | 0.913 | 49.03% | 43.68% |
| LightGBM | seen | 0.432 | 0.900 | 34.65% | 42.12% |
| LightGBM | cold-start | 0.387 | 0.904 | 28.83% | 13.91% |

**Conclusion:** GraphSAGE learns substantial signal and beats the instrument-only logistic reference,
but it does not beat LightGBM in any cohort. The overall PR-AUC gap is 0.160. Likely explanations are the
tree model's strength on structured nonlinear interactions, redundancy between two-hop aggregation and
the company histories already present in node features, dilution at very high-degree companies, and the
v1 model's intentionally limited receptive field. These are hypotheses for validation-only experiments,
not reasons to retune against the reported test period.

### Frozen-configuration seed robustness

Four additional deterministic CPU runs use exactly the same configuration with seeds 7, 19, 73, and
101. Across all five seeds:

| Cohort | Mean PR-AUC | Sample SD | Minimum | Maximum |
|---|---:|---:|---:|---:|
| all | 0.244 | 0.079 | 0.115 | 0.305 |
| seen | 0.276 | 0.085 | 0.127 | 0.337 |
| cold-start | 0.202 | 0.087 | 0.083 | 0.319 |

Seed 42 is the maximum overall and on cold-start, not a typical run. No configuration was changed after
examining these results. The variance strengthens the conclusion: v1 GraphSAGE is not competitive or
stable enough to replace LightGBM, and future neural experiments must report multiple seeds by default.

Implementation: `src/graph_ml/models/hetero_graphsage.py` and
`src/graph_ml/training/graphsage.py`; derivation: `notebooks/01_architectures/graphsage.ipynb`; applied
studybook: `notebooks/02_project/04_hetero_graphsage.ipynb`; five-seed run log:
`results/gnn_metrics.csv`.

## Results logging & reproducibility

- **Results log**: a committed `results/` table (markdown or CSV) — one row per run/cohort: date, model, target,
  split/cutoff, maturity rule, metrics (PR-AUC, ROC AUC, operating point), and a pointer to the
  notebook/script that produced it. Enough to satisfy `CONSTITUTION.md` §2.3 without heavyweight tooling;
  mlflow/W&B only if the run count grows enough to justify it.
- **Seeds**: a single seed-setting utility in `src/graph_ml/` used everywhere; record the seed per run.
- **MPS caveat**: PyTorch on Apple MPS is not fully deterministic; note this next to reported numbers and,
  where exact reproducibility matters, fall back to CPU for the final reported run.

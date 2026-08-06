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

### First implementation and viable target (2026-08-02)

`src/graph_ml/evaluation/point_in_time.py` now represents label availability explicitly and builds
rolling-origin masks that require a label to be knowable at the relevant boundary. It supports:

- event targets: positives become available at a verified event timestamp, negatives at a verified
  resolution timestamp; missing timestamps remain unsupervised;
- horizon targets: both classes become conservatively available at `due_date + horizon`.

On the current fixed T/A dates, the horizon rule gives:

| Target | Known-before-T train | Positives | Known-through-A test | Positives |
|---|---:|---:|---:|---:|
| p90 | 38,083 | 3,041 | 10,554 | 222 |
| p180 | 29,552 | 2,169 | 2,504 | 0 |

p180 is not evaluable at this cutoff because the mature test cohort has no positives. Impairment remains
the main business target but needs a verified event timestamp. **p90 is therefore the practical next
implementation target** for exercising the complete causal tabular/GNN pipeline without inventing
impairment timing. This is a protocol decision, not permission to select a target based on which test
score looks best.

### First point-in-time p90 baseline (implemented 2026-08-02)

The corrected LightGBM uses four origination-time instrument values plus strictly-prior count/mean
histories. Each seller and buyer endpoint receives both earlier seller-role and buyer-role history, so
hybrid companies remain represented. Same-timestamp invoices cannot see one another. During tree-count
selection, medians and category vocabularies are fitted only on the early training cohort; after selecting
58 trees, preprocessing and LightGBM are refitted on labels/features legally available before T.

Rolling boundaries are train end 2017-08-01, validation/deployment cutoff 2018-04-30, and end-exclusive
test boundary 2018-12-19 (therefore data through A=2018-12-18). Results:

| Cohort | Rows | Positives | Prevalence | PR-AUC | ROC AUC | Precision@5% | Recall@5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 10,554 | 222 | 2.10% | **0.079** | 0.819 | 12.88% | 30.63% |
| seen | 8,119 | 166 | 2.04% | **0.102** | 0.867 | 10.59% | 25.90% |
| cold-start | 2,435 | 56 | 2.30% | **0.026** | 0.585 | 0.00% | 0.00% |

The overall model ranks risk substantially above the 0.021 no-skill PR-AUC, but cold-start performance is
nearly at prevalence and retrieves no positives in the top 5%. Endpoint history features dominate gain;
new companies lack precisely that signal. This makes causal cold-start handling a concrete temporal-GNN
design requirement rather than a generic aspiration.

Do **not** compare 0.079 numerically with the old 0.465: p90 and impairment have different labels,
prevalences, maturity rules, and cohorts. The relevant comparison is the point-in-time p90 LightGBM
versus the temporal p90 GNN on these identical masks and feature/event availability, reported below.

Implementation: `src/graph_ml/baselines/point_in_time.py`; tests:
`tests/baselines/test_temporal_tabular.py`; executed visual studybook:
`notebooks/02_project/05_point_in_time_and_bond_audit.ipynb`; run log:
`results/point_in_time_p90_metrics.csv`.

### First causal temporal GNN (implemented 2026-08-02)

The temporal role GNN uses the identical p90 availability object, rolling fold, origination features, and
test cohorts as the corrected LightGBM. For every invoice it constructs four exponentially decayed
strictly-prior contexts, preserving both the current endpoint (seller/buyer) and the historical role in
which that company appeared. The frozen half-life is 180 days. Relation-specific learned transforms and
gates receive the context plus log-count, log-age, and history-presence metadata; a root path retains the
current invoice's own features.

Epoch count is selected on the early rolling validation block, then a fresh deterministic CPU model is
refitted on every label legally available before T. The test labels never enter preprocessing, temporal
contexts, epoch selection, or fitting. Five seeds use one frozen configuration:

| Cohort | Temporal GNN mean PR-AUC | Sample SD | Minimum | Maximum | LightGBM PR-AUC |
|---|---:|---:|---:|---:|---:|
| all | 0.053 | 0.033 | 0.023 | 0.105 | **0.079** |
| seen | 0.065 | 0.044 | 0.024 | 0.135 | **0.102** |
| cold-start | 0.023 | 0.003 | 0.019 | 0.027 | **0.026** |

The seed-42 run reaches 0.034 overall / 0.041 seen / 0.025 cold-start. Seed 73 exceeds LightGBM overall
and seen, but it is the held-out maximum, not evidence that the GNN wins; choosing it after inspecting
test scores would be leakage. The five-seed distribution is the result. On average the GNN trails
LightGBM overall and seen, is initialization-sensitive, and leaves cold-start almost exactly at its 0.023
prevalence. Four of five seeds retrieve no cold-start positives within the top 5% review budget.

This comparison closes the first causal temporal slice without establishing robust superiority. It also
motivated validation-only root, relation, decay, and recent-neighbour controls while the reported 2018
test period remained sealed. Those controls are now complete; none dominates both origins. Better
transferable current/company features for cold-start remain a separate direction.

The temporal role GNN ablations verify exact zero-message empty fallback. Shared relation transforms have
fold means 0.1374/0.0195 versus separate 0.1436/0.0143. A 60-day half-life leads fold 1 at 0.1699, while
no decay leads fold 2 at 0.0258. Recent K=8 raises both medians (0.1406→0.1520 and 0.0084→0.0123) and
reduces fold-2 SD (0.0168→0.0048), but lowers its mean from 0.0143 to 0.0109. It remains an
additional-origin candidate. Artifacts: `results/temporal_gnn_relation_ablation.csv`,
`results/temporal_gnn_decay_ablation.csv`, `results/temporal_gnn_recent_ablation.csv`;
studybook: `notebooks/02_project/10_temporal_gnn_component_ablations.ipynb`.
Multiple rolling test windows are still future work; the current result is one fixed-origin held-out
period, evaluated across multiple neural initializations.

The first predeclared diagnostic removes all relation messages while retaining the nonlinear invoice-root
network and identical rolling protocol. Across five seeds this **root-only neural control** averages
PR-AUC 0.035 overall / 0.038 seen / 0.033 cold-start. Relation messages therefore improve the overall and
seen means, but reduce cold-start from 0.033 to 0.023 and introduce much more seed variance. The graph
history is useful when it exists; empty/sparse-history fallback and gating are the next validation target.
Run log: `results/root_only_p90_metrics.csv`.

### Pre-holdout temporal backtests (implemented 2026-08-05)

The reported April–December 2018 holdout must no longer guide architecture choices. Two earlier
expanding-window backtests now provide development evidence. A fold generator guarantees that their test
boundaries do not cross the final holdout start. The high-level runner rebuilds preprocessing, selection,
refit, and cold-start identity inside every fold and rejects any train/validation/test partition with
fewer than ten examples of either class.

Four-month windows produced zero mature validation positives; an initial eight-month design left one
validation block with only nine. The accepted twelve-month folds are:

| Fold | Train end | Validation end | Development test end | Train / validation / test positives |
|---|---|---|---|---:|
| 1 | 2015-04-01 | 2016-04-01 | 2017-04-01 | 19 / 72 / 27 |
| 2 | 2016-04-01 | 2017-04-01 | 2018-04-01 | 107 / 27 / 2,106 |

PR-AUC results (neural entries are five-seed mean ± sample SD):

| Fold | Cohort | Prevalence | LightGBM | Root only | Temporal role GNN | Temporal Transformer |
|---|---|---:|---:|---:|---:|---:|
| 1 | all | 0.41% | 0.007 | 0.008 ± 0.005 | 0.012 ± 0.018 | 0.016 ± 0.011 |
| 1 | seen | 0.37% | 0.009 | 0.011 ± 0.007 | 0.006 ± 0.004 | 0.034 ± 0.028 |
| 1 | cold-start | 0.45% | 0.007 | 0.010 ± 0.007 | 0.018 ± 0.030 | 0.005 ± 0.001 |
| 2 | all | 9.77% | 0.120 | 0.094 ± 0.041 | 0.119 ± 0.046 | 0.087 ± 0.011 |
| 2 | seen | 16.22% | 0.157 | 0.138 ± 0.034 | 0.194 ± 0.071 | 0.146 ± 0.018 |
| 2 | cold-start | 7.08% | 0.119 | 0.077 ± 0.039 | 0.092 ± 0.039 | 0.067 ± 0.010 |

The experiment does not establish a universal winner. Prevalence and model ordering change materially
through time; fold 1 contains only 27 test positives and its neural ranges are correspondingly unstable.
In fold 2 the temporal GNN is effectively tied with LightGBM overall, stronger on seen companies, and
weaker on cold-start. The Transformer improves sparse fold-1 seen-company ranking, but trails both leading
families in fold 2 and does not solve cold start. Together with the final-holdout result, this supports
selective temporal context for history-rich companies but not promotion of attention on novelty alone.

The backtests remain the selection environment for bounded recent-event attention ablations.
Implementation: `src/graph_ml/backtesting.py`; fold construction:
`src/graph_ml/evaluation/point_in_time.py`; summary artifact:
`results/temporal_backtest_p90_summary.csv`; visual studybook:
`notebooks/02_project/07_temporal_backtesting.ipynb`; Transformer derivation and result:
`notebooks/02_project/08_temporal_graph_transformer.ipynb`.

The bounded input contract was implemented on 2026-08-06. Every query retains at most K newest events
per endpoint/role after applying strict `< t_i` eligibility, with aligned ages, validity masks, and source
indices. Future-feature mutation and simultaneous-event tests protect the causal mask before any
Transformer parameters exist. This separates data correctness from architecture correctness. The first
attention comparison now uses this exact contract; its run-level artifact is
`results/temporal_transformer_backtest_p90_metrics.csv`.

The first follow-up uses validation scores only. Across the same five paired seeds, learned log-age,
fixed 180-day logit decay, and no-age treatments are nearly tied: fold-1 means are 0.3016, 0.3046, and
0.2968; fold-2 means are 0.0203, 0.0199, and 0.0195. No development-test or sealed-holdout outcome is
used to choose the treatment. The negative result keeps learned time as the default and moves the next
ablation to root/message fusion. Artifact: `results/temporal_transformer_time_ablation.csv`; studybook:
`notebooks/02_project/09_model_comparison_and_time_ablation.ipynb`.

The paired fusion follow-up replaces full-strength residual addition with a scalar coverage-aware gate.
Fold 1 improves from 0.3016 to 0.4272 mean validation PR-AUC, while fold 2 falls from 0.0203 to 0.0101.
Because the direction reverses, residual fusion remains the default and no development-test labels are
consulted. Artifact: `results/temporal_transformer_fusion_ablation.csv`.

The next paired validation control is a 2×2 width/regularization design. The 64-unit current model remains
best in fold 1 (0.3016); 32-unit variants average 0.2911–0.2932. Wide strong regularization has a nominal
fold-2 mean of 0.0279, but its median is 0.0100 and one 0.0941 seed drives the mean. Compact variants are
weaker at 0.0073 and 0.0055. No treatment is promoted. Artifact:
`results/temporal_transformer_capacity_ablation.csv`.

The final Transformer control varies the causal recent-event budget K. Fold 1 favours K=2 (0.3301 mean)
and declines as history grows; fold 2 collapses at K=2 (0.0069) and favours K=8 (0.0203). K=16 doubles
K=8 tensor memory to about 222.7 MiB without improving either mean. Because no K dominates, the
predeclared K=8 configuration remains frozen. Artifacts: `results/temporal_transformer_k_ablation.csv`
and `results/temporal_transformer_k_coverage.csv`.

Implementation: `src/graph_ml/data/temporal_graph.py`,
`src/graph_ml/models/temporal_role_gnn.py`, and `src/graph_ml/training/temporal_gnn.py`; concept guide:
`wiki/gnn-concepts/temporal-role-gnn.md`; executed studybook:
`notebooks/02_project/06_temporal_role_gnn.ipynb`; run log:
`results/temporal_gnn_p90_metrics.csv`.

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

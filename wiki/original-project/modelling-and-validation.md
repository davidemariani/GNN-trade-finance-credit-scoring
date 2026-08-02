# Original project: modelling & validation methodology

## Models used

- **SGDClassifier** (`loss='log'`, i.e. logistic regression) — linear baseline.
- **RandomForestClassifier** — the strongest classical model throughout.
- **MLP** (Keras/TensorFlow) — ReLU hidden layers, sigmoid output, Adam, binary cross-entropy, dropout
  (~0.5) + batch norm, class-1 sample weighting (25-50×) for imbalance. 2 to 5 hidden layers.
- **RNN** (LSTM/GRU, 3-5 stacked layers) — applied only to p90/p180 (impairment was judged sufficiently
  solved by RF).

`imp` / `p90` / `p180` are modelled **independently end-to-end** — separate feature engineering where
applicable, separate hyperparameter tuning, separate trained models per algorithm. Not a shared/sequential
model.

## Validation strategy — the methodologically important part

Four progressively stricter modes (this rigor is worth preserving in the GNN rework):

1. **Shuffle mode** — random shuffle, ~80/20 split, `StratifiedShuffleSplit` k-fold CV. Ignores time
   entirely. Used deliberately as a "look-into-the-future" baseline — shown to **overstate** real
   performance.
2. **Time mode** — single time cutoff (30 Apr 2018 for imp/p90; 20 Feb 2018 for p180, earlier because
   p180 needs ≥6 months of runway before the Sep 2018 data end). Train = before cutoff, test = after.
   Standard k-fold CV still used within the training set.
3. **Time-sequential mode** — same cutoff, but validation uses rolling/sequential splits instead of
   k-fold CV, so training data always precedes its validation data — appropriate because k-fold CV
   assumes i.i.d. samples, which time-correlated transactions violate.
4. **Time-sequential mode with time-leak prevention** — as (3), but feature extraction (especially the
   bond-graph/shock features, which otherwise "see" the whole graph's future topology) is done
   **separately** on train/test portions per fold, not on the full dataset before splitting. The most
   rigorous mode, and the only one used to validate the enriched (bond-graph) models.

**Primary metric**: ROC AUC (validation-fold AUCs computed by aggregating fold predictions into one array,
not averaging per-fold AUCs).

## Hyperparameter tuning

- SGD & RF: random search to narrow the space, then grid search on the narrowed space, scored on
  validation AUC.
- MLP & RNN: manual/iterative tuning (architecture, learning rate, batch size, dropout, class weights) —
  automated search was computationally prohibitive.

## Why this matters for the GNN rework

The **time-leak problem** generalizes directly to a GNN setting: if node/edge features (or the graph
structure itself) used at training time reflect information only available in the future relative to a
given instrument, evaluation will be optimistic in the same way shuffle-mode was here. Any GNN evaluation
in this project should default to a time-mode or time-sequential split, matching or improving on rigor
mode (4) above — not the shuffle-mode equivalent. See `specs/roadmap.md` Phase 3.

## Applied rework audit (2026-08-02)

The v1 rework correctly blocks post-cutoff rows and edges, but it does **not yet match mode (4)** inside
the training period. Endpoint histories and static topology are constructed through the common cutoff for
every earlier row, preprocessing sees the late validation window, and label maturity uses final-snapshot
status rather than proven as-of-cutoff event availability. The v1 numbers are therefore retrospective
benchmarks. Phase 3.6 restores the original project's strongest methodological principle with a
strictly-as-of event builder and rolling-origin folds before adding temporal message passing. See
`wiki/this-project/evaluation.md`.

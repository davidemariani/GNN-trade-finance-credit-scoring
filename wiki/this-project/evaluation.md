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
- **Cutoff T**: start with the original's **2018-04-30** for impairment (comparability). Data ends
  2018-12-18, giving ~8 months of test runway.
- **Inductive**: company node features are computed from pre-T instruments only; the model trains on the
  pre-T subgraph and is evaluated on post-T instrument nodes attached to it. See `graph-design.md`.

## Label maturity / censoring (must handle — easy to get silently wrong)

`has_impairment1` is an *eventual* outcome. At any cutoff, some instruments are still `is_open` and their
final label is not yet determined: **~32% of test and ~8% of train instruments are open** at the data's
end. Training or scoring on unresolved outcomes is a survivorship/future-information problem (it is *why*
the original gave p180 six months of runway).

**Decision (v1 rule, revisit if too aggressive):** define a maturity window and exclude from *evaluation*
any instrument not yet resolved (or not yet old enough to have had the chance to resolve) by the analysis
date. Document exactly how many instruments this drops. Do not report a headline number without stating
the maturity rule used — an unstated rule makes the metric uninterpretable.

## Cold-start companies

~56% of test-period companies are unseen in training; ~25.5% of test instruments involve one. This is
handled by the node-feature choice (aggregated features → new company gets an honest zero-history vector,
not a random embedding — see `graph-design.md`), **and** results should be **reported split by
seen/cold-start** so a strong aggregate number doesn't hide poor cold-start behaviour (or vice versa).

## Baselines (the comparison must be against a *strong* baseline)

A GNN beating a weak baseline proves nothing. The baseline set:
1. **Trivial**: predict base rate / majority — sanity floor.
2. **Logistic regression** on instrument raw features — linear reference (≈ the original's SGD).
3. **Gradient-boosted trees (LightGBM)** on instrument raw features **+ pre-T company aggregates** — the
   *strong* modern tabular baseline and the real bar to clear. (The original used Random Forest; gradient
   boosting is the stronger, more honest 2020s equivalent.) This is the number that matters.

All baselines use the *same* split, maturity rule, and metrics as the GNN — no comparing across different
evaluation setups.

## Results logging & reproducibility

- **Results log**: a committed `results/` table (markdown or CSV) — one row per run: date, model, target,
  split/cutoff, maturity rule, metrics (PR-AUC, ROC AUC, operating point), and a pointer to the
  notebook/script that produced it. Enough to satisfy `CONSTITUTION.md` §2.3 without heavyweight tooling;
  mlflow/W&B only if the run count grows enough to justify it.
- **Seeds**: a single seed-setting utility in `src/graph_ml/` used everywhere; record the seed per run.
- **MPS caveat**: PyTorch on Apple MPS is not fully deterministic; note this next to reported numbers and,
  where exact reproducibility matters, fall back to CPU for the final reported run.

# Temporal role GNN

## Definition

A temporal role GNN is a message-passing model in which both **edge meaning** and **event time** determine
which information reaches a prediction. In this project, an invoice at time `t_i` receives four separate
summaries: its seller's earlier seller-role events, its seller's earlier buyer-role events, its buyer's
earlier seller-role events, and its buyer's earlier buyer-role events. Only events with `t_j < t_i` are
eligible.

This is the first applied implementation of the broader event-time principles in
[Temporal / dynamic graph learning](temporal-graphs.md). The teaching derivation and real-data experiment
are in `notebooks/02_project/06_temporal_role_gnn.ipynb`.

## Intuition

Two invoices may touch the same company but still be unusable neighbours if one did not exist at the
other's prediction time. Among legal past neighbours, a recent invoice may also be more informative than
one several years old. The model therefore combines three ideas:

1. a strict causal neighbourhood;
2. role-specific channels so selling and owing history are not collapsed;
3. recency-weighted aggregation plus count/age metadata so the model can judge the evidence behind a
   message.

For current invoice `i`, historical event `j`, and half-life `H`, the frozen recency rule is

`w_ij = exp(-log(2) * (t_i - t_j) / H)`.

The context for relation `r` is a weighted mean over only `t_j < t_i`. At `H` days an event has weight
one half; at `2H`, one quarter. Empty history becomes a zero context with an explicit history-present flag,
rather than a fabricated company embedding.

## Model update

The current invoice retains a root transformation. Each of the four relations owns independent context,
time-metadata, and gating transforms:

`h_i = Root(x_i) + sum_r sigmoid(G_r m_ir) * (W_r c_ir + U_r m_ir)`.

Here `c_ir` is the causal decayed context and `m_ir` contains log-count, log-age, and a history-present
indicator. Layer normalization, a nonlinear residual refinement, and a classifier complete the current
one-layer implementation. The architecture has 10,177 trainable parameters with 64 hidden channels.

## Leakage contract

The implementation is safe only because time enters the full pipeline:

- context queries exclude the current, simultaneous, and future events;
- input features are origination-time values, not lifecycle outcomes;
- p90 labels become available at `due_date + 90 days`;
- training, validation, and refit masks require labels to be available by their boundary;
- preprocessing is fitted within the appropriate rolling window;
- epoch selection uses validation only, followed by a fresh refit;
- the held-out test period is evaluated only after selection.

A date column added to a static full graph would not provide these guarantees. It would still allow
future topology or features to travel through messages.

## What the first result says

Across five fixed seeds, p90 PR-AUC is `0.053 ± 0.033` overall, `0.065 ± 0.044` on seen companies, and
`0.023 ± 0.003` on cold-start invoices. The causal LightGBM benchmark is `0.079`, `0.102`, and `0.026`.
One seed exceeds LightGBM overall and seen, but selecting that maximum after viewing the test set would be
test leakage. The distribution shows that this first model is not robustly stronger.

Cold-start stays approximately at its 2.30% prevalence. That is structurally understandable: temporal
history cannot help when neither endpoint has a useful history. Better current-invoice features or
transferable company descriptors are needed alongside graph improvements.

A five-seed root-only neural control sharpens this diagnosis. Removing all relation contexts lowers mean
overall/seen PR-AUC from 0.053/0.065 to 0.035/0.038, showing that causal histories do add signal. But it
raises cold-start from 0.023 to 0.033 and is substantially more stable. The current learned history gates
therefore need a better empty/sparse-history fallback before the model needs additional graph depth.

## When this architecture matters

Use this pattern when events share typed entities, ordering is essential, and a transparent aggregated
state is preferable to a recurrent memory model. It is a good causal intermediate step between a static
GraphSAGE graph and a TGN/TGAT-style event model. It does not yet learn event-to-event attention, recurrent
company memory, or long hybrid-mediated contagion paths.

The planned validation-only component sequence is complete. Fully empty history is proven to contribute
an exact zero message. Sharing relation transforms lowers parameter count and improves fold-2 stability,
but lowers fold-1 mean. Short 60-day decay leads fold 1, whereas no decay leads fold 2; the predeclared
180-day prior remains the compromise. Bounded recent K=8 improves both fold medians and sharply reduces
fold-2 variance, but not both means, so it awaits additional temporal origins rather than replacing the
frozen model. See notebook 10 and `results/temporal_gnn_*_ablation.csv`.
Repeatedly choosing changes from the reported test period would invalidate the comparison.

## Implementation pointers

- Causal history aggregation: `src/graph_ml/data/temporal.py`
- Four-channel tensors: `src/graph_ml/data/temporal_graph.py`
- Neural update: `src/graph_ml/models/temporal_role_gnn.py`
- Rolling training/refit: `src/graph_ml/training/temporal_gnn.py`
- Five-seed run log: `results/temporal_gnn_p90_metrics.csv`
- Root-only diagnostic: `results/root_only_p90_metrics.csv`
- Visual studybook: `notebooks/02_project/06_temporal_role_gnn.ipynb`

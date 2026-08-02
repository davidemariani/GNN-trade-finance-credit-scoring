# Temporal and dynamic graphs

A static graph answers “who is connected?” A temporal graph also asks **when did each connection or state
change become available?** That extra question is essential here: invoices arrive over time, repayments
and impairments are observed later, and a company history that is valid in December was not necessarily
known when an invoice was scored in April.

## The causal rule

For an instrument predicted at time `t_i`, its representation may use only events with timestamp `< t_i`.
This applies to raw features, aggregates, edges, neighbor messages, fitted preprocessing, and labels used
to train the model. A chronological train/test split does not enforce this rule inside the training
window by itself.

An event-stream view of this project is:

`instrument originates → seller/buyer states are read → risk is predicted → the event updates history`

The order matters. Updating company history with the current invoice before predicting it would leak the
row into its own context. Updating with later invoices would leak the future.

## Practical model families

- **Rolling snapshots:** build one leakage-safe graph per time window and reuse a relation-aware GNN.
  This is easiest to inspect and teach, but coarse windows can hide ordering within a snapshot.
- **Time-aware message passing:** attach event age/time encodings to edges and learn or prescribe recency
  weights. This is a natural incremental extension of the current GraphSAGE.
- **Memory-based temporal GNN:** maintain a company state that is updated after each chronological event.
  This captures longer histories efficiently, but state-reset, batching, and simultaneous-event semantics
  require careful testing.
- **Survival modeling:** predict event hazard or time-to-event rather than forcing censored open invoices
  into binary labels. This addresses label availability, not merely graph architecture.

## Recommended progression for this project

The first complete progression is now implemented for p90: a due-date-plus-horizon label clock,
strictly-as-of feature builder, rolling-origin evaluation, causal LightGBM, and a four-channel temporal
role GNN with explicit age/recency decay. The GNN is causally cleaner than the earlier static benchmark
but does not robustly beat causal LightGBM. See [Temporal role GNN](temporal-role-gnn.md).

Time was therefore a **correctness improvement**, not a promise of a higher score. The matched
time-aware LightGBM baseline was essential: it showed that correctness and predictive advantage are
separate questions. Next come validation-only component ablations and multiple rolling test windows;
learned recurrent company memory should be considered only after the transparent model is understood.

Project-specific audit and protocol: `wiki/this-project/evaluation.md`. Graph schema and temporal extension:
`wiki/this-project/graph-design.md`. Visual applied studybook:
`notebooks/02_project/06_temporal_role_gnn.ipynb`.

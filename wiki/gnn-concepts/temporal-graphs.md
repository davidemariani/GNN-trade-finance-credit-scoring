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

First recover and document reliable event timestamps and snapshot semantics. Then create a streaming
as-of feature builder and rolling-origin evaluation, and run LightGBM on that output. Next implement a
role-aware temporal GraphSAGE with explicit edge age/recency decay. Only after this transparent model is
correct should a learned-memory model be considered.

Time is therefore the next major direction, but initially as a **correctness improvement**, not a promise
of a higher score. The time-aware LightGBM baseline is essential: otherwise any gain could come from a
better data protocol rather than from the GNN.

Project-specific audit and protocol: `wiki/this-project/evaluation.md`. Graph schema and temporal extension:
`wiki/this-project/graph-design.md`.

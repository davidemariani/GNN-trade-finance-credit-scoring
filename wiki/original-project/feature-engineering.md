# Original project: feature engineering

Two tiers of hand-engineered features, merged at the instrument (row) level. This whole layer is what a
GNN rework aims to replace or reduce reliance on, by learning representations directly from the graph.

## Tier 1 — baseline "trade relationship" features

Computed per node (buyer `d_` / seller `c_`) and per edge (`cd_`), relative to the report/decision date:

- `lent_c` — number of lendings for that buyer/seller.
- `repaid_c`, `repaid_r` — count/proportion of repaid instruments.
- `impaired1_c`, `impaired_r` — count/proportion of impaired instruments.
- `pastdue90_c`, `pastdue90_r`, `pastdue180_c`, `pastdue180_r` — counts/proportions of delayed instruments.
- `trend_a` — trend of invoice amount over time (linear regression on the instrument sequence, log-transformed slope, sign preserved).
- `we_payment_share` — proportion of payments made on weekends.
- `pd_mismatch_mean`, `pd_mismatch_std` — mean/std of offset between last payment date and due date.
- Date-lifecycle offsets (`dd_*`): day-offset of each lifecycle event date (due, discharge, input, creation, debt collection, last payment, reminder, cancellation, value, first/last posting) from `invoice_date`.
- Edge-level (`cd_*`) versions of the same statistics, describing direct buyer-seller relationships.

## Tier 2 — bond-graph / "enriched" features

Derived via the bond-graph formalism (see `glossary.md`):

- Edge-level: edge effort, edge flow (per instrument/edge).
- Node-level "static": node total flow (buyers), node total effort and node energy (sellers) — computed
  per credit event as `{event}_d_node_flow`, `{event}_c_node_eff`, `{event}_energy` for
  event ∈ {imp1, p90, p180}.
- Propagated: `flow_shock_{event}` — simulated credit-event flow propagated through the network via
  `networkx.max_flow_min_cost`, summed across paths reaching a node.

## Preprocessing pipeline

A scikit-learn `DataFrameMapper` combining: `SimpleImputer` (mean imputation), a custom date→ordinal
transform, a custom outlier cap (beyond 4 standard deviations), a custom log-scaler (for
exponentially-distributed numeric features), `LabelBinarizer` (categorical → dummy), and `StandardScaler`
(zero-mean, unit-variance). Baseline models used only Tier 1; enriched models added Tier 2 on top.

## Relevance to the rework

The GNN rework's bet: a message-passing model operating directly on the transaction graph (node/edge raw
attributes, not hand-derived effort/flow/energy) should be able to learn equivalent or better structural
signal automatically. Tier 1 features remain useful as a **classical baseline** to compare against (see
`modelling-and-validation.md`, `results.md`) — the GNN doesn't need to reproduce them, it needs to beat
them honestly.

# Bond-graph leakage and artifact audit (2026-08-02)

This audit asks whether the original Tier-1/Tier-2 engineered columns can safely be reused in the new
tabular or GNN pipeline. Short answer: **not from the stored tables as they stand**. Bond-graph modelling
can be point-in-time safe, but only if every outcome, edge, and propagated path is reconstructed inside
each temporal fold using information observed before the prediction time.

## What was inspected

The audit compared the schemas and values of the four local instrument stages:

| Stage | New columns | Main concern |
|---|---:|---|
| `02_instrumentsdf_2` | base 57 | Final outcomes, status, repayments, and lifecycle fields. |
| `03_instrumentsdf_deg1stats` | +53 | Lifecycle offsets plus buyer/seller/pair histories of repayment and eventual credit events. |
| `04_instrumentsdf_bondgraph` | +20 | Target-specific effort, flow, energy, and propagated shock features. |
| `04_instrumentsdf_bondgraph2` | +5 nominally | Currency dummies and p180 effort, but many existing bond values also change. |

## Four ways leakage can enter

1. **Post-origination fields.** `payment_date_mismatch`, last payment, discharge, collection, reminder,
   cancellation, posting dates, and most `dd_*` offsets are only known after invoice origination. They are
   invalid for an origination-time prediction unless a later prediction timestamp is explicitly chosen.
2. **Outcome-history availability.** Columns such as `cd_impaired1_r`, `d_pastdue90_r`, and
   `c_repaid_r` summarize other instruments' eventual outcomes. The stored counts appear shifted by
   invoice order—the first relationship event has `lent_c=0`—which avoids the simplest self-inclusion.
   That is not enough: a previous invoice's eventual outcome may still have been unresolved when the next
   invoice arrived.
3. **Target-derived bond quantities.** The bond layer propagates those outcome histories. In the stored
   stage, `p90_edge_flow` equals `cd_pastdue90_r` wherever the latter is defined (missing first histories
   become zero); the same holds for p180. A downstream max-flow calculation cannot make an unavailable
   upstream label safe.
4. **Future topology.** Shock propagation over a graph containing later companies, relationships, or
   transaction volume reveals future network structure even if all edge attributes were otherwise safe.
   The graph must be rebuilt from the temporal prefix available to each training/validation fold.

The original thesis recognized the fourth problem and described a time-sequential leak-prevention mode.
The local precomputed artifacts do not preserve enough provenance or observation timestamps to verify
that all four conditions held for every row.

## Artifact consistency anomalies

Value-level comparison found:

- in `04_instrumentsdf_bondgraph`, `flow_shock_imp1` and `flow_shock_p180` are identical on all 59,820
  rows;
- in `04_instrumentsdf_bondgraph2`, all seven corresponding p90/p180 fields are identical row-for-row:
  edge effort, edge flow, debtor node flow, customer node effort, node flow, energy, and flow shock;
- 14 existing bond columns change between `04_instrumentsdf_bondgraph` and
  `04_instrumentsdf_bondgraph2`, although the nominal schema addition is only currency dummies and
  `p180_edge_eff`.

Because p90 and p180 labels differ, universal equality across their entire target-specific bond feature
families is a strong generation/overwrite warning. It is not proof of the exact historical bug—the source
generator and run metadata are unavailable—but it prevents treating the final table as authoritative.

## Decision for this rework

- The existing v1 LightGBM and GraphSAGE do **not** use Tier-1 lifecycle/outcome aggregates or Tier-2 bond
  columns, so these artifact problems do not contaminate the reported v1 inputs.
- All stored Tier-1 outcome and Tier-2 bond columns are denied by default in the new point-in-time feature
  contract.
- If bond features are used as an ablation later, regenerate them from raw events inside each rolling
  fold: strictly prior observable outcomes, no current-row contribution, prefix-only topology, and no
  validation/test events during fit.
- Perturbing any later outcome or edge must leave all earlier bond features unchanged. Cross-target
  identity and stage-to-stage mutation checks must also pass before modelling.

The schema guard and first causal-history primitive live in `src/graph_ml/data/temporal.py`, with tests in
`tests/data/test_temporal.py`. General evaluation audit: `wiki/this-project/evaluation.md`.

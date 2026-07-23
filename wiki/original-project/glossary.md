# Original project: glossary

## Prediction targets (three independent binary classification tasks)

| Term | Meaning |
|------|---------|
| **imp** / impairment | An instrument becomes a credit loss / is written off as uncollectable from the debtor. The most severe of the three events. Dataset column: `has_impairment1`. |
| **p90** / pastdue90 | Instrument's payment is delayed **≥90 days** past its due date. Dataset column: `is_pastdue90`. |
| **p180** / pastdue180 | Instrument's payment is delayed **≥180 days** past its due date — rarer/more severe than p90. Dataset column: `is_pastdue180`. |

These are **three separate targets**, each with its own trained model(s), not stages of one label or a
multi-class/ordinal target.

## Trade finance vocabulary

| Term | Meaning |
|------|---------|
| Instrument | An individual financed trade receivable (invoice) — the atomic row of the dataset. |
| Debtor / Buyer | The company that owes and will eventually pay the invoice (`debtor_`/`d_` columns). Network role: **flow source**. |
| Customer / Seller | The company that sold the goods/services and receives financing against the invoice (`customer_`/`c_` columns). Network role: **accumulator / energy sink**. |
| Hybrid | A company acting as both buyer and seller across different transactions (only 15 in the dataset). Network role: **capacitor** (both emits and receives). |
| Trade relationship | The aggregate of all instruments exchanged between one specific buyer and one specific seller — this defines a graph **edge**. Edge-level columns prefixed `cd_`. |
| uid | Unique instrument identifier. |
| Factoring / receivables financing / invoice financing | General terms for how sellers obtain liquidity by selling/discounting receivables early. |

## Bond graph theory terms (mapped onto the buyer-seller network)

Bond graph theory is a physics formalism for modelling energy/power flow in dynamic systems (mechanical,
electrical, ...). The thesis maps its core concepts onto the trade-finance network as follows:

| Term | Physical analogy | Meaning in this project |
|------|-------------------|--------------------------|
| **Effort** | Force / voltage | For impairment: sum of total purchase amount between a given buyer and seller. For delays: number of instruments between a given buyer and seller. Direction: **seller → buyer**. |
| **Flow** | Velocity / current | For impairment: proportion of impaired amount between a given buyer and seller. For delays: accumulated delay × ratio of delayed instruments between a given buyer and seller. Direction: **buyer → seller** (financial stress being absorbed by the seller). |
| **Energy** | Power (force × velocity) | Computed at seller (accumulator) nodes as total effort × total flow. |
| Flow source | — | Buyers — can only emit flow, not receive it. |
| Accumulator / sink | — | Sellers — only receive flow, dissipate/accumulate it, emit effort. |
| Capacitor | — | Hybrids — both emit and receive flow and effort; "the most interesting element from the shock propagation point of view." |
| **Shock propagation** | — | Simulating how a credit-event "shock" propagates from a buyer, through chains of hybrid nodes, to seller sinks, using `networkx.max_flow_min_cost` on directed sub-graphs (edge weight = inverse of edge flow, edge capacity = buyer's energy). Produces the `flow_shock_{event}` feature. |
| Time leak | — | Because bond-graph/shock features reflect the *entire* network's topology (including future information relative to any instrument's invoice date), computing them once over the full dataset before a train/test split leaks the future into training. Motivates the "time-sequential with time-leak prevention" validation mode — see `modelling-and-validation.md`. |

## Engineered feature naming convention

- `_c` suffix — count
- `_r` suffix — ratio
- `_a` suffix — amount-derived
- `_mean` / `_std` — distribution statistics
- `d_` / `c_` prefix — node-level, buyer (debtor) / seller (customer) perspective
- `cd_` prefix — edge-level (buyer-seller pair)
- `dd_*` — day-offset of a lifecycle event date from `invoice_date` (e.g. `dd_due_date`, `dd_discharge_date`)

# Original project: data & network construction

## Dataset

- **Source**: an anonymised factoring-company transaction dataset supplied by Tradeteq. Only transaction
  information was used — no companies' accounting data.
- **File**: `00_transactionsdf_simNames.pkl` — **163,437 rows × 37 columns**. The `simNames` suffix
  signals company names throughout are **simulated/fictitious**, not real — confirmed anonymized.
- **Time period**: August 2013 – September 2018 (~5 years / 61 months).
- **Network size**: **3,219 buyers**, **115 sellers**, **15 hybrids** (both buyer and seller).
- **Currencies**: multiple — CHF, EUR, USD, GBP present as categorical features.
- **Topology**: the graph decomposes into 44 connected components; the largest holds ~77% of all nodes
  (78% of buyers, 93% of hybrids, 41% of sellers) — most analysis focused on this dominant component.
  The network is **non-stationary**: node/edge counts grow sharply after 2017, analysed via rolling
  ~20-day time windows across the full date range (never more than ~500 actively-trading nodes in any
  window).
- **Target variables**: `has_impairment1` (imp), `is_pastdue90` (p90), `is_pastdue180` (p180) — all rare/
  imbalanced binary events at the instrument level. See `glossary.md` for precise definitions.

**Status for this rework**: whether the original anonymized dataset is still accessible is an open
question — see `BACKLOG.md` / `specs/roadmap.md` Phase 1. The public `networkAnalysisForML` GitHub repo
does not contain the raw 163k-row file; only smaller derived dashboard artifacts (`base_graph.pkl`,
`network_edges.pkl`, `network_nodes.pkl`) were found there.

## Network construction

1. **Nodes** = individual companies, classified as **buyer** (debtor only), **seller** (customer only),
   or **hybrid** (both roles across different transactions).
2. **Edges** = a **trade relationship**: the aggregate of all financed instruments exchanged between one
   specific buyer-seller pair. Multiple individual instruments between the same pair collapse into one
   edge.
3. Built with **NetworkX**, visualised with **Bokeh** using a spring layout (node size ∝ betweenness
   centrality) and a concentric-circle layout (sellers by centrality toward the centre). Common structural
   motifs found: one buyer↔one seller; many buyers↔one seller (most common); many sellers↔one buyer;
   hybrids bridging both; longer buyer-seller-buyer-seller chains, some running through hybrids.
4. **Directionality**: once bond-graph modelling is applied, edges become directed — flow flows
   buyer → seller, effort flows seller → buyer (see `glossary.md`).
5. **Non-stationarity handling**: node/edge statistics computed per rolling ~20-day window (95 windows
   total across the date range), including only companies actively trading within that window.

This is the closest analogue to what `src/graph_ml/data/` needs to reproduce (or approximate, if using a
different/synthetic dataset) for the GNN rework: buyer/seller/hybrid node typing, buyer-seller trade
relationships as edges, and a time-aware (not static) view of the graph.

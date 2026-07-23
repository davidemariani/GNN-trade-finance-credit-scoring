# This project: graph design (v1, decided 2026-07-23)

The graph construction and task-framing decisions for the rework, with rationale. Supersedes guessing —
this is what actually gets built first. See `wiki/original-project/` for the thesis this is reworking and
`wiki/original-project/limitations-and-motivation-for-gnn.md` for why a naive port of the original design
wasn't the goal.

## Schema: heterogeneous, instrument-centric, static (v1)

**Three node types:**
- **Instrument** (59,820 nodes, from `04_instrumentsdf_bondgraph2.pkl`'s row set, but see "features" below
  for which columns actually get used) — one per invoice. This is where the label lives
  (`has_impairment1` / `is_pastdue90` / `is_pastdue180`).
- **Buyer** (3,176 nodes, keyed by `debtor_id`).
- **Seller** (132 nodes, keyed by `customer_id`).

**Two edge types**, both from the instrument node: `instrument→buyer` and `instrument→seller`. No direct
buyer↔seller edges — the relationship between a specific buyer and seller is already fully represented by
every instrument node that links to both.

**Confirmed empirically (not assumed) 2026-07-23**: `customer_id` and `debtor_id` in
`04_instrumentsdf_bondgraph2.pkl` have **zero overlapping values** — no hybrid-identity merging needed for
v1 (contradicts the original report's "15 hybrids," likely a name-based match in an earlier-stage table
the report used; not investigated further since it doesn't affect this dataset). Degree distribution is
heavily skewed: sellers 1-5,636 instruments (mean 453), buyers 1-1,892 (median 5) — a few clear hub nodes.

## Why this shape (not the alternatives)

- **Not a company-only graph** (nodes = buyers/sellers, like the original): the label is per-instrument;
  forcing it onto a company-level graph would require re-aggregating multiple instruments' outcomes onto
  one node/edge, throwing away the exact resolution the task needs and reintroducing hand-aggregation.
- **Not a homogeneous instrument-only graph** (edge between instruments sharing a buyer/seller): projecting
  a bipartite/star structure onto one side turns hubs into cliques — the largest seller (5,636 instruments)
  would produce up to ~15.9M edges projected directly, vs. 5,636 edges routed through an explicit seller
  node. Keeping buyer/seller as real intermediate nodes avoids this and lets them get their own learned
  identity/embedding rather than being reduced to "an edge exists."
- **Heterogeneous, not homogeneous**: buyers, sellers, and instruments are structurally and semantically
  different things — PyTorch Geometric's `HeteroData` + relation-aware convs exist for exactly this, and
  using them matches the tool to the problem's actual structure rather than flattening it.
- **Static graph for v1, not temporal snapshots**: ~63k nodes / ~120k edges fits as one full-batch graph in
  memory — no neighbor sampling or mini-batching infrastructure needed yet. Isolates "does message-passing
  over the transaction graph help at all" before adding the harder temporal-dynamics axis (deferred, per
  the owner's explicit direction to start simpler).

## The leakage problem a "simple" graph doesn't get to skip

In a static graph, a buyer/seller node's embedding aggregates messages from **all** its instrument
neighbors, including ones dated after any test cutoff — so even an honest instrument-level time split can
leak future information into training through the shared buyer/seller node, structurally reproducing the
exact time-leak failure mode `wiki/original-project/modelling-and-validation.md` describes (just relocated
from hand-engineered features into message passing itself).

**Decision**: treat this as **inductive** learning, not transductive. Train on the subgraph built from
instruments dated before cutoff T (plus the buyer/seller nodes they touch). At evaluation, extend the graph
with post-T instrument nodes attaching to the already-trained buyer/seller nodes, and predict only on those
new nodes — no forward/backward pass ever lets post-T information reach pre-T embeddings. This is
GraphSAGE's native setting, which is a concrete, motivated reason to reach for it early in
`specs/roadmap.md` Phase 4, not just because it's next in the learning progression.

## Node features

- **Instrument nodes**: raw attributes only (amount, currency, dates converted to safe relative features,
  transaction/factoring type) — the "Tier-0" layer from `00_transactionsdf_simNames.pkl` /
  `01_instrumentsdf.pkl`. Deliberately **not** the original's hand-engineered Tier 1/Tier 2 features
  (`cd_*`, `imp_*`, `flow_shock_*`, see `wiki/original-project/feature-engineering.md`) — reproducing those
  on the node would defeat the point of using a GNN.
- **Buyer/seller nodes**: **pure learned embeddings, no intrinsic features**, initialized randomly and
  shaped entirely by message passing from neighboring instruments. Decided over adding hand-computed static
  attributes, to get the cleanest possible test of whether graph structure alone captures useful signal,
  and to avoid smuggling leakage in through a seemingly-static aggregate.

## Task framing

**Node classification** on instrument nodes. **Impairment only for v1** (clearest signal, and the original
project's own strongest result — RF 0.954 AUC, `wiki/original-project/results.md` — to compare against);
p90 and p180 are deferred until the pipeline is validated end-to-end on one target, not built multi-task
from the start. Revisit multi-task (one model, three heads) once the single-target pipeline works.

## Open items (not yet decided, revisit when reached)

- Exact inductive train/test cutoff date (mirror the original's 30 Apr 2018 for impairment, or re-derive).
- Whether instrument-node dates need any transform to avoid leaking absolute calendar position as a
  shortcut feature.
- Self-loops / normalization details — implementation-level, decide when writing `src/graph_ml/data/`.

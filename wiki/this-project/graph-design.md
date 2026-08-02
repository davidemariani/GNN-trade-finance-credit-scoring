# This project: graph design (v1, decided 2026-07-23, revised 2026-07-24)

The graph construction and task-framing decisions for the rework, with rationale. Supersedes guessing —
this is what actually gets built first. See `wiki/original-project/` for the thesis this is reworking,
`wiki/original-project/limitations-and-motivation-for-gnn.md` for why a naive port of the original design
wasn't the goal, and `wiki/this-project/evaluation.md` for how models built on this graph get measured.

> **Revision note (2026-07-24)**: the original v1 draft used three *disjoint* node types
> (instrument/buyer/seller), justified partly by an incorrect "zero buyer/seller overlap" finding. That
> finding was an artifact of checking IDs only — see the hybrid correction below. The schema is now
> **company + instrument** (two node types), which handles hybrids and network contagion correctly. The
> earlier draft also paired "pure learned embeddings" with an "inductive split"; those contradict each
> other and the node-feature decision has been revised (see "Node features").

## Entity resolution: companies are matched by NAME, not ID

`customer_id` (seller role) and `debtor_id` (buyer role) live in **separate ID spaces** — the same company
gets a *different* ID depending on which side of a transaction it is on. Matching on ID therefore finds
zero overlap, which is misleading. Matching on **normalized company name** finds exactly **15 hybrid
companies** (matching the original report), i.e. companies that act as both seller and buyer across
different instruments. Graph construction must therefore assign a **canonical company node per unique
company name**, unifying its seller-role and buyer-role appearances into one node.

Hybrids are not a rounding error: instruments touching at least one hybrid are **18.7% of the dataset**
(12,465 / 66,593), and every hybrid appears on both sides. They are the genuine bridge/"capacitor" nodes
that make buyer→seller→buyer contagion paths exist — the exact mechanism the original bond-graph "shock
propagation" feature was built to hand-encode. This *strengthens* the motivation for a graph approach.

## Schema: heterogeneous, two node types, static (v1)

**Two node types:**
- **Instrument** (~59,820 nodes) — one per invoice. This is where the label lives (`has_impairment1` /
  `is_pastdue90` / `is_pastdue180`) and where prediction happens.
- **Company** (canonical per company name; ~3,200+ nodes) — unifies seller-role and buyer-role identity.
  A hybrid is simply a company node that has both incoming and outgoing instrument edges — no special case.

**Two edge types** (relations), both between instrument and company, distinguished by role:
- `(instrument) --sold_by--> (company)` — the company is the **seller/customer** on this instrument.
- `(instrument) --owed_by--> (company)` — the company is the **buyer/debtor** on this instrument.

PyG stores each relation with reverse edges so messages flow both ways; the role distinction is preserved
by relation type (this is what lets the model treat "money I'm owed as a seller" differently from "money I
owe as a buyer" — the asymmetry the bond-graph effort/flow direction encoded by hand).

## Why this shape (not the alternatives)

- **Not a company-only graph** (nodes = companies, edges = trade relationships, like the original): the
  label is per-instrument; collapsing instruments onto company edges would re-aggregate multiple
  instruments' outcomes and throw away the resolution the task needs.
- **Not a homogeneous instrument-only graph** (edge between instruments sharing a company): projecting the
  bipartite structure onto one side turns hubs into cliques — the largest seller (5,636 instruments) would
  produce ~15.9M edges projected directly, vs. 5,636 edges routed through one company node. Company nodes
  as real intermediaries avoid this.
- **Company as one type, not disjoint buyer/seller types**: three disjoint types cannot represent a hybrid
  (a node can't be in two `HeteroData` types at once), and disjoint types sever the buyer→seller→buyer
  bridge paths that carry contagion signal. One company type with role-typed *edges* keeps the role
  distinction where it belongs (the relation) while keeping the entity singular.
- **Static graph for v1, not temporal snapshots**: the full graph (~63k nodes / ~120k edges) fits as one
  full-batch graph in memory — no neighbor sampling infra needed yet. Isolates "does message passing over
  the transaction graph help at all" before adding the temporal-dynamics axis (deferred, per the owner's
  direction to start simpler). See "Known simplification" below for the honest caveat this carries.

## Node features (revised — resolves the embedding/inductive contradiction)

- **Instrument nodes**: origination-time "Tier-0" attributes only, implemented as log invoice amount,
  purchase/invoice ratio, payment-term days, input-lag days, and cutoff-fitted one-hot encodings of
  currency and factoring type. Numeric medians/means/scales and categorical vocabularies are fitted on
  pre-T instruments only; post-T categories unseen during fitting map to an explicit `__unknown__`
  feature. Lifecycle fields such as `purchase_amount_open`, last payment, discharge, and final document
  status are excluded because they reveal information that was not necessarily available when the
  instrument originated. The original hand-engineered Tier 1/Tier 2 features (`cd_*`, `imp_*`,
  `flow_shock_*`) are also deliberately excluded — reproducing those defeats the point of using a GNN.
- **Company nodes**: **features aggregated from that company's instruments dated strictly before the split
  cutoff T** — NOT a per-node learned embedding, and NOT computed over the whole dataset. The implemented
  v1 vector contains seller-role and buyer-role history counts plus role-specific means of the four
  numeric origination features above. A post-T-only company gets an all-zero vector.
  - *Why v1 does not yet include prior outcome rates*: the instrument table holds eventual impairment/
    repayment outcomes, not a proven as-of-T state. Aggregating those final outcomes merely because an
    invoice originated before T could leak resolutions that happened after T. Outcome-history aggregates
    stay deferred until a verified event-time source can establish what was actually known at each
    prediction time. This is more conservative than the earlier illustrative list of company aggregates.
  - *Why not pure learned embeddings (the earlier draft's choice)*: a per-node learned embedding is
    inherently **transductive** — a company absent from training has no trained vector. On this data
    **56% of test-period companies are unseen in training**, and **25.5% of test instruments involve a
    cold-start company**, so pure embeddings would make a quarter of the predictions we care about on
    random vectors. Aggregated features degrade gracefully instead: a genuinely new company simply has a
    zero-history feature vector, which is the honest representation of "no track record."
  - *Why time-windowed (pre-T only)*: computing company aggregates over the full dataset would leak future
    outcomes into the company node, the same failure mode described below. Restricting to pre-T instruments
    keeps post-T rows out and supports cold-start evaluation. It does not make histories point-in-time for
    every earlier training row; that limitation is handled below.

## Leakage: two distinct graph problems, only one fully controlled in v1

1. **Test-set leakage (must eliminate).** A company node aggregating over *all* its instruments — or a
   learned embedding shaped by post-T message passing — lets future information reach the representation
   used to score test instruments. **Fix**: inductive setup — company features are computed from pre-T
   instruments only, and the model is trained on the pre-T subgraph, then evaluated on post-T instrument
   nodes attached to it. No forward/backward pass lets post-T info reach the trained parameters.
2. **Intra-training temporal leakage (known v1 limitation).** Within the training subgraph, a company
   node still aggregates messages from all its *pre-T* instruments, so an early-2018 instrument's
   representation can be informed by a later (still pre-T) sibling. The same applies to the cutoff-wide
   company aggregates used by LightGBM. This does not expose post-T test labels, but it can make training
   easier than deployment and can bias early-stopping choices because validation-period features/topology
   were already included in preprocessing. The fully-correct treatment is an **event-time graph** in
   which each prediction receives messages only from strictly earlier events. The separate final-snapshot
   label-timing problem is documented in `evaluation.md` and must be fixed at the same time.

## Task framing

**Node classification** on instrument nodes. **Impairment only for v1** (the original's strongest,
clearest signal); p90/p180 deferred until the single-target pipeline is validated end-to-end, then
revisit multi-task. Metrics, the label-maturity/censoring rule, the split, and the baseline set are
defined in `wiki/this-project/evaluation.md` — not here, to keep one home per concern.

## Implemented graph artifact (2026-07-27)

`src/graph_ml/data/graph.py` now constructs this schema as a PyG `HeteroData` object, with deterministic
instrument/company mappings returned as separate metadata. Instruments are ordered by invoice date then
UID; companies are ordered by their conservative normalized-name key. The four stored directed relations
are `instrument → sold_by → company`, its `company → sells → instrument` reverse, `instrument → owed_by →
company`, and its `company → owes → instrument` reverse.

On `02_instrumentsdf_2.parquet` at T=2018-04-30 the build produces:

- 59,820 instrument nodes with 12 features;
- 3,349 canonical company nodes with 10 history features;
- 119,640 underlying role edges (239,280 directed edges when the required reverse stores are counted);
- 46,102 pre-cutoff and 13,718 post-cutoff instruments;
- 849 companies with zero pre-cutoff history.

The graph carries a `pre_cutoff_mask` as temporal metadata. The evaluation layer turns it into label and
edge-view masks, but the 2026-08-02 audit established that these are retrospective final-snapshot masks,
not yet fully as-of-time labels. The implementation is explained end-to-end in
`notebooks/02_project/00_graph_construction.ipynb` and tested with hand-built in-memory tables in
`tests/data/test_graph.py`.

## Known simplification of v1 (honest scope)

With a 2-layer GNN, an instrument's receptive field reaches its companies (1 hop) and its sibling
instruments of the same companies (2 hops) — which overlaps heavily with what the original's Tier-1
`d_*`/`c_*` aggregates already encoded by hand. Genuine multi-hop contagion (buyer→shared-seller→another
buyer, i.e. through hybrids) needs ≥3-4 hops and runs into over-smoothing. So v1 is a fair test of "does
same-company aggregation, learned rather than hand-built, help?" — but it is **not yet** a test of the
network-contagion hypothesis. That is an explicit later step (deeper/temporal models), and pretending v1
covers it would be overclaiming.

## First model-level decisions (2026-08-02)

The v1 model is two-layer relation-aware GraphSAGE. Each typed relation owns a mean-aggregator
transformation; relation outputs are summed at their destination, followed by node-wise layer
normalization, ReLU, and dropout. PyG's `SAGEConv` root transformation preserves a node's own state, so
no synthetic homogeneous self-loop relation is added. The second layer computes only instrument
destinations because company outputs from that layer would not affect the instrument classifier and
would create dead parameters.

Full-batch aggregation is deliberate: the graph fits comfortably in memory, so neighbour sampling would
add variance and infrastructure without answering a scaling problem. Architecture studybook:
`notebooks/01_architectures/graphsage.ipynb`; implementation:
`src/graph_ml/models/hetero_graphsage.py`.

## Next graph design: causal time-aware messages

The next applied graph should keep the same company/instrument identity and role-typed relations while
turning each instrument origination into a timestamped event. At prediction time `t_i`, only events with
time `< t_i` may update seller/buyer company state. Start with an interpretable snapshot/event GraphSAGE:
add edge age or a learned time encoding, decay or attend to older neighbors, keep seller and buyer state
role-specific, and update company memory in chronological order. Recency-based neighbor sampling can
prevent large hubs from washing out recent evidence. A memory-based temporal GNN is a later candidate,
after the event stream and label-time contract are verified. See `wiki/gnn-concepts/temporal-graphs.md`.

Useful static ablations remain—root-only neural baseline, removal of pre-aggregated histories, relation
collapse, degree-aware aggregation, and multiple seeds—but temporal correctness takes priority over
architecture tuning.

### First causal temporal implementation (2026-08-02)

The first implementation now materializes each invoice's strictly-prior event context through four
channels: seller endpoint in seller role, seller endpoint in buyer role, buyer endpoint in seller role,
and buyer endpoint in buyer role. Within each channel, origination-safe event features receive
exponential recency weights with a frozen 180-day half-life. Log history count, log age of the latest
event, and a history-present indicator accompany each context.

`TemporalRoleGNN` gives every channel independent message, time-metadata, and gate transforms, while a
root path preserves the current invoice's own attributes. This is a causal one-layer bipartite message
model over aggregated event histories, not yet a recurrent company-memory TGN. It eliminates the static
model's within-training future-sibling visibility without using target-derived bond features.

Five fixed seeds average p90 PR-AUC 0.053 overall versus causal LightGBM's 0.079. The wide overall seed
variation and near-prevalence cold-start results mean the implementation establishes a correct temporal
foundation, not a model win. Validation-only ablations should now isolate the root path, role separation,
decay, and hub aggregation before adding recurrent memory or longer contagion paths. See
`wiki/gnn-concepts/temporal-role-gnn.md` and
`notebooks/02_project/06_temporal_role_gnn.ipynb`.

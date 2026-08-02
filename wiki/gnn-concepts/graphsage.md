# GraphSAGE

## Definition

GraphSAGE is an inductive message-passing framework: it learns functions that aggregate a node's
features and its neighbours' features instead of learning one embedding parameter per node. The learned
function can therefore be applied to nodes that were absent during training, provided their features and
neighbourhood are available.

For the mean-aggregator form used here, one layer can be written as

$$m_v^{(k)} = \operatorname{mean}_{u \in \mathcal{N}(v)} h_u^{(k-1)},$$

$$h_v^{(k)} = \sigma\left(W_{self}^{(k)}h_v^{(k-1)} +
W_{neigh}^{(k)}m_v^{(k)} + b^{(k)}\right).$$

The mean is permutation invariant: reordering a node's neighbours does not change the result. The two
weight matrices distinguish the node's current state from the summary arriving from its neighbourhood.

## Intuition

Think of each node as asking for a fixed-size summary of a variable-size set of neighbours. The model
learns how to combine that summary with what the node already knows. After two layers, information can
cross two edges; depth therefore controls the receptive field as well as model capacity.

“SAGE” originally abbreviates *sample and aggregate*. Sampling is a scalability device, not the defining
inductive property. This project's approximately 63,000-node graph fits in memory, so the first model uses
all visible neighbours in a deterministic full-batch pass.

## Heterogeneous extension used here

Vanilla GraphSAGE assumes one relation semantics. The trade-finance model instead gives each typed edge
relation its own GraphSAGE transformation and sums relation-specific outputs at their common destination.
For relation $r$,

$$h_v^{(k)} = \sigma\left(\sum_{r \in \mathcal{R}(v)}
\left[W_{self,r}^{(k)}h_v^{(k-1)} + W_{neigh,r}^{(k)}m_{v,r}^{(k)}\right]\right).$$

This preserves `sold_by` versus `owed_by` and both reverse directions. PyG's `SAGEConv` supplies the
self/root transformation directly, so the graph does not need invented same-type self-loop edges.

## When it matters here

Post-cutoff invoices and many endpoint companies are unseen during fitting. Feature-based GraphSAGE can
score them without an ID lookup table. Two layers test whether learned same-company aggregation improves
on fixed endpoint histories; they do not yet test longer buyer→hybrid→buyer contagion paths.

Studybook: `notebooks/01_architectures/graphsage.ipynb`. Project design:
`wiki/this-project/graph-design.md`. Original source: Hamilton, Ying, and Leskovec,
[*Inductive Representation Learning on Large Graphs*](https://arxiv.org/abs/1706.02216), NeurIPS 2017.

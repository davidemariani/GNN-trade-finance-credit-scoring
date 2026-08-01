# Heterogeneous graphs

## Definition

A heterogeneous graph has more than one node type, edge/relation type, or both. Instead of treating every
neighbor interaction identically, a heterogeneous GNN can use relation-specific transformations.

PyTorch Geometric identifies a relation with a triple:

`(source node type, relation name, destination node type)`.

## Intuition

The meaning of a connection changes with its role. An invoice being *sold by* a company is economically
different from the same invoice being *owed by* another company. Encoding those as separate relations
preserves that distinction without inventing two identities for a company that participates in both roles.

## When it matters here

This graph has `instrument` and `company` nodes and two business relations, `sold_by` and `owed_by`, plus
their reverses for message flow. A single company node may participate in both roles: the 15 hybrids are
precisely the bridges needed for buyer→seller→buyer paths. Splitting buyer and seller into disjoint node
types would sever those paths; collapsing the relations would discard their direction and meaning.

See `notebooks/02_project/00_graph_construction.ipynb` for the studybook treatment and
`wiki/this-project/graph-design.md` for the project-specific design rationale.

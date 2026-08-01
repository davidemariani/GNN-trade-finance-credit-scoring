# Graph representations

## Definition

A graph is a pair `G = (V, E)`: a set of nodes `V` and relationships `E`. Machine-learning code also
associates feature vectors with nodes and/or edges, and may attach prediction labels at node, edge, or
whole-graph level.

Common storage forms:

- **Dense adjacency matrix**: an `|V| × |V|` matrix whose `(i, j)` entry records whether/how node `i`
  connects to node `j`. Simple algebraically, but costs `O(|V|²)` memory even for a sparse graph.
- **Edge list / coordinate list**: two aligned arrays of source and destination indices, costing `O(|E|)`.
  PyTorch Geometric's `edge_index` is a `[2, E]` tensor in this form.
- **Sparse adjacency matrix**: retains matrix operations without allocating absent edges; useful in graph
  convolution implementations.

## Intuition

The representation should pay for relationships that exist, not every relationship that could exist.
Real transaction networks are sparse, so an edge list is natural. Integer node indices point into dense
feature matrices (`x`), while external identifiers remain metadata rather than model inputs.

## When it matters here

The project stores 59,820 instruments and 3,349 companies. Every instrument has one seller and one buyer,
so there are only 119,640 underlying role edges. PyG stores each relation separately and also stores its
reverse for bidirectional message flow. See
`notebooks/02_project/00_graph_construction.ipynb` for the worked explanation and
`src/graph_ml/data/graph.py` for the implementation.

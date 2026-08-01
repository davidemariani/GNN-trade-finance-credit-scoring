# Graph topology: degree, components, and ego graphs

## Definition

- **Degree** is the number of edges incident to a node; typed graphs can report a degree per relation.
- A **hub** is a node with unusually high degree.
- A **connected component** is a maximal node set in which every pair is joined by some path.
- An **ego graph** is the bounded neighborhood around one focal node.

## Intuition

Degree describes local connectivity, while components describe global reachability. Message passing can
aggregate many signals at a hub but can never cross between disconnected components. Heavy-tailed degree
means most nodes have few neighbors while a small number have orders of magnitude more, making aggregation
normalization important.

## When it matters here

The filtered graph has median company degree 5 but maximum 5,636, and 45 components with 81.55% of all
nodes in the largest. Its 15 buyer/seller hybrids bridge role directions and touch 20.84% of modelling
instruments. Anonymous bounded ego graphs expose these motifs without publishing company identities.

See `notebooks/02_project/03_eda_and_topology.ipynb` and `wiki/this-project/visualization.md`.

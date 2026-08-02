# Original project: limitations & motivation for this rework

## What the thesis itself flagged (2019, its own words)

From the conclusions chapter:

1. **Feature selection** — "a more strict process of features selection would probably bring additional
   advantages, reducing the noise and discarding redundant information."
2. **Bond-graph modelling needs iteration** — "The process of bond graph modelling also needs to be
   further explored for payment delays, iterating through different metrics for effort and flow
   calculation" — the effort/flow definitions for delays were a first pass, not final.
3. **Neural network maturity** — "The work on neural networks architectures still needs improvement,
   research and development, especially on the recurrent neural networks side."
4. **Other architectures unexplored** — "The effectiveness of other configurations like 1-D convolutional
   neural networks must be explored."
5. **The explicit GNN pointer** — *"working towards neural ensembles and graph neural networks seems to
   be the natural prosecution of the project."* The literature review had already surveyed GNN
   fundamentals (Niepert, Ahmed & Kutzkov; Zhou et al.; Battaglia et al.) but deliberately deferred them
   in favor of the hand-engineered bond-graph approach used in the thesis itself.

## Why this rework can't just be "the same thing, in PyTorch" — 2019 vs. now

The thesis is a *reference point*, not a spec to reproduce with different tooling. Graph ML has moved
substantially since 2019, and several aspects of the original approach are dated in ways that matter for
how this rework should actually be designed — not just which library implements the model:

- **The graph is heterogeneous, and the original approach didn't treat it as one.** Buyers, sellers, and
  hybrids play structurally different roles (flow source / accumulator / capacitor, in the thesis's own
  bond-graph framing) — that's a heterogeneous graph by definition. A plain GCN/GraphSAGE/GAT applied
  naively (treating all nodes as one type) throws that structure away. Modern heterogeneous GNN approaches
  (relation-specific message passing, e.g. R-GCN-style typed edges, or heterogeneous graph transformers)
  are a more faithful fit and didn't exist in mature, easily-usable form in 2019.
- **The graph is temporal/dynamic, and the original approach worked around this with hand-rolled rolling
  windows rather than a model that natively understands time.** The thesis explicitly notes the network is
  non-stationary and resorts to ~20-day rolling windows plus careful time-sequential validation to avoid
  leakage. Temporal graph learning (e.g. time-aware message passing, memory-based temporal GNNs) has
  matured significantly since 2019 and is a more natural fit than treating time purely as a
  validation-split concern bolted onto an otherwise static-graph feature pipeline.
- **Hand-engineered bond-graph features were a workaround for the absence of a good graph representation
  learner, not an end in themselves.** The entire effort/flow/energy feature engineering exercise (Tier 2
  in `feature-engineering.md`) exists to inject "networked" signal into models that can't otherwise see
  graph structure (SGD, RF, MLP, RNN — none of which take a graph as input). A GNN's entire premise is
  that it learns this signal directly from the graph; a good rework result would be a GNN that
  meets-or-beats the enriched RF *without* needing bond-graph feature engineering as an input, not a GNN
  bolted on top of the same hand-crafted features.
- **Hyperparameter tuning and experiment tooling have moved on.** Manual/iterative tuning for MLP/RNN
  (necessary in 2019 because automated search was "computationally prohibitive") is no longer the state of
  the art — modern tools (e.g. Optuna) make principled search practical even for neural architectures, and
  should be used here rather than repeating manual tuning by hand.
- **What *should* carry forward unchanged**: the time-leak-aware validation discipline
  (`modelling-and-validation.md`) is not "2019 technology" — avoiding future information leaking into
  training is a permanent correctness requirement, and if anything a GNN needs *more* care here (a node's
  learned embedding can leak future graph structure just as easily as a hand-engineered feature could).
  This principle should be preserved and adapted, not modernized away.

## What this means concretely for `specs/roadmap.md`

The architecture progression in `notebooks/01_architectures/` (GCN → GraphSAGE → GAT → GIN) is still the
right **learning path** — understanding the fundamentals in their original, simpler form before reaching
for anything more elaborate is good pedagogy. But the model(s) actually **applied** to this project's task
in `notebooks/02_project/` / `src/graph_ml/models/` should be chosen with the heterogeneous + temporal
nature of the graph in mind, not just "whichever foundational architecture we learned first." This is a
design decision to make explicitly once we're at that stage of the roadmap, not something to lock in now.

That decision has now been exercised once: relation-aware GraphSAGE was the appropriate simple v1 model,
but it underperformed LightGBM and remained static. The 2026-08-02 audit makes temporal reconstruction the
next applied step—not because time guarantees a higher score, but because the comparison is not fully
deployment-like until labels, aggregates, preprocessing, and messages all obey event order. The project
will establish a point-in-time tabular baseline before attributing any change to a temporal GNN. See
`wiki/this-project/evaluation.md` and `wiki/gnn-concepts/temporal-graphs.md`.

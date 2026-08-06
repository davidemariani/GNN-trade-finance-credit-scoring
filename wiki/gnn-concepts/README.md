# GNN concepts

A growing reference of graph ML concepts, written as lookup entries (definition, intuition, when it
matters, pointer to the notebook that teaches it in depth). Starts empty and fills in alongside
`specs/roadmap.md` Phase 2 (foundations) and Phase 4 (architectures) — see
`specs/instructions/wiki-standards.md` for the convention to follow when adding a page.

Planned entries (create as each is actually covered — don't pre-write stubs with no content):

**Foundations** (mirrors `notebooks/00_foundations/`)
- [Graph representations](graph-representations.md) (first covered in the applied graph-construction
  studybook; a deeper foundations notebook remains planned)
- [Graph topology](graph-topology.md) (degree, hubs, components, and ego graphs; introduced in the EDA
  studybook)
- The message-passing framework
- Spectral vs. spatial convolutions
- Over-smoothing and GNN depth limitations
- [Transductive vs. inductive learning](transductive-vs-inductive.md) (introduced in the temporal
  evaluation studybook)
- [Heterogeneous graphs](heterogeneous-graphs.md) (first covered in the applied graph-construction
  studybook)
- [Temporal / dynamic graph learning](temporal-graphs.md) (causal event ordering, snapshots, time-aware
  messages, memory, and the next applied direction)
- [Temporal role GNN](temporal-role-gnn.md) (the implemented four-channel, time-decayed causal model and
  its leakage contract)
- [Temporal graph Transformers](temporal-graph-transformers.md) (causal attention over bounded typed
  event histories, its promise, and its leakage/cold-start limits)

**Architectures** (mirrors `notebooks/01_architectures/`)
- GCN
- [GraphSAGE](graphsage.md) (mean aggregation, inductive representations, and this project's
  relation-aware extension)
- GAT
- GIN
- [Temporal role GNN](temporal-role-gnn.md) (the first applied causal temporal architecture)
- [Temporal graph Transformers](temporal-graph-transformers.md) (the next candidate after causal event
  sequence construction)

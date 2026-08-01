# GNN concepts

A growing reference of graph ML concepts, written as lookup entries (definition, intuition, when it
matters, pointer to the notebook that teaches it in depth). Starts empty and fills in alongside
`specs/roadmap.md` Phase 2 (foundations) and Phase 4 (architectures) — see
`specs/instructions/wiki-standards.md` for the convention to follow when adding a page.

Planned entries (create as each is actually covered — don't pre-write stubs with no content):

**Foundations** (mirrors `notebooks/00_foundations/`)
- [Graph representations](graph-representations.md) (first covered in the applied graph-construction
  studybook; a deeper foundations notebook remains planned)
- The message-passing framework
- Spectral vs. spatial convolutions
- Over-smoothing and GNN depth limitations
- [Transductive vs. inductive learning](transductive-vs-inductive.md) (introduced in the temporal
  evaluation studybook)
- [Heterogeneous graphs](heterogeneous-graphs.md) (first covered in the applied graph-construction
  studybook)
- Temporal / dynamic graph learning (relevant given the non-stationary transaction network)

**Architectures** (mirrors `notebooks/01_architectures/`)
- GCN
- GraphSAGE
- GAT
- GIN
- (heterogeneous and/or temporal GNN architecture(s), once Phase 4's applied-model decision is made)

# Backlog — graph-ml

Lightweight tracker for this project. This is a **starting point**, seeded from `specs/roadmap.md` and
what's known about the original `networkAnalysisForML` pipeline (see `CONSTITUTION.md` §0) — it gets built
out in conversation as work proceeds. Keep this file up to date: move items between sections as work
starts and finishes, and add new items as they come up rather than letting them live only in chat history.
For the full phased plan and rationale, see `specs/roadmap.md`.

## To Do

**Study & foundations** (`notebooks/00_foundations/`)
- [ ] Graph representation basics (adjacency matrix, edge list, `torch_geometric.data.Data`)
- [ ] Message-passing framework (aggregate-and-update, permutation invariance)
- [ ] Spectral vs. spatial convolutions
- [ ] Over-smoothing and GNN depth limitations
- [ ] Transductive vs. inductive learning on graphs
- [ ] Heterogeneous graphs (buyers/sellers/hybrids are structurally different node types)
- [ ] Temporal / dynamic graph learning (the transaction network is explicitly non-stationary)

**Baseline & data pipeline** (`src/graph_ml/`, `notebooks/02_project/`)
- [x] Dataset decision resolved: the owner has the real anonymized pipeline artifacts locally in `data/`
      (gitignored) — see `wiki/this-project/data-availability.md` for the full inventory (raw
      transactions through the final bond-graph feature set, plus a 3,217-column temporal snapshot file).
      No public/synthetic substitute needed.
- [x] Graph design decided: heterogeneous, instrument-centric, static-first — see
      `wiki/this-project/graph-design.md` for full schema and rationale.
- [x] Task framing decided: node classification on instrument nodes, **impairment target only for v1**
      (p90/p180 deferred until the single-target pipeline is validated end-to-end).
- [x] Node feature policy decided: instrument nodes get raw Tier-0 attributes; buyer/seller nodes get
      **pure learned embeddings**, no hand-computed intrinsic features (avoids smuggling leakage in).
- [ ] Implement graph construction (`src/graph_ml/data/`): build `torch_geometric.data.HeteroData` from
      `00_transactionsdf_simNames.pkl` / `01_instrumentsdf.pkl` per `wiki/this-project/graph-design.md`'s
      schema. Tests in `tests/`.
- [ ] Implement the **inductive** train/test split (train subgraph = instruments before cutoff T +
      buyer/seller nodes they touch; eval = post-T instrument nodes attached to the trained graph) — this
      is not optional, it's how the graph-design doc's leakage fix actually gets built. Pick/derive cutoff
      T (original used 30 Apr 2018 for impairment).
- [ ] Understand `04_network_snapshots.pkl`'s exact snapshot semantics (what time window each `sshot_N`
      covers) — deferred until temporal structure is added (see `specs/roadmap.md` Phase 4/beyond), not
      needed for the static v1 graph.
- [ ] Reproduce a classical baseline (logistic regression / random forest) for fair comparison — target
      numbers to compare against are in `wiki/original-project/results.md` (RF 0.954 AUC for impairment).
- [ ] Carry forward the time-leak-aware validation discipline from `wiki/original-project/modelling-and-validation.md`
      — addressed structurally via the inductive split above, not just a metrics-reporting concern.

**Architectures** (`notebooks/01_architectures/`, `src/graph_ml/models/`)
- [ ] GCN, GraphSAGE, GAT, GIN — foundational learning progression (see `specs/roadmap.md` Phase 4).
      GraphSAGE has a concrete extra motivation here beyond "next in the sequence": its native inductive
      setting matches the graph-design doc's leakage fix directly (`wiki/this-project/graph-design.md`).
- [ ] Explicit design decision (with rationale recorded in `wiki/gnn-concepts/`) on what architecture
      family actually gets applied to the project task, given the graph's heterogeneous + temporal nature
      — not just whichever foundational architecture was learned last. See
      `wiki/original-project/limitations-and-motivation-for-gnn.md`.
- [ ] Apply the chosen candidate(s) to the project task; report honest comparison vs. baseline.

**Engineering scaffolding**
- [ ] `specs/instructions/` workflows in place — first real use will validate whether they need revising.

## In Progress

_(nothing yet)_

## Done

- [x] Isolated project scaffold set up (git identity, uv environment, governing docs).
- [x] Reframed project as a dual-purpose GNN learning + engineering-portfolio showcase; `specs/` folder
      added (`mission.md`, `tech-stack.md`, `roadmap.md`, `instructions/`).
- [x] Public GitHub repo created and pushed (`davidemariani/GNN-trade-finance-credit-scoring`).
- [x] Deep-read the original thesis report and built `wiki/` (original-project knowledge base +
      `gnn-concepts/` placeholder), with an explicit analysis of what needs modernizing (heterogeneous +
      temporal graph learning) rather than reproducing the 2019 methodology as-is.
- [x] Confirmed real anonymized data is available locally (`data/`, gitignored) across every original
      pipeline stage; documented in `wiki/this-project/data-availability.md`.
- [x] Decided v1 graph design (heterogeneous, instrument-centric, static, inductive split), task framing
      (node classification, impairment-only first), and node feature policy (pure learned embeddings for
      buyer/seller nodes) — `wiki/this-project/graph-design.md`.

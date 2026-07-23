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
- [ ] Understand `04_network_snapshots.pkl`'s exact snapshot semantics (what time window each `sshot_N`
      covers) — the strongest candidate basis for temporal graph construction.
- [ ] Implement graph construction from transaction data (`00_transactionsdf_simNames.pkl` /
      `01_instrumentsdf.pkl`), with tests.
- [ ] Precisely define the prediction task (node / edge-link / graph-level) — three independent targets
      exist (impairment 2.06% positive, p90 7.01%, p180 6.02% — real, confirmed class imbalance, see
      `wiki/original-project/glossary.md` and `wiki/this-project/data-availability.md`); decide whether
      the rework keeps all three or focuses on one first.
- [ ] Reproduce a classical baseline (logistic regression / random forest) for fair comparison — target
      numbers to compare against are in `wiki/original-project/results.md` (RF 0.954 / 0.861, MLP 0.884).
- [ ] Carry forward the time-leak-aware validation discipline from `wiki/original-project/modelling-and-validation.md`
      — a GNN can leak future graph structure into a node embedding just as easily as a hand-engineered
      feature could.

**Architectures** (`notebooks/01_architectures/`, `src/graph_ml/models/`)
- [ ] GCN, GraphSAGE, GAT, GIN — foundational learning progression (see `specs/roadmap.md` Phase 4).
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

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

**Original project study**
- [ ] Write up an honest summary of the original pipeline (graph construction, bond-graph features,
      models used, reported results) to ground the eventual comparison.
- [ ] Decide the dataset strategy (original data is likely private/unavailable — pick a public dataset or
      a synthetic buyer/seller transaction graph generator, document the tradeoff).

**Baseline & data pipeline** (`src/graph_ml/`, `notebooks/02_project/`)
- [ ] Implement graph construction from transaction data, with tests.
- [ ] Precisely define the prediction task (node / edge-link / graph-level).
- [ ] Reproduce a classical baseline (logistic regression / random forest) for fair comparison.

**Architectures** (`notebooks/01_architectures/`, `src/graph_ml/models/`)
- [ ] GCN
- [ ] GraphSAGE
- [ ] GAT
- [ ] GIN
- [ ] Apply the strongest candidate(s) to the project task; report honest comparison vs. baseline.

**Engineering scaffolding**
- [ ] `specs/instructions/` workflows in place — first real use will validate whether they need revising.

## In Progress

_(nothing yet)_

## Done

- [x] Isolated project scaffold set up (git identity, uv environment, governing docs).
- [x] Reframed project as a dual-purpose GNN learning + engineering-portfolio showcase; `specs/` folder
      added (`mission.md`, `tech-stack.md`, `roadmap.md`, `instructions/`).

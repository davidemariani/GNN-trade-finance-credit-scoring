# Backlog — graph-ml

Live **work-item** tracker (to do / in progress / done). Decisions and their rationale do **not** live
here — those go in `wiki/` with a one-line pointer in `STUDYBOOK.md`'s decision log (see `CONSTITUTION.md`
§2.8). This file is just "what's the next thing to build." For the full phased plan, see `specs/roadmap.md`.

## To Do

**Study & foundations** (`notebooks/00_foundations/`)
- [ ] Graph representation basics (adjacency matrix, edge list, `torch_geometric.data.Data`)
- [ ] Message-passing framework (aggregate-and-update, permutation invariance)
- [ ] Spectral vs. spatial convolutions
- [ ] Over-smoothing and GNN depth limitations
- [ ] Transductive vs. inductive learning on graphs
- [ ] Heterogeneous graphs (multiple node/edge types — here: company vs. instrument, role-typed edges)
- [ ] Temporal / dynamic graph learning (the transaction network is explicitly non-stationary)

**Baseline & data pipeline** (`src/graph_ml/`, `notebooks/02_project/`)
_Design is decided (see `wiki/this-project/graph-design.md` + `evaluation.md`); these are the build tasks._
- [ ] **Convert data to Parquet (zstd)** + back up off-GitHub; keep `data/` gitignored (see
      `wiki/this-project/data-availability.md` "Storage format & policy").
- [ ] **Synthetic data generator** (`src/graph_ml/data/synthetic.py`): schema-faithful fake dataset
      (company + instrument, hybrids, imbalanced labels) so the pipeline runs without the private data and
      doubles as test fixtures + CI.
- [ ] Implement graph construction (`src/graph_ml/data/`): `HeteroData` with **company + instrument** node
      types, role-typed edges, company identity resolved by **name** (unifies the 15 hybrids), company
      features aggregated from pre-cutoff instruments only. Tests in `tests/`.
- [ ] Implement inductive temporal split + label-maturity filter + metrics (PR-AUC primary), per
      `wiki/this-project/evaluation.md`; report seen vs. cold-start breakdown.
- [ ] **Strong baseline**: LightGBM on instrument features + pre-T company aggregates (plus trivial +
      logistic-regression reference points) — the real bar the GNN must clear.
- [ ] `04_network_snapshots.pkl` snapshot semantics — deferred to the temporal phase, not needed for v1.

**Visualization** (`src/graph_ml/viz/`, cross-cutting — see `wiki/this-project/visualization.md`)
- [ ] Scaffold `src/graph_ml/viz/` + the seed-aware plotting conventions (`specs/instructions/viz-standards.md`).
- [ ] EDA visuals: class imbalance, temporal volume, degree distributions, missingness, hybrid footprint.
- [ ] Interactive company↔instrument topology view (pyvis) + static matplotlib/networkx snapshots.
- [ ] Results visuals come with the models: PR/ROC curves, calibration, embedding projections, GAT attention.
- [ ] Architecture visuals: Mermaid message-passing diagrams + torchview computational graphs (per notebook).
- [ ] Curated static gallery (PNG/SVG) embedded in `README.md`/`wiki/` so the repo is legible on GitHub.

**v1 vertical slice** (`specs/roadmap.md` Phase 3.5)
- [ ] End-to-end thin slice: data → LightGBM baseline → one GNN → honest PR-AUC comparison + written
      conclusion, all runnable from the synthetic generator. Do this before the full foundations sweep.

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
- [x] Decided v1 graph design, task framing (node classification, impairment-only first), and evaluation
      methodology — `wiki/this-project/graph-design.md` + `evaluation.md`.
- [x] Plan review (2026-07-24): corrected hybrid finding (15 hybrids by name; contagion is real); revised
      schema to company + instrument; replaced learned embeddings with time-windowed aggregated company
      features (resolves the embedding/inductive contradiction); adopted PR-AUC + label-maturity rule +
      LightGBM strong baseline; added synthetic generator + vertical-slice milestones.

# Roadmap

This file is the **plan and the status of record** — phases, goals, and checkboxes. `BACKLOG.md` holds
only the currently-active, finer-grained tasks and points here for the full picture.

## Plan at a glance

| Phase | What | Status |
|---|---|---|
| 0 | Environment & isolation setup (repo, uv env, docs, wiki) | ✅ done |
| 1 | Study the original 2019 project (deep-read → `wiki/original-project/`) | ✅ done |
| — | Design & methodology decisions (graph, evaluation, visualization, data storage) | ✅ done |
| 3 | Data pipeline + **strong (LightGBM) baseline** | ⬜ **next** |
| 3.5 | **v1 vertical slice** — data → baseline → one GNN → honest comparison (the priority) | ⬜ |
| 2 | GNN foundations notebooks (backfilled *around* the slice) | ⬜ ongoing |
| 4 | GNN architectures (GCN→GraphSAGE→GAT→GIN) + applied model choice | ⬜ |
| 5 | Portfolio quality gate | ⬜ |
| 6 | Interactive Hugo showcase dashboard (D3 for hero pieces) | ⬜ deferred |

**Execution order (not strictly by phase number).** Phases 0-1 and all the design decisions are done. The
next concrete work is Phase 3 (data pipeline + baseline) leading straight into the **Phase 3.5 vertical
slice**, which is the priority: one complete, honest, end-to-end story is worth more — for both learning
and showcase — than many half-finished notebooks. Phase 2 (foundations) and Phase 4 (architectures) are
then **backfilled around** that slice, not completed before it; a foundations notebook can be written when
its concept first becomes relevant to the applied work. Visualization is cross-cutting (see the note in
Phase 3). Phase 6 is explicitly deferred and blocks nothing.

Check items off as they're completed — the checkboxes below are the current status, not chat history.

---

## Phase 0 — Environment & Isolation Setup

**Goal:** Project lives in an isolated, public personal GitHub repo with its own Python environment.

- [x] Local git repo initialized, repo-local identity set (`davidemariani.ai@gmail.com`) ✓
- [x] `.gitignore` covering Python artifacts, the venv, and any data/model/secret files ✓
- [x] `uv`-managed environment (`pyproject.toml`, `uv.lock`, `.python-version`) with `torch`,
      `torch_geometric`, `networkx`, `scikit-learn`, `matplotlib`, `pytest`, `jupyter`, `ruff` ✓
- [x] Verified `torch` + `torch_geometric` import and MPS (Apple Silicon GPU) availability ✓
- [x] Skeleton layout: `src/graph_ml/`, `notebooks/{00_foundations,01_architectures,02_project}/`, `tests/`, `data/` (gitignored) ✓
- [x] Governing docs written: `CONSTITUTION.md`, `CLAUDE.md`, `BACKLOG.md`, `USAGE.md`, `README.md`, `specs/` ✓
- [x] Public GitHub repo created (`davidemariani/GNN-trade-finance-credit-scoring`), initial commit pushed ✓
- [x] `wiki/` knowledge base scaffolded (`wiki/README.md`, `wiki/original-project/`, `wiki/gnn-concepts/`, `wiki/this-project/`) ✓

---

## Phase 1 — Original Project Study ✓

**Goal:** Understand exactly what's being reworked before writing new code.

- [x] Deep-read the original thesis report (`wiki/original-project/source/Report.pdf`, local-only) and
      wrote up a full summary across `wiki/original-project/`: `overview.md`, `glossary.md`,
      `data-and-network-construction.md`, `feature-engineering.md`, `modelling-and-validation.md`,
      `results.md`, `limitations-and-motivation-for-gnn.md` ✓
- [x] Confirmed the original anonymized dataset **is** accessible — the owner has the full original
      pipeline artifacts (raw transactions through final bond-graph features, plus a temporal snapshot
      file) locally in `data/` (gitignored). See `wiki/this-project/data-availability.md`. ✓

---

## Phase 2 — GNN Foundations (`notebooks/00_foundations/`)

**Goal:** Build and demonstrate first-principles understanding. **Backfilled around the vertical slice**
(see Execution order above) — write each notebook when its concept first becomes relevant, not as a
gating sweep before any applied work.

- [ ] Graph representation basics: adjacency matrix vs. edge list vs. `torch_geometric.data.Data`;
      directed vs. undirected; node/edge/graph-level features.
- [ ] The message-passing framework: aggregate-and-update, permutation invariance, why it generalizes
      convolution to non-Euclidean structure.
- [ ] Spectral vs. spatial convolutions: graph Fourier basics, why most modern GNNs (GCN onward) use the
      spatial/message-passing view in practice.
- [ ] Over-smoothing and depth limitations in GNNs — why "just stack more layers" doesn't work the way it
      does in CNNs/Transformers.
- [ ] Transductive vs. inductive learning on graphs (node classification on a fixed graph vs. generalizing
      to unseen graphs/nodes) — relevant to which setting this project's problem falls into.

---

## Phase 3 — Baseline & Data (`notebooks/02_project/`, `src/graph_ml/`)

**Goal:** A working, honest, *strong* baseline before any GNN is judged against it. Design decided:
`wiki/this-project/graph-design.md` (company + instrument graph) and `wiki/this-project/evaluation.md`
(metrics, split, maturity, baselines).

- [x] Dataset decided: the real anonymized pipeline data in `data/` (see
      `wiki/this-project/data-availability.md`) — no public/synthetic substitute needed. ✓
- [x] Graph design, task framing, node-feature policy, and evaluation methodology decided (see the two
      `wiki/this-project/` docs above). ✓
- [x] **Convert working data to Parquet (zstd)** — done via `src/graph_ml/data/convert.py`, verified with
      an exact value-level round-trip check. Off-GitHub backup still open. See
      `wiki/this-project/data-availability.md` "Storage format & policy".
- [x] Implement graph construction (`src/graph_ml/data/`): build `HeteroData` with **company + instrument**
      node types and role-typed edges, resolving company identity by **name** (unifies the 15 hybrids),
      per `graph-design.md`. Company features aggregated from pre-cutoff instruments only, built directly
      from the real data. Tests use small, hand-built in-memory fixtures per `testing-standards.md` — a
      full synthetic dataset was tried and dropped (see `wiki/this-project/data-availability.md`: labels
      independent of features defeat the point of testing whether real structure predicts real outcomes).
      Implemented in `src/graph_ml/data/graph.py`, covered by `tests/data/test_graph.py`, and explained in
      `notebooks/02_project/00_graph_construction.ipynb`. ✓
- [x] Implement the inductive temporal split + target-aware label-maturity filter + metrics (PR-AUC
      primary) exactly as `evaluation.md` specifies; report seen vs. cold-start breakdown. Includes
      edge-filtered training/inference graph views so test instruments cannot update company states.
      Implemented in `src/graph_ml/evaluation/`, tested in `tests/evaluation/`, and explained in
      `notebooks/02_project/01_temporal_split_and_metrics.ipynb`. ✓
- [x] **Strong baseline**: LightGBM on instrument raw features + pre-T company aggregates (plus trivial +
      logistic-regression reference points) — this is the real bar the GNN must clear
      (`wiki/this-project/evaluation.md`). Implemented in `src/graph_ml/baselines/`, tested without access
      to test labels, explained in `notebooks/02_project/02_tabular_baselines.ipynb`, and logged in
      `results/baseline_metrics.csv`. Overall LightGBM PR-AUC: 0.465. ✓
- [x] **EDA + topology visualization** (`src/graph_ml/viz/`, per `wiki/this-project/visualization.md`):
      class imbalance, temporal volume, degree distributions, hybrid footprint, and an interactive
      company↔instrument network view. This is both understanding and showcase material. Implemented as
      tested aggregate/static/pyvis builders and explained with anonymous outputs in
      `notebooks/02_project/03_eda_and_topology.ipynb`. ✓
- [ ] `04_network_snapshots.pkl` snapshot semantics — deferred to the temporal phase, not needed for v1.

> **Visualization is cross-cutting, not a phase.** Per `wiki/this-project/visualization.md`, each phase
> produces its own visuals: EDA/topology here (Phase 3), architecture/message-passing diagrams in Phase 4,
> results/embedding/attention plots when models are evaluated. The dedicated *showcase* build is Phase 6.

---

## Phase 3.5 — v1 vertical slice (the minimum lovable version) — **PRIORITY**

**Goal:** One *complete, honest, end-to-end* story before breadth — the single most important near-term
milestone. A reviewer values this far more than many half-finished notebooks. Do this as a thin slice,
then backfill foundations (Phase 2) and architectures (Phase 4) around it.

- [ ] data → strong baseline → one GNN (GCN or GraphSAGE on the company+instrument graph) → honest
      comparison on PR-AUC with the maturity rule and cold-start breakdown → short written conclusion
      (including "the GNN did/didn't beat LightGBM, and here's the likely why"). Run against the real data
      locally (not reproducible from a bare public clone — see `wiki/this-project/data-availability.md`
      "Runnability trade-off"). Include the topology + results visuals so the slice is legible as a
      showcase — via committed notebook outputs, not by re-running — not just a metrics table.

---

## Phase 4 — GNN Architectures (`notebooks/01_architectures/`, `src/graph_ml/models/`)

**Goal:** Implement, understand, and apply a progression of architectures, each documented before/alongside
its use in the applied project. The foundational progression below is a *learning path* — the model(s)
actually applied to the project's task should be chosen deliberately for fit, not just picked because they
were learned first (see `wiki/original-project/limitations-and-motivation-for-gnn.md`: this graph is
heterogeneous — company vs. instrument node types with role-typed edges, and hybrid companies bridging
buyer/seller roles — and temporal/non-stationary, which a plain homogeneous, static-graph architecture
doesn't capture).

- [ ] GCN (Kipf & Welling) — first spatial convolution, simplest baseline GNN.
- [ ] GraphSAGE — inductive setting, neighbor sampling.
- [ ] GAT — attention-based neighbor weighting.
- [ ] GIN — expressiveness ceiling (Weisfeiler-Lehman test), why it matters.
- [ ] Each architecture notebook includes its **message-passing diagram (Mermaid)** and the **actual model
      computational graph (torchview)** — the visual half of the "explain before implement" rule
      (`wiki/this-project/visualization.md`).
- [ ] Explicit design decision: choose the architecture family for the applied model, informed by the
      foundational progression above but decided on fit to this graph's heterogeneous + temporal nature
      (e.g. relation-aware/heterogeneous message passing, and/or a temporal graph learning approach) —
      not defaulted to whichever foundational architecture came last. Document the decision and rationale
      in `wiki/gnn-concepts/` before implementing it.
- [ ] **Test the contagion hypothesis explicitly**: only once past the 2-hop same-company aggregation of
      v1 (which overlaps the original's hand-features) does the graph genuinely exercise buyer→hybrid→buyer
      contagion paths. Deeper/temporal models are where the "networked signal" claim actually gets tested —
      see `wiki/this-project/graph-design.md` "Known simplification".
- [ ] Apply the chosen candidate(s) to the project's prediction task; compare honestly against the
      Phase 3 strong (LightGBM) baseline on PR-AUC and record what did/didn't help and why.

---

## Phase 5 — Portfolio Quality Gate

**Goal:** Confirm the repo meets its dual bar (learning depth + engineering discipline) before treating it
as "done" for portfolio purposes.

- [ ] Every function/class in `src/graph_ml/` has a test; `pytest` passes cleanly.
- [ ] Every architecture in `src/graph_ml/models/` has a corresponding notebook explaining it.
- [ ] `README.md` gives a reviewer, in under two minutes, a clear picture of what was built, what was
      learned, and how to run it.
- [ ] No fabricated or unverified results anywhere — all reported numbers trace back to a runnable
      notebook/script.
- [ ] `ruff` clean; no committed data, weights, or secrets.
- [ ] Repo is visually legible on GitHub: a small curated gallery of static figures (topology, an
      architecture diagram, the results comparison) embedded in `README.md`/`wiki/`.

---

## Phase 6 — Showcase: interactive Hugo dashboard (deferred)

**Goal:** Redo the original 2019 dashboard as a modern, interactive showcase on a Hugo static site — the
public-facing portfolio artifact. Explicitly *later*; nothing here blocks Phases 2-5.

- [ ] Decide scope: which visuals become interactive web pieces (topology is the prime candidate).
- [ ] For each: use Plotly's HTML export where sufficient; reserve a hand-built **D3.js** component only
      for a hero piece that warrants it (adapt the working D3 force-graph from the owner's `dave_the_human`
      `/brain` site). See `wiki/this-project/visualization.md` "Deferred: Hugo dashboard".
- [ ] Build the Hugo site (mirroring the isolated, personal-GitHub pattern of `dave_the_human`), embedding
      the exported/handmade visuals + a written narrative of the project and its honest results.

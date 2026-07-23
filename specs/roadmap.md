# Roadmap

Phases are broadly sequential, but Phase 2 (foundations notebooks) and Phase 3 (baseline + data) can
overlap — there's no need to finish every foundational notebook before touching data. Check items off as
they're completed; this file (along with `BACKLOG.md`) is the source of truth for current status, not
chat history.

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

**Goal:** Build and demonstrate first-principles understanding before reaching for architectures.

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

**Goal:** A working, honest baseline before any GNN is judged against it.

- [x] Dataset decided: the real anonymized pipeline data in `data/` (see
      `wiki/this-project/data-availability.md`) — no public/synthetic substitute needed. ✓
- [ ] Understand `04_network_snapshots.pkl`'s exact snapshot semantics before relying on it for temporal
      graph construction.
- [ ] Implement graph construction (`src/graph_ml/data/`): nodes (buyers/sellers/hybrids), edges (trade
      relationships), raw attributes → node/edge features, built from `00_transactionsdf_simNames.pkl` /
      `01_instrumentsdf.pkl`. Tests in `tests/`.
- [ ] Precisely define the prediction task (node, edge/link, or graph-level) and justify the choice against
      what the original project did. Account for real, confirmed class imbalance (impairment 2.06%, p90
      7.01%, p180 6.02% positive).
- [ ] Reproduce a classical baseline (logistic regression / random forest on simple graph features) —
      this is the number every GNN result gets compared against (target: `wiki/original-project/results.md`).

---

## Phase 4 — GNN Architectures (`notebooks/01_architectures/`, `src/graph_ml/models/`)

**Goal:** Implement, understand, and apply a progression of architectures, each documented before/alongside
its use in the applied project. The foundational progression below is a *learning path* — the model(s)
actually applied to the project's task should be chosen deliberately for fit, not just picked because they
were learned first (see `wiki/original-project/limitations-and-motivation-for-gnn.md`: this graph is
heterogeneous — buyers/sellers/hybrids are structurally different node types — and temporal/non-stationary,
which a plain homogeneous, static-graph architecture doesn't capture).

- [ ] GCN (Kipf & Welling) — first spatial convolution, simplest baseline GNN.
- [ ] GraphSAGE — inductive setting, neighbor sampling.
- [ ] GAT — attention-based neighbor weighting.
- [ ] GIN — expressiveness ceiling (Weisfeiler-Lehman test), why it matters.
- [ ] Explicit design decision: choose the architecture family for the applied model, informed by the
      foundational progression above but decided on fit to this graph's heterogeneous + temporal nature
      (e.g. relation-aware/heterogeneous message passing, and/or a temporal graph learning approach) —
      not defaulted to whichever foundational architecture came last. Document the decision and rationale
      in `wiki/gnn-concepts/` before implementing it.
- [ ] Apply the chosen candidate(s) to the project's prediction task; compare honestly against the
      Phase 3 baseline and record what did/didn't help and why.

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

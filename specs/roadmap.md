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
- [ ] Public GitHub repo created (`davidemariani/GNN-trade-finance-credit-scoring`), initial commit pushed

---

## Phase 1 — Original Project Study

**Goal:** Understand exactly what's being reworked before writing new code.

- [ ] Write up a short, honest summary (in a notebook or `specs/` note) of the original
      `networkAnalysisForML` pipeline: how the transaction graph was built, what bond-graph features
      were engineered, what models were used, and what results were reported — this grounds the "before"
      side of the eventual comparison.
- [ ] Identify what data was used originally and whether any equivalent is available (real data was
      likely private/thesis-specific) — decide the dataset strategy for this rework (see Phase 3).

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

- [ ] Decide the dataset: a public trade-finance/transaction-network dataset, or a synthetic buyer/seller
      transaction graph generator with realistic structure — document the choice and its limitations.
- [ ] Implement graph construction (`src/graph_ml/data/`): nodes (buyers/sellers), edges (transactions),
      raw attributes → node/edge features. Tests in `tests/`.
- [ ] Precisely define the prediction task (node, edge/link, or graph-level) and justify the choice against
      what the original project did.
- [ ] Reproduce a classical baseline (logistic regression / random forest on simple graph features) —
      this is the number every GNN result gets compared against.

---

## Phase 4 — GNN Architectures (`notebooks/01_architectures/`, `src/graph_ml/models/`)

**Goal:** Implement, understand, and apply a progression of architectures, each documented before/alongside
its use in the applied project.

- [ ] GCN (Kipf & Welling) — first spatial convolution, simplest baseline GNN.
- [ ] GraphSAGE — inductive setting, neighbor sampling.
- [ ] GAT — attention-based neighbor weighting.
- [ ] GIN — expressiveness ceiling (Weisfeiler-Lehman test), why it matters.
- [ ] Apply the strongest candidate(s) to the project's prediction task; compare honestly against the
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

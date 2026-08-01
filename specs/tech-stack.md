# Tech Stack

## Core

| Layer | Technology | Notes |
|-------|-----------|-------|
| Language | **Python 3.12** | Pinned via `.python-version`; chosen for broad, stable compatibility with the current PyTorch/PyG release rather than bleeding-edge Python. |
| ML framework | **PyTorch** (`torch`) | Apple M4 → MPS (Metal) backend for GPU acceleration, no CUDA needed on this machine. |
| GNN library | **PyTorch Geometric** (`torch_geometric`) | Chosen over DGL for its tighter PyTorch integration and the breadth of example architectures/datasets useful for a learning-focused project. |
| Classical ML (baselines) | `scikit-learn` | Logistic-regression / trivial baselines and metrics (PR-AUC, ROC AUC). |
| Strong tabular baseline | `lightgbm` + system `libomp` | Gradient-boosted trees — the *strong* modern baseline a GNN must actually beat to justify itself. On macOS the wheel requires `brew install libomp`. See `wiki/this-project/evaluation.md`. |
| Graph handling / prototyping | `networkx` | Used for exploratory graph construction and visualization before/alongside `torch_geometric.data.Data` objects. |
| Data storage | `pyarrow` (Parquet) | Working data stored as Parquet (zstd) — columnar, compressed (~7-14% of pickle size), safe (no arbitrary-code-execution on load), fast. Replaces the 2019 `.pkl` files. See `wiki/this-project/data-availability.md`. |
| Notebooks | `jupyter` / `jupyterlab` | All educational and experiment notebooks. |
| Static plotting | `matplotlib` | Default for GitHub-rendered, reproducible, publication-quality figures. |
| Interactive plotting | `plotly` | Interactive EDA/results; exports self-contained HTML that embeds into the future Hugo site. |
| Interactive network topology | `pyvis` (vis.js) | Force-directed, draggable company↔instrument network views. |
| Architecture diagrams | Mermaid (markdown) + `torchview` (dev) | Mermaid for conceptual/message-passing diagrams (renders on GitHub); torchview for the actual model computational graph. **torchview needs the system `graphviz` binary** (`brew install graphviz`) to render. |
| Testing | `pytest` | All non-exploratory code in `src/graph_ml/` gets tests in `tests/`. |
| Linting | `ruff` | Fast, single-tool lint + format. |

Full visualization approach and the deferred D3.js/Hugo showcase decision: `wiki/this-project/visualization.md`.

## Local Environment (Isolated)

All dependencies are managed by **uv** — scoped to this project folder only via `.venv/`. Nothing is
installed globally, into system Python, or shared with any other project.

| Item | Value |
|------|-------|
| Version manager | **uv** (already installed via Homebrew on this machine) |
| Config files | `pyproject.toml` (dependencies), `uv.lock` (resolved/pinned versions), `.python-version` (interpreter) |
| Activation | `source .venv/bin/activate` from the project root |

**To replicate on any machine:**
```bash
brew install uv graphviz libomp   # graphviz: torchview; libomp: LightGBM on macOS
git clone https://github.com/davidemariani/GNN-trade-finance-credit-scoring.git ~/graph_ml
cd ~/graph_ml && uv sync
source .venv/bin/activate
```

## Isolation & Git Identity

This is a **personal learning/portfolio project**, fully separated from work infrastructure.

| Item | Value |
|------|-------|
| Project location | `~/graph_ml/` (`/Users/dmariani/graph_ml/`) |
| Sync mechanism | **Public GitHub repo** — single source of truth for code, notebooks, and docs |
| Personal VCS | GitHub (`github.com/davidemariani/GNN-trade-finance-credit-scoring`) |
| Work VCS | GitLab (separate host — no conflict, never referenced here) |
| Git identity scope | **Repo-local only** — set via `git config --local` (`davidemariani.ai@gmail.com`), overriding the machine's global (work) identity |
| CI/secrets | None required so far — no deployment pipeline, this is a code+notebook repo |

> On any machine, always set `git config --local user.email` before the first commit so the global
> (work) identity is never used for this repo.

## Constraints

- Never commit real/private data, trained model weights, or credentials — see `.gitignore` and
  `CONSTITUTION.md` §2.
- Never push this project's code to a work GitLab remote; the only remote is personal GitHub.
- Every non-exploratory function/class in `src/graph_ml/` needs a test in `tests/`.
- Every GNN architecture used in `src/graph_ml/` needs a corresponding explanatory notebook in
  `notebooks/01_architectures/` before (or alongside) being adopted in the applied project code.

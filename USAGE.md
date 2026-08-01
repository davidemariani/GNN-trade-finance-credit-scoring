# Usage — graph-ml

Practical how-to for this repo. Extend this as real code/scripts get added.

## Environment setup

Requires [`uv`](https://docs.astral.sh/uv/) (already installed on this machine) — no conda, no system
Python involved.

```bash
cd /path/to/GNN-trade-finance-credit-scoring
uv sync              # creates/updates .venv and installs all deps from pyproject.toml + uv.lock
source .venv/bin/activate
```

System dependencies on macOS: `brew install libomp` is required by LightGBM; `brew install graphviz` is
optional and needed only to *render* `torchview` architecture diagrams (Mermaid needs nothing).

Adding a new dependency (keeps `pyproject.toml` and `uv.lock` in sync — don't hand-edit or `pip install`
directly into the venv):

```bash
uv add <package>          # runtime dependency
uv add --group dev <package>   # dev-only dependency
```

## Sanity check

```bash
python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__, torch.backends.mps.is_available())"
```

Should print the installed versions and `True` for MPS availability (Apple Silicon GPU acceleration).

## Running tests

```bash
pytest
```

## Notebooks

```bash
jupyter lab notebooks/
```

- `notebooks/00_foundations/` — GNN fundamentals, dataset-independent.
- `notebooks/01_architectures/` — one notebook per architecture (GCN, GraphSAGE, GAT, GIN, ...).
- `notebooks/02_project/` — applied work on the trade-finance transaction graph problem.

See `specs/instructions/notebook-standards.md` before adding a new notebook.

## Git / GitHub

This repo is isolated from the rest of the machine — local git identity is the personal
`davidemariani.ai@gmail.com`, and the only remote is
`https://github.com/davidemariani/GNN-trade-finance-credit-scoring` on the personal `davidemariani`
GitHub account (never the work GitLab). See `CONSTITUTION.md` §1 before changing any of this.

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

The executed applied notebooks are ordered `00` through `06`; notebook `05` begins the point-in-time
remediation with label clocks, causal histories, and the bond audit, while notebook `06` derives and
evaluates the temporal role GNN. The static GraphSAGE architecture derivation lives
in `notebooks/01_architectures/graphsage.ipynb`; the real-data comparison is
`notebooks/02_project/04_hetero_graphsage.ipynb`. Both require Graphviz only when re-rendering the traced
computational graph; their committed outputs remain visible without rerunning.

The current `data/` workspace has the eight Parquet pipeline stages, but not the historical
`04_network_snapshots.pkl` or original pickle files. The `00`–`04` results use a final-snapshot cohort and
should be read as retrospective benchmarks. The causal p90 temporal model can be rerun because its
due-date-plus-horizon label clock is explicit; impairment temporal runs remain blocked until event-time
semantics are verified. See `wiki/this-project/evaluation.md` and
`wiki/this-project/data-availability.md`.

The point-in-time utilities currently support verified event/resolution dates and conservative
due-date-plus-horizon targets. The first complete causal rerun uses p90; p180 has no mature positives in
the current test window, and impairment remains blocked on event-time provenance.

The causal p90 baseline is implemented in `src/graph_ml/baselines/point_in_time.py` and executed in
notebook 05. Its committed-style run artifact is `results/point_in_time_p90_metrics.csv`; rerunning the
notebook rebuilds the strictly-prior histories and model from the local Parquet data.

The first causal temporal GNN is implemented across `src/graph_ml/data/temporal_graph.py`,
`src/graph_ml/models/temporal_role_gnn.py`, and `src/graph_ml/training/temporal_gnn.py`, and executed in
notebook 06. Its five-seed artifact is `results/temporal_gnn_p90_metrics.csv`. A seed-42 notebook rerun is
deterministic on CPU; the other frozen seeds remain in the result artifact so the notebook teaches
variability without retraining five models every time. The five-seed root-only diagnostic is retained in
`results/root_only_p90_metrics.csv` and visualized in the same notebook.

## Git / GitHub

This repo is isolated from the rest of the machine — local git identity is the personal
`davidemariani.ai@gmail.com`, and the only remote is
`https://github.com/davidemariani/GNN-trade-finance-credit-scoring` on the personal `davidemariani`
GitHub account (never the work GitLab). See `CONSTITUTION.md` §1 before changing any of this.

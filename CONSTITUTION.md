# Constitution — graph-ml

This file governs how this folder works. It applies to any agent (Claude, another AI assistant, or the
human owner) operating in this workspace. If you are an agent picking this up cold: **read this file in
full before creating, editing, or moving anything here.**

Owner: Davide Mariani. Purpose: rework the original
[`networkAnalysisForML`](https://github.com/davidemariani/networkAnalysisForML) project — a buyer/seller
trade-finance transaction network modeled with hand-engineered `networkx`/bond-graph features and
classical ML (linear model, random forest, MLP, RNN) — into a graph neural network approach built on
PyTorch and PyTorch Geometric.

This is **not just an applied project — it is a learning and portfolio showcase**. It exists to build and
demonstrate real depth in graph machine learning (from first principles through specific architectures)
*and* real software engineering discipline (specs-driven development, tested code, clear documentation).
It's tracked as one of the works in the job-application portfolio maintained in `~/Desktop/studybook`, so
it needs to be good enough to show, not just good enough to work. Full detail on the mission is in
`specs/mission.md` — read that alongside this file.

---

## 0. Where this came from

`networkAnalysisForML` was a Msc Data Science thesis project (Davide Mariani, Birkbeck College,
University of London, in collaboration with Tradeteq Ltd; supervisors Prof. George D. Magoulas and
Michael Boguslavsky). It:

- built a graph of buyer/seller trade-finance transactions using `networkx`
  (`scripts_network/network_modelling.py`, `scripts_network/network_bond_graph.py` — the latter borrows
  effort/flow/energy concepts from bond-graph theory to hand-engineer node/edge features),
- ran feature engineering pipelines over that graph plus the raw instrument-level transaction data
  (`scripts_preproc/`),
- fed the resulting tabular features into classical models — `SGDClassifier`, `RandomForestClassifier`
  (`scripts_ml/models_utils.py`), plus MLP and RNN notebooks — to predict credit/default-related outcomes
  at the transaction level,
- tracked experiments with `mlflow` and exposed results via a `dash`-based dashboard.

This project does **not** vendor that code. The original repo remains public at the URL above as the
historical reference; consult it directly (or ask the owner for context) rather than assuming behavior.
The idea being carried forward is the *problem* (predict trade-finance/credit outcomes from a network of
buyer/seller transactions) and the *data shape* (a transaction graph), not the original implementation.
The bet of this rework is that a GNN can learn useful structural representations directly from the graph,
reducing or replacing the hand-engineered bond-graph feature step.

---

## 1. Isolation rules (do not violate)

This folder must stay fully isolated from the rest of the machine, the same way
`~/Desktop/studybook` and `~/dave_the_human` are:

1. **Git identity**: this repo's local git config (`user.name` / `user.email`) is set to the owner's
   personal identity (`davidemariani.ai@gmail.com`), overriding the machine's global git config (which
   is set to a work email for a separate GitLab remote). Never change this repo's local git config to
   the work identity, and never add a remote pointing at a work GitLab instance.
2. **GitHub only, personal account**: the only remote for this repo is
   `github.com/davidemariani/GNN-trade-finance-credit-scoring` (public), authenticated via the
   `davidemariani` `gh` account. Never push this project's code anywhere else.
3. **Isolated Python environment**: dependencies live only in this project's `uv`-managed `.venv`
   (`pyproject.toml` + `uv.lock`). Never install project dependencies into system Python, `pyenv`
   global environments, or any other project's environment. Use `uv add <package>` to add a dependency
   so `pyproject.toml`/`uv.lock` stay authoritative.
4. **No credential or data bleed**: don't reference, copy, or depend on files, credentials, or datasets
   from other projects on this machine (in particular, nothing from work-related repos or environments).

---

## 2. Core principles

1. **Honest benchmarking.** Any comparison between the new GNN approach and the original classical
   models must be run, not assumed or invented. If a fair comparison isn't possible (e.g. the original
   dataset isn't available), say so explicitly rather than fabricating numbers.
2. **No data or secrets in git.** Raw transaction data, trained model weights, and any credentials never
   get committed — `.gitignore` already excludes `data/`, `*.pt`/`*.pth`/`*.ckpt`, `mlruns/`, `wandb/`,
   `.env*`. If real trade-finance data is ever used, treat it as confidential and keep it entirely local.
3. **Reproducibility.** New experiments should be runnable end-to-end from a fresh `uv sync` plus a
   documented command (see `USAGE.md`). Record what was run and with what config well enough that it
   can be repeated.
4. **Don't restructure without saying so.** If the folder layout needs to change, say what and why
   before doing it — this file is easy to update, but silent structural drift makes the workspace harder
   to navigate for the next session.
5. **Specs-driven development.** Non-trivial work is planned in `specs/` (`mission.md`, `tech-stack.md`,
   `roadmap.md`) before it's built, and `specs/instructions/` defines the recurring workflows (adding an
   architecture, writing a notebook, writing a test). Update these as decisions are made — they are the
   record of *why*, which chat history and code alone don't preserve.
6. **Explain, don't just implement.** Every GNN concept or architecture used in `src/graph_ml/` needs a
   corresponding notebook that explains it in prose and math before/alongside the code — see
   `specs/instructions/new-architecture.md` and `specs/instructions/notebook-standards.md`. Code without
   an explanation defeats the point of this project.
7. **Test what you build.** Everything in `src/graph_ml/` gets a test in `tests/` — see
   `specs/instructions/testing-standards.md`. This is a portfolio piece; untested code undercuts the
   engineering-discipline half of the point.

---

## 3. Folder structure

```
graph_ml/
├── CONSTITUTION.md          ← this file — read first
├── CLAUDE.md                ← pointer file for Claude Code, read second
├── BACKLOG.md                ← current work items (To Do / In Progress / Done)
├── USAGE.md                  ← environment setup and common commands
├── README.md                 ← public-facing, portfolio-facing project description
├── specs/                    ← specs-driven development: the "why" behind the work
│   ├── mission.md               (what this project is for and who it's for)
│   ├── tech-stack.md             (technology choices and isolation setup)
│   ├── roadmap.md                 (phased plan, checked off as completed)
│   └── instructions/               (recurring workflows: new architecture, notebook standards, testing)
├── pyproject.toml            ← dependencies (managed via uv)
├── uv.lock
├── .python-version
├── src/graph_ml/              ← installable package: graph construction, models, training, evaluation code
├── notebooks/
│   ├── 00_foundations/          (GNN fundamentals, dataset-independent)
│   ├── 01_architectures/         (one notebook per architecture: GCN, GraphSAGE, GAT, GIN, ...)
│   └── 02_project/                (applied work on the trade-finance transaction graph problem)
├── tests/                     ← pytest tests, mirroring src/graph_ml structure
└── data/                      ← gitignored, local-only; raw/processed data lives here, never committed
```

Application code goes in `src/graph_ml/` as it stabilizes; exploratory work starts in `notebooks/`. Add
tests under `tests/` for anything in `src/graph_ml/` that isn't purely exploratory — see
`specs/instructions/testing-standards.md`.

---

## 4. Where to look next

- `specs/mission.md`, `specs/tech-stack.md`, `specs/roadmap.md` for the full plan and its rationale.
- `specs/instructions/` for how to add an architecture, write a notebook, or write a test.
- `BACKLOG.md` for what's currently being worked on and what's next.
- `USAGE.md` for how to set up the environment and run tests/notebooks/scripts.

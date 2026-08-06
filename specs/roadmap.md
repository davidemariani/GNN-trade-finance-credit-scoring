# Roadmap

This file is the **plan and the status of record** — phases, goals, and checkboxes. `BACKLOG.md` holds
only the currently-active, finer-grained tasks and points here for the full picture.

## Plan at a glance

| Phase | What | Status |
|---|---|---|
| 0 | Environment & isolation setup (repo, uv env, docs, wiki) | ✅ done |
| 1 | Study the original 2019 project (deep-read → `wiki/original-project/`) | ✅ done |
| — | Design & methodology decisions (graph, evaluation, visualization, data storage) | ✅ done |
| 3 | Data pipeline + **strong (LightGBM) baseline** | ✅ done |
| 3.5 | **v1 vertical slice** — data → baseline → one GNN → honest comparison (the priority) | ✅ done |
| 3.6 | **Point-in-time remediation** — causal labels/features → tabular baseline → temporal GNN | 🟨 ongoing |
| 2 | GNN foundations notebooks (backfilled *around* the slice) | ⬜ ongoing |
| 4 | GNN architectures (GCN→GraphSAGE→GAT→GIN) + applied model choice | ⬜ |
| 5 | Portfolio quality gate | ⬜ |
| 6 | Interactive Hugo showcase dashboard (D3 for hero pieces) | ⬜ deferred |

**Execution order (not strictly by phase number).** Phases 0-1, the design decisions, Phase 3, and the
Phase 3.5 vertical slice are done. The slice establishes the first honest result: relation-aware
GraphSAGE reaches 0.305 PR-AUC at seed 42 and 0.244 ± 0.079 across five seeds, but does not beat
LightGBM's 0.465. A subsequent audit classifies both as retrospective final-snapshot benchmarks because
label availability and within-training history are not fully as-of-time. Phase 3.6 therefore comes next;
Phase 2 foundations continue alongside it, while broader architecture comparisons wait for the corrected
yardstick. Visualization remains cross-cutting. Phase 6 is explicitly deferred and blocks nothing.

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
- [x] Confirmed the original anonymized tabular pipeline **is** accessible. The current `data/` inventory
      contains eight Parquet artifacts from raw transactions through final bond-graph features. The
      historical temporal snapshot is no longer present and recovery is a Phase 3.6 task. See
      `wiki/this-project/data-availability.md`. ✓

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
- [ ] Recover `04_network_snapshots.pkl` if possible and audit its timestamp/availability semantics; it is
      absent from the current workspace and was not needed for v1.

> **Visualization is cross-cutting, not a phase.** Per `wiki/this-project/visualization.md`, each phase
> produces its own visuals: EDA/topology here (Phase 3), architecture/message-passing diagrams in Phase 4,
> results/embedding/attention plots when models are evaluated. The dedicated *showcase* build is Phase 6.

---

## Phase 3.5 — v1 vertical slice (the minimum lovable version) — **PRIORITY**

**Goal:** One *complete, honest, end-to-end* story before breadth — the single most important near-term
milestone. A reviewer values this far more than many half-finished notebooks. Do this as a thin slice,
then backfill foundations (Phase 2) and architectures (Phase 4) around it.

- [x] data → strong baseline → one GNN (relation-aware GraphSAGE on the company+instrument graph) → honest
      comparison on PR-AUC with the maturity rule and cold-start breakdown → short written conclusion
      (including "the GNN did/didn't beat LightGBM, and here's the likely why"). Run against the real data
      locally (not reproducible from a bare public clone — see `wiki/this-project/data-availability.md`
      "Runnability trade-off"). Include the topology + results visuals so the slice is legible as a
      showcase — via committed notebook outputs, not by re-running — not just a metrics table.
      GraphSAGE PR-AUC: 0.305 overall / 0.291 seen / 0.319 cold-start, below LightGBM's
      0.465 / 0.432 / 0.387. Conclusion and visuals: `notebooks/02_project/04_hetero_graphsage.ipynb`;
      run log: `results/gnn_metrics.csv`. A frozen five-seed robustness pass finds overall
      0.244 ± 0.079 (range 0.115–0.305), so seed 42 is explicitly not presented as typical. ✓

---

## Phase 3.6 — Point-in-time remediation and temporal modelling — **NEXT**

**Goal:** turn the retrospective vertical slice into a causal, deployment-like comparison before spending
the test budget on more architecture tuning.

- [ ] Recover/audit event-time sources and define prediction time, target horizon, event timestamp, closure,
      censoring, and simultaneous-event semantics. Do not treat `debt_collection_date` as impairment time
      without evidence.
- [x] Implement strictly-as-of tabular company histories and label-availability masks. Every row sees events
      `< t_i`; cumulative aggregates are shifted; preprocessing is fitted inside each training fold.
      Generic strictly-prior history construction plus a lifecycle/outcome/bond schema guard are now
      implemented and tested in `src/graph_ml/data/temporal.py`. Explicit event/horizon label availability
      and rolling masks are implemented in `src/graph_ml/evaluation/point_in_time.py`; temporal-graph
      integration remains. The tabular contract, including fold-fitted preprocessing, is complete. ✓
- [x] Add adversarial leakage tests: modifying future labels/features/topology must not change earlier
      features, labels, selected hyperparameters, embeddings, or scores. The current point-in-time feature,
      context, and model tests cover strict ordering, simultaneous events, future-row isolation, and target
      exclusion; extend them with each new temporal component. ✓
- [x] Use rolling-origin validation/test windows and report prevalence, seen/cold-start status, and multiple
      time folds. If impairment timing cannot be established, use an explicit p90/p180 horizon or survival
      formulation rather than inventing event times. Implemented for p90 with two pre-holdout development
      folds plus the reported final origin. ✓
- [x] Re-run LightGBM first on the point-in-time feature stream: p90 PR-AUC is 0.079 all / 0.102 seen /
      0.026 cold-start with 58 trees selected on rolling validation. ✓
- [x] Compare a role-aware temporal
      GraphSAGE using event age/recency and causal company-state updates. Report multiple neural seeds.
      Use p90 for the first complete pass: impairment timing is unresolved and the mature p180 test cohort
      has zero positives at the current dates. The first four-channel temporal role model averages PR-AUC
      0.053 ± 0.033 overall versus LightGBM's 0.079 across five frozen seeds. ✓
- [x] Create a visual studybook showing an event timeline, legal/illegal messages, rolling folds, causal
      context updates, and the temporal-vs-tabular comparison. Notebook 06 explains the model from first
      principles and visualizes seed instability. ✓
- [ ] Run temporal component ablations chosen on validation folds: root-only control, relation collapse,
      no-decay/predeclared half-lives, and hub-aware recent-neighbour aggregation. Root-only is complete:
      relation contexts improve overall/seen means but hurt cold-start; remaining ablations stay open.
- [x] Add pre-holdout expanding-window backtests with minimum class support. Two twelve-month folds now
      show time-varying prevalence/model ordering and provide the selection environment for architecture
      changes without reusing the reported final holdout. ✓
- [x] Build bounded strictly-prior role-event tensors: newest-first values, positive ages, validity masks,
      and auditable source indices with zero padding. Future/simultaneous isolation is tested and notebook
      07 inspects the real K=8 artifact. ✓
- [x] Evaluate a small causal temporal graph Transformer against fixed-decay GNN, root-only, and LightGBM
      on the development folds. The model, masks/fallback, rolling refit, five-seed results, and learned
      attention diagnostic are tested and explained in notebook 08. It helps sparse fold-1 seen rows but
      trails both leading models in fold 2, so it is not promoted. ✓
- [x] Run validation-only Transformer ablations: explicit decay versus learned time, smaller capacity and
      stronger regularization, K, and root/message fusion. The paired time ablation is complete and finds
      no stable gain: fixed/learned/no-age validation means are nearly tied. Coverage gating is also
      complete; it improves fold 1 from 0.302 to 0.427 but harms fold 2 from 0.020 to 0.010. The 2×2
      capacity/regularization control is also negative: compact variants are weaker and strong
      regularization is not seed-stable. K=2 helps only fold 1, K=8 leads fold 2, and K=16 doubles memory
      without improving either mean. The original K=8 model remains frozen; the sealed holdout stayed
      untouched. → notebook 09. ✓

Detailed audit: `wiki/this-project/evaluation.md`; concept guide:
`wiki/gnn-concepts/temporal-graphs.md`, `wiki/gnn-concepts/temporal-role-gnn.md`, and
`wiki/gnn-concepts/temporal-graph-transformers.md`; bond-feature audit:
`wiki/this-project/bond-graph-leakage-audit.md`.

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
- [x] GraphSAGE — inductive mean aggregation; relation-aware full-batch extension chosen for this
      graph's size and cold-start setting. `notebooks/01_architectures/graphsage.ipynb`. ✓
- [ ] GAT — attention-based neighbor weighting.
- [ ] GIN — expressiveness ceiling (Weisfeiler-Lehman test), why it matters.
- [ ] Each architecture notebook includes its **message-passing diagram (Mermaid)** and the **actual model
      computational graph (torchview)** — the visual half of the "explain before implement" rule
      (`wiki/this-project/visualization.md`).
- [x] Explicit design decision: choose the architecture family for the applied model, informed by the
      foundational progression above but decided on fit to this graph's heterogeneous + temporal nature
      (e.g. relation-aware/heterogeneous message passing, and/or a temporal graph learning approach) —
      not defaulted to whichever foundational architecture came last. Document the decision and rationale
      in `wiki/gnn-concepts/` before implementing it. Relation-aware GraphSAGE selected for the v1
      inductive slice; see `wiki/gnn-concepts/graphsage.md`. ✓
- [ ] **Test the contagion hypothesis explicitly**: only once past the 2-hop same-company aggregation of
      v1 (which overlaps the original's hand-features) does the graph genuinely exercise buyer→hybrid→buyer
      contagion paths. Deeper/temporal models are where the "networked signal" claim actually gets tested —
      see `wiki/this-project/graph-design.md` "Known simplification".
- [x] Apply the first chosen candidate to the project's prediction task; compare honestly against the
      Phase 3 strong (LightGBM) baseline on PR-AUC and record what did/didn't help and why.
      GraphSAGE does not win; the likely limitations are documented without test-driven retuning. ✓

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

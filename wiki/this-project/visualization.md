# This project: visualization (decided 2026-07-24)

Visualization is a first-class part of this project, not decoration — it serves both goals: building
genuine intuition (graph topology, message passing, what the model learns) and the portfolio showcase.
This page owns the visualization approach; `specs/instructions/viz-standards.md` owns the how-to
conventions.

## Two layers, clear boundary

- **Python = the working + authoring layer** (used *while doing the analysis*, regenerable, tested, tied
  to the pipeline). This is where essentially all visualization happens.
- **JavaScript / D3.js = a deferred, showcase-only presentation layer** for the future Hugo static site —
  reserved for a *small number* of hero visualizations that a generic export can't do justice to. Not part
  of the core stack, not written now. See "Deferred: Hugo dashboard" below.

The key point: we do **not** hand-write JavaScript for routine work. Plotly and pyvis already emit
self-contained interactive HTML/JS (Plotly *is* `plotly.js` with a Python wrapper; pyvis wraps `vis.js`),
so interactivity comes for free from Python — and Plotly's HTML export embeds into Hugo with near-zero
effort, covering most of the "interactive on the website" need without touching D3.

## Tooling

| Purpose | Tool | Why |
|---|---|---|
| Static, publication-quality figures | **matplotlib** | Renders inline on GitHub (portfolio-visible without running anything), fully reproducible, zero surprises. Default for EDA/results figures that go in the README/wiki. |
| Interactive EDA & results | **Plotly** | Interactive exploration; exports self-contained HTML that drops straight into the future Hugo site. The interactive default. |
| Interactive network topology | **pyvis** (vis.js) | Force-directed, draggable company↔instrument network views with little code — the graph is a showcase highlight. |
| Architecture / message-passing diagrams | **Mermaid** | Renders natively in GitHub markdown, version-controlled, regenerable — no binary needed to view. For conceptual diagrams in notebooks/wiki. |
| Actual model computational graph | **torchview** (dev) | Auto-renders a real model's layer/tensor-flow graph. Needs the system `graphviz` binary (`brew install graphviz`) to render to image — the Python package alone isn't enough. |
| Embedding projections (results phase) | **scikit-learn t-SNE** (already a dep); UMAP only if needed | Visualize learned node embeddings. t-SNE needs no new dep; defer `umap-learn` (pulls `numba`) until it's actually wanted. |

## The four kinds of visualization (each has a home)

1. **Data / EDA** — class imbalance, temporal transaction volume, degree distributions, missingness, hybrid
   footprint. Reusable functions in `src/graph_ml/viz/`, called from `notebooks/02_project/`.
2. **Graph topology** — the company↔instrument network, hybrids as bridge nodes, connected components,
   ego-graphs around a company/instrument. The most visually striking piece and the clearest "I understand
   graphs" signal. pyvis for interactive, matplotlib/networkx for static snapshots.
3. **Architecture** — message-passing intuition and layer schematics (Mermaid) + the actual computational
   graph (torchview), one per architecture notebook. Ties directly to the "explain before implement" rule
   (`specs/instructions/new-architecture.md`).
4. **Results** — PR/ROC curves, calibration, confusion at the chosen operating point, learned-embedding
   projections, and (for GAT) attention weights drawn on the graph. Compared honestly against the baseline
   per `wiki/this-project/evaluation.md`.

## Where visualization lives

- `src/graph_ml/viz/` — reusable, importable plotting/graph-drawing functions (not copy-pasted between
  notebooks). Tested where the logic is non-trivial (see `viz-standards.md`).
- `notebooks/` — call the viz module; notebooks show the figures inline.
- `wiki/` — a small curated **gallery** of the best static exports (PNG/SVG) so the repo is visually
  legible on GitHub without running anything. Keep committed images small; large interactive HTML exports
  are build artifacts, not committed (see `viz-standards.md`).

## EDA and topology implementation (2026-08-01)

`src/graph_ml/viz/` now provides tested aggregate tables and matplotlib figures for class balance,
monthly volume/impairment rate, company role-degree distributions, connected components, hybrid footprint,
and bounded anonymous ego graphs. The same ego graph can be converted to pyvis for local draggable
exploration; HTML remains an ignored build artifact. The executed studybook is
`notebooks/02_project/03_eda_and_topology.ipynb`.

Facts for the filtered 59,820-instrument modelling graph:

- 45 connected components; the largest contains 81.55% of all instrument + company nodes;
- median company total degree 5, 99th percentile 561, maximum 5,636 — strongly heavy-tailed;
- 15 hybrids touch 12,464 instruments, or 20.84% of this filtered graph.

The last number differs from the earlier 18.7% because that statistic used 12,465 / 66,593 instruments
from the larger pre-filter table. The model-facing figure is 20.84%; both denominators remain explicit.

## Visual studybook language (2026-08-01)

The applied notebooks now use a consistent set of tested teaching figures from `src/graph_ml/viz/`:

- notebook 00 moves from the typed business schema to an anonymous bounded neighborhood from the real
  graph, then visualizes how a two-layer GNN's receptive field expands;
- notebook 01 makes maturity/censoring and seen/cold-start cohort sizes visible, then places PR and ROC
  curves beside their different no-skill references;
- notebook 02 compares baseline PR-AUC against cohort prevalence and turns feature gain into a readable
  ranking;
- notebook 03 covers imbalance, time variation, heavy-tailed degrees, components, hybrid footprint, and
  local topology.
- notebook 04 shows validation-only model selection, cohort-level baseline comparison, class-conditional
  score overlap, seed variability against the fixed LightGBM bar, and an anonymous PCA projection of
  learned instrument embeddings.
- notebook 05 separates prediction and label clocks on a timeline, compares mature p90/p180 cohorts,
  shows how simultaneous events share strictly-prior history, and visualizes the exact p90/p180 identity
  anomaly in the stored bond-feature families. It then compares causal p90 PR-AUC with cohort prevalence
  and ranks the fold-safe LightGBM feature gains.
- notebook 06 contrasts legal and illegal temporal neighbours, draws the four role-aware message channels,
  makes the 180-day exponential decay tangible, inspects real causal tensor/history coverage, shows
  validation-only epoch selection, plots all five seeds against the fixed causal LightGBM bar, and uses a
  root-only comparison to make the graph context's seen/cold-start trade-off visible.
- notebook 07 draws expanding train/validation/development-test windows beside the sealed reported
  holdout, makes label-support checks visible, and compares fold-level PR-AUC with prevalence and full
  neural seed ranges. It then traces temporal attention from graph-constrained candidate events through
  query/key/value scoring, causal masking, softmax, and the weighted message, alongside first-principles
  explanations of time/relation encodings, multi-head attention, neighbour caps, padding, and fallback.
  It also renders one anonymous real K=8 role/age mask and reports tensor shapes and memory cost.
- notebook 08 connects the actual masked multi-head Transformer layers to their query/key/value equations,
  plots validation-only epoch selection, renders a real anonymous relation-by-slot attention heatmap with
  event ages, and compares all four model families over both folds. It distinguishes attention diagnostics
  from causal explanation and records the Transformer's mixed five-seed result.
- notebook 09 puts retrospective impairment and causal p90 models on explicitly separate scoreboards,
  repeats the four-family rolling comparison, and connects paired seeds across no-age, fixed-decay, and
  learned-time treatments so initialization variance remains visible rather than hidden by means. A
  second paired view shows that coverage gating's large fold-1 gain reverses in fold 2.
  A third paired view shows the complete width/regularization factorial and a mean-plus-median table,
  making the fold-2 strong-regularization outlier explicit.
  A fourth paired view links K=2/4/8/16 seeds, followed by a coverage/memory table that makes K's
  information and compute trade-off concrete.
- notebook 10 revisits the temporal role GNN, proves its empty-history root fallback algebraically, compares
  role-specific versus shared transforms, draws 60/180/365-day decay curves, and connects paired seeds for
  relation, decay, and recent-K=8 controls. Means, medians, and ranges stay visible together.

Every graph legend maps color to node or relation semantics. Real-data network figures expose anonymous
topology only: generic `C*`/`I*` labels replace business identifiers, and bounded ego views replace an
unreadable full-graph hairball.

## Deferred: Hugo dashboard / D3 (later milestone, not now)

Redoing the original 2019 Bokeh dashboard as a showcase on a Hugo static site is an explicit **later**
milestone (`specs/roadmap.md`). When we get there, decide **per visualization** whether Plotly's HTML
export is good enough, or whether a specific hero piece — most likely the **network topology** — justifies
a hand-built **D3.js** component. There's a working D3 force-graph reference to adapt in the owner's
`dave_the_human` site (`/brain`). D3 is recorded here as a deliberate, showcase-only, deferred option — it
adds no JavaScript complexity to the analysis work now.

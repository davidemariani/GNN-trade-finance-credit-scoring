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

## Deferred: Hugo dashboard / D3 (later milestone, not now)

Redoing the original 2019 Bokeh dashboard as a showcase on a Hugo static site is an explicit **later**
milestone (`specs/roadmap.md`). When we get there, decide **per visualization** whether Plotly's HTML
export is good enough, or whether a specific hero piece — most likely the **network topology** — justifies
a hand-built **D3.js** component. There's a working D3 force-graph reference to adapt in the owner's
`dave_the_human` site (`/brain`). D3 is recorded here as a deliberate, showcase-only, deferred option — it
adds no JavaScript complexity to the analysis work now.

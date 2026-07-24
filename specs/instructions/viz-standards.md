# Instruction: visualization standards

How visualization gets made in this project. The *what and why* (tooling choices, the four kinds, the
D3/Hugo deferral) lives in `wiki/this-project/visualization.md`; this file is the working convention.

## Where code goes

- Reusable plotting / graph-drawing logic lives in **`src/graph_ml/viz/`** and is imported by notebooks —
  never copy-pasted between notebooks. A function that builds a figure takes data + returns/save a figure;
  it does not also do modelling or data loading.
- Notebooks call the viz module and display figures inline; they don't define reusable plotting functions
  in-cell (a one-off tweak is fine, a reusable chart is not).

## Static vs. interactive

- **Default to matplotlib** for any figure meant to be seen on GitHub or in the README/wiki gallery — it
  renders without running anything and is reproducible.
- Use **Plotly** when interactivity genuinely adds understanding (hover detail, zoom on a dense series) or
  when the figure is destined for the Hugo site. Prefer Plotly over hand-written JS.
- Use **pyvis** for interactive network topology.
- Use **Mermaid** (in markdown) for conceptual/architecture diagrams; **torchview** for the actual model
  computational graph.

## Reproducibility & hygiene

- Every figure must be regenerable from committed code + the (synthetic or real) data — no figures whose
  source is lost. Set the seed (the shared seed utility) for anything stochastic (e.g. layout, t-SNE).
- **Commit small static exports only** (PNG/SVG for the wiki gallery, kept lightweight). Large interactive
  HTML exports (Plotly/pyvis) are **build artifacts, not committed** — generate them on demand or as part
  of the later Hugo build. Add such output dirs to `.gitignore` if they land in the tree.
- Figures that appear in the portfolio narrative should be legible standalone: title, axis labels, units,
  and a one-line caption in the surrounding markdown saying what to take away.
- Don't plot the real confidential data into a committed image if it would expose row-level values; the
  wiki gallery should use synthetic data or aggregates (consistent with `CONSTITUTION.md` §2).

## Testing

- Test the *data-shaping* logic behind a plot (e.g. "the degree-distribution helper returns correct
  counts"), not the pixels. A smoke test that a plotting function runs and returns a figure object on a
  tiny synthetic input is enough for the drawing layer.

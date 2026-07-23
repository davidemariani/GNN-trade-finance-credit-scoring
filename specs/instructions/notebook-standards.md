# Instruction: notebook standards

Notebooks are primary evidence of understanding in this project, not scratch space — they should read
like a short technical article, not a stream of `df.head()` calls.

## Every notebook should

- Start with a markdown cell stating: what this notebook covers, and (for `01_architectures/` and
  `02_project/`) what it assumes the reader already knows from earlier notebooks.
- Prefer prose + math (LaTeX in markdown cells) explaining *why* before the code cell that implements it.
- Keep cells small and single-purpose; a cell that does five things is five cells that got merged.
- End with a short "takeaways" markdown cell — what was learned or concluded, in plain language.
- Have all outputs cleared or re-run fresh before committing (`Restart Kernel and Run All` before saving)
  so the committed notebook reflects a real, reproducible run.

## Folder conventions

- `notebooks/00_foundations/` — numbered by concept order (`00_graph_representations.ipynb`,
  `01_message_passing.ipynb`, ...), no dependency on the project's dataset.
- `notebooks/01_architectures/` — one notebook per architecture, named after it
  (`gcn.ipynb`, `graphsage.ipynb`, `gat.ipynb`, `gin.ipynb`), self-contained with a toy example.
- `notebooks/02_project/` — applied work on the actual trade-finance transaction graph problem; numbered
  in the order they should be read/run (data prep → baseline → GNN experiments → comparison).

## What doesn't belong in a notebook

Anything reused across notebooks (data loading, plotting helpers, model definitions) belongs in
`src/graph_ml/` and gets imported, not copy-pasted between notebooks.

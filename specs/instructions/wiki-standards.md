# Instruction: maintaining the wiki

`wiki/` (see `wiki/README.md`) is a git-tracked, agent-and-human-readable knowledge base — treat it as
living documentation, not a one-time dump.

## When to add or update a page

- **New GNN concept understood** — once a concept from `specs/roadmap.md` Phase 2/4 has actually been
  covered in a `notebooks/00_foundations/` or `notebooks/01_architectures/` notebook, add a corresponding
  lookup entry under `wiki/gnn-concepts/`. Don't pre-write stub pages with no real content — the wiki
  should reflect what's actually been learned, not a wishlist (the wishlist lives in
  `wiki/gnn-concepts/README.md` and `specs/roadmap.md`).
- **A design decision gets made** — e.g. choosing a dataset, choosing which architecture(s) to apply to
  the project's actual task, choosing a validation strategy. Record the decision and its rationale in the
  relevant page (or a new one under `wiki/original-project/` or a future `wiki/this-project/` if the
  decision is about the rework itself rather than the original thesis).
- **Something in the wiki turns out to be wrong or outdated** — fix it in place. Don't leave stale content
  next to new content; a wiki that contradicts itself is worse than no wiki.

## Style

- Each page should be useful read in isolation — link to related pages rather than assuming reading order,
  but don't require jumping through five pages to get one definition.
- Cite the source (a notebook, the original thesis report, a paper) rather than asserting facts unmoored
  from where they came from — especially for anything derived from `wiki/original-project/source/Report.pdf`,
  since that source is not itself published (see `wiki/README.md`).
- Prefer lookup-entry style (definition → intuition → when it matters) over narrative prose — the
  narrative/teaching version belongs in the notebooks, not duplicated here.
- Keep `wiki/gnn-concepts/README.md`'s planned-entries list in sync as items move from "planned" to
  "written."

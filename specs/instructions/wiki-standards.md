# Instruction: maintaining the wiki

`wiki/` (see `wiki/README.md`) is a git-tracked, agent-and-human-readable knowledge base — treat it as
living documentation, not a one-time dump.

## When to add or update a page

- **New GNN concept understood** — once a concept from `specs/roadmap.md` Phase 2/4 has actually been
  covered in a `notebooks/00_foundations/` or `notebooks/01_architectures/` notebook, add a corresponding
  lookup entry under `wiki/gnn-concepts/`. Don't pre-write stub pages with no real content — the wiki
  should reflect what's actually been learned, not a wishlist (the wishlist lives in
  `wiki/gnn-concepts/README.md` and `specs/roadmap.md`).
- **A design decision, discovery, or significant change gets made** — e.g. choosing a dataset, choosing
  which architecture(s) to apply to the project's actual task, choosing a validation strategy, finding
  something unexpected in the data. Record the decision/finding and its rationale in the relevant page
  (`wiki/original-project/` if it's about the original thesis, `wiki/this-project/` if it's about this
  rework), **and** add a one-line pointer to it in `STUDYBOOK.md`'s decision log — see
  `specs/instructions/studybook-standards.md`. Per `CONSTITUTION.md` §2.8, both updates happen in the same
  turn as the decision, not deferred.
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

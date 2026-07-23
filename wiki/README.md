# Wiki — graph-ml knowledge base

A growing, git-tracked knowledge base for this project, meant to be read by both humans and agents
working in this repo. It's the reference layer underneath the notebooks: notebooks teach a concept
step-by-step with code and narrative; the wiki is the fast-lookup version — definitions, decisions, and
context you shouldn't have to re-derive or re-research every session.

See `specs/instructions/wiki-standards.md` for how and when to add to it.

## Structure

- **`original-project/`** — everything about the 2019 thesis project (`networkAnalysisForML`) this repo
  reworks: problem statement, terminology/glossary, data and network construction, feature engineering,
  modelling and validation methodology, results, and the limitations/future-work section that explicitly
  motivates this GNN follow-up. Start with `original-project/overview.md`.
  - `original-project/source/` — the original thesis report PDF. **Gitignored, local-only** (contains a
    minor personal detail in the acknowledgements not suitable for a public repo); the markdown pages
    derived from it are what's published.
- **`gnn-concepts/`** — a growing reference of graph ML concepts and architectures as they're learned in
  this project, written as lookup-style entries (definition, intuition, when it matters) that complement
  the deeper, code-driven treatment in `notebooks/00_foundations/` and `notebooks/01_architectures/`.
  Starts empty/stubbed and fills in alongside the roadmap's Phase 2/4 work.
- **`this-project/`** — decisions and facts specific to *this rework* (not the original thesis) that
  don't belong in `specs/` (which is the plan) or a notebook (which is the how) — e.g. confirmed data
  availability, dataset-specific quirks discovered while building.

## How agents should use this

Before writing code, a notebook, or documentation that touches project-specific terminology (e.g. "what
does p90 mean," "how was the network built") or a GNN concept already covered here, check this wiki first
rather than re-deriving it from scratch or guessing. If something here turns out to be wrong or outdated,
fix the wiki page directly — don't silently work around it.

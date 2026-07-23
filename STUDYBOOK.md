# Studybook — rapid consultation

The fast-orientation doc for this project. Read this to get back up to speed in a few minutes — for depth
on any item, follow the link into `wiki/`, `specs/`, or `CONSTITUTION.md`. This is not a replacement for
those files, it's the index/cheat-sheet layer on top of them. See `specs/instructions/studybook-standards.md`
for how this file gets maintained.

## What this project is, in one paragraph

A from-scratch graph neural network (PyTorch + PyTorch Geometric) rework of `networkAnalysisForML`, a 2019
Msc thesis that predicted trade-finance credit risk (impairment / 90-day delay / 180-day delay) from a
buyer-seller transaction network using hand-engineered "bond graph" features and classical ML. This repo
is deliberately dual-purpose: a genuine graph-ML learning exercise (fundamentals through architectures,
each explained) and a software-engineering portfolio piece (specs-driven, tested, documented), tracked
among the works referenced from the job-application portfolio at `~/Desktop/studybook`. Full mission:
`specs/mission.md`.

## Terminology cheat-sheet

(Full glossary: `wiki/original-project/glossary.md`)

| Term | Meaning |
|---|---|
| imp | Impairment — instrument written off as uncollectable. Rarest, most severe event (2.06% positive). |
| p90 / p180 | Payment delayed ≥90 / ≥180 days. 7.01% / 6.02% positive. |
| Instrument | One invoice/transaction — the atomic unit and label-bearing node in this project's graph. |
| Buyer / debtor | Company that owes payment. Seller / customer | Company financed against the invoice. |
| Effort / flow / energy | Original thesis's bond-graph-theory hand-engineered features (see glossary for exact defs) — this rework aims to learn equivalent signal via message passing instead. |

## Key numbers to remember

- Original results to beat (`wiki/original-project/results.md`): **impairment RF 0.954 AUC**, p90 RF
  0.861, p180 MLP 0.884.
- Real data (`wiki/this-project/data-availability.md`): 59,820 instruments (final feature set), 132
  sellers, 3,176 buyers, dates 2013-07-23 to 2018-12-18. Zero ID overlap between sellers and buyers (no
  hybrid-merging needed for v1).
- v1 graph (`wiki/this-project/graph-design.md`): ~63k nodes / ~120k edges — small enough for full-batch
  training, no sampling infra needed yet.

## Decision log (most recent first — one line + why, link for detail)

- **Graph design v1**: heterogeneous (instrument/buyer/seller node types), static, instrument-centric star
  schema; instrument nodes get raw features, buyer/seller nodes get pure learned embeddings; task = node
  classification on impairment only for v1; leakage handled via an **inductive** train/test split (not a
  transductive one). *Why*: matches label granularity, avoids the clique-blowup of projecting onto a
  homogeneous instrument graph, and closes the same time-leak failure mode the original thesis fought.
  → `wiki/this-project/graph-design.md`
- **Confirmed real dataset available**: owner has the full original pipeline's pickled artifacts locally
  in `data/` (gitignored), not just raw data — every stage from raw transactions to the final bond-graph
  feature set, plus a temporal snapshot file. *Why it matters*: no public/synthetic dataset substitute
  needed. → `wiki/this-project/data-availability.md`
- **2019 methodology is a reference point, not a spec to reproduce**: the rework must genuinely modernize
  (heterogeneous + temporal graph learning), not just port bond-graph feature engineering into PyTorch.
  *Why*: graph ML has moved on since 2019; the original's own limitations section (and the fact it names
  GNNs as "the natural prosecution of the project") supports this. → `wiki/original-project/limitations-and-motivation-for-gnn.md`
- **Deep-read the original thesis in full** rather than working from the README/script skim. *Why*: needed
  precise definitions (imp/p90/p180, bond-graph terms) and the actual results to benchmark against, not
  guesses. → `wiki/original-project/`
- **`Report.pdf` kept local-only (gitignored); only derived wiki markdown is published.** *Why*: minor
  personal detail (family first names in the acknowledgements) unsuited to a public repo; no technical
  content is lost since the wiki captures everything useful.
- **`wiki/` built as plain markdown, in-repo** (not an Obsidian vault, not a static site — yet). *Why*:
  directly agent-readable, git-tracked alongside the code, renders natively on GitHub (helps the portfolio
  angle), no extra tooling dependency. → `wiki/README.md`
- **Specs-driven development structure adopted** (`specs/mission.md`, `tech-stack.md`, `roadmap.md`,
  `instructions/`), mirroring the pattern already used in `~/dave_the_human`. *Why*: consistency with an
  established personal workflow, and it's genuinely useful for a project meant to demonstrate process.
- **Reframed as a dual-purpose learning + portfolio project**, not just an applied rework. *Why*: explicit
  owner direction — this repo needs to be reviewable, not just runnable.
- **New public GitHub repo** (`davidemariani/GNN-trade-finance-credit-scoring`), not a fork/overwrite of
  the original. *Why*: clean separation; the original stays untouched as historical reference.
- **`uv`-managed venv, Python 3.12, isolated per-repo git identity** (personal GitHub, not work GitLab).
  *Why*: matches the isolation pattern already used for other personal projects on this machine.
  → `specs/tech-stack.md`, `CONSTITUTION.md` §1

## Where things stand right now

Roadmap phase: **Phase 3 (Baseline & Data)** in progress — graph design and task framing are decided
(above); implementing `src/graph_ml/data/` graph construction and the inductive train/test split is next.
See `specs/roadmap.md` for the full phased plan and `BACKLOG.md` for the live task list.

## Map of the docs (what to read for what)

- **This file** — fast orientation, decision log.
- `CONSTITUTION.md` — the rules (isolation, principles, folder structure). Read first if you're an agent
  picking this up cold.
- `specs/` — the plan (`mission.md`, `tech-stack.md`, `roadmap.md`) and recurring workflows (`instructions/`).
- `wiki/` — the knowledge base: `original-project/` (the 2019 thesis, in depth), `this-project/` (decisions
  and facts about this rework), `gnn-concepts/` (growing GNN reference).
- `BACKLOG.md` — live task tracker.
- `USAGE.md` — how to actually run things.

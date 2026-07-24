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

- Original results as a **reference point** (`wiki/original-project/results.md`): impairment RF 0.954, p90
  RF 0.861, p180 MLP 0.884 — these are *ROC AUC*, which misleads at 2% positives, so they're for
  comparability, not a target to chase. This project's headline metric is **PR-AUC** (`evaluation.md`).
- Real data (`wiki/this-project/data-availability.md`): 59,820 instruments (final feature set), 132 seller
  IDs, 3,176 buyer IDs, dates 2013-07-23 to 2018-12-18. **15 hybrids** (companies that are both buyer and
  seller) — resolvable only by company *name*, since buyer/seller IDs are separate spaces; hybrids touch
  18.7% of instruments (the real network-contagion signal).
- Split facts at T=2018-04-30 (impairment): ~56% of test companies unseen in training (cold-start),
  ~32% of test instruments still `is_open` (label maturity) — both drive design choices, see evaluation.md.
- v1 graph (`wiki/this-project/graph-design.md`): company + instrument node types, ~63k nodes / ~120k
  edges — small enough for full-batch training, no sampling infra needed yet.

## Decision log (most recent first — one line + why, link for detail)

- **Data converted to Parquet + a `.gitignore` bug fixed (2026-07-24)**: ran the conversion
  (`src/graph_ml/data/convert.py`, tested), verified **exact** cell-by-cell value equality against the
  originals (not just shape) — only cosmetic, expected representation changes (list→array, object→
  StringDtype), documented so future code doesn't assume the old shapes. While building this, found
  `.gitignore`'s unanchored `data/` rule was silently excluding **any** directory named `data` anywhere in
  the repo, not just the top-level one — it was hiding the brand-new `src/graph_ml/data/` and `tests/data/`
  source code from git. Fixed to `/data/` (anchored). *Why it matters*: caught before any commit, but would
  have silently lost source code otherwise — a reminder to `git status`-check after adding any new
  top-level-named directory. Originals kept until the off-GitHub backup is actually done (not yet — backup
  location is still an open choice). → `wiki/this-project/data-availability.md`
- **Visualization approach (2026-07-24)**: Python stack — matplotlib (static, GitHub-rendered) + Plotly
  (interactive, Hugo-ready HTML export) + pyvis (interactive network topology); Mermaid + torchview for
  architecture/message-passing diagrams. Visualization is cross-cutting (woven into every phase, reusable
  code in `src/graph_ml/viz/`), not a phase. **D3.js is deferred, showcase-only** — reserved for a few hero
  pieces on the eventual Hugo static-site dashboard (Phase 6), adapting the D3 force-graph from
  `dave_the_human`. *Why*: get interactivity from Python for free now; reserve bespoke JS for the showcase.
  → `wiki/this-project/visualization.md`
- **Data storage → Parquet, out of repo (2026-07-24)**: convert working data from 2019 `.pkl` to Parquet
  (zstd) — measured 7-14% of pickle size, safe, fast; keep `data/` gitignored (not committed even though it
  would fit), back up off-GitHub, revisit further anonymization later. Reviewers get a synthetic generator
  for runnability. *Why*: efficient + safe + confidential, without repo bloat. → `wiki/this-project/data-availability.md`
- **Plan review + corrections (2026-07-24)**: adversarial review of the whole plan against the data.
  Corrected the hybrid finding (15 hybrids exist, by *name* not ID — earlier "zero overlap" was wrong);
  revised the graph schema to **company + instrument** (two node types) so hybrids/contagion work;
  replaced "pure learned company embeddings" with **time-windowed aggregated company features** (the
  embedding+inductive combo was self-contradictory and 25% of test instruments are cold-start); added
  `evaluation.md` (PR-AUC co-primary since AUC misleads at 2% positives, label-maturity rule, strong
  gradient-boosting baseline). *Why*: correctness holes + honest-benchmarking. → `wiki/this-project/graph-design.md`, `wiki/this-project/evaluation.md`
- **Graph design v1** (superseded in part by the review above): heterogeneous, static, instrument-centric;
  node classification on impairment only for v1; inductive train/test split for leakage control. *Why*:
  matches label granularity, avoids clique-blowup of a homogeneous instrument graph, closes the time-leak
  failure mode the original fought. → `wiki/this-project/graph-design.md`
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

All planning/design is done (Phases 0-1 + the design decisions). **Nothing is built yet** — `src/graph_ml/`
is still a skeleton. The immediate next action is the start of the data pipeline: **convert `data/` to
Parquet + back it up**, then the **synthetic data generator**, then graph construction → split/metrics →
LightGBM baseline → the **Phase 3.5 vertical slice** (the priority milestone). See `specs/roadmap.md` for
the full phased plan (and the "plan at a glance" table) and `BACKLOG.md` for the ordered next tasks.

## Map of the docs (what to read for what)

- **This file** — fast orientation, decision log.
- `CONSTITUTION.md` — the rules (isolation, principles, folder structure). Read first if you're an agent
  picking this up cold.
- `specs/` — the plan (`mission.md`, `tech-stack.md`, `roadmap.md`) and recurring workflows (`instructions/`).
- `wiki/` — the knowledge base: `original-project/` (the 2019 thesis, in depth), `this-project/`
  (`data-availability`, `graph-design`, `evaluation`, `visualization`), `gnn-concepts/` (growing GNN reference).
- `BACKLOG.md` — the ordered next few tasks (Now / Next up / Parked); roadmap holds the full plan.
- `USAGE.md` — how to actually run things.

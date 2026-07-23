# Mission

Build a from-scratch graph neural network (PyTorch + PyTorch Geometric) rework of
[`networkAnalysisForML`](https://github.com/davidemariani/networkAnalysisForML) that serves **two goals
at once**:

1. **A deep, demonstrable, hands-on education in graph machine learning** — starting from the basics of
   representing and reasoning about graphs, through the message-passing framework, to specific
   architectures (GCN, GraphSAGE, GAT, GIN, ...), each understood well enough to explain *why* it works
   and *when* to reach for it — not just "run the library call."
2. **A portfolio piece** demonstrating software engineering practice alongside the ML content: specs-driven
   development, well-organized and tested code, clear documentation, and honest reporting of results. This
   is one of several works tracked for job applications in `~/Desktop/studybook` — it needs to stand on its
   own as evidence of both ML depth and engineering discipline.

This isn't a speculative choice of problem — the original thesis's own conclusion states *"working towards
neural ensembles and graph neural networks seems to be the natural prosecution of the project,"* after a
literature review that had already surveyed core GNN papers but deliberately deferred them. This rework is
that stated next step. See `wiki/original-project/` for the full grounding, especially
`limitations-and-motivation-for-gnn.md` for why this needs to be a genuine modernization (heterogeneous,
temporal-aware graph learning) rather than a same-methods-different-library port of 2019 techniques.

## Core Goals

1. **Understand before implementing.** Every architecture used in `src/graph_ml/` has a companion notebook
   under `notebooks/01_architectures/` that explains the intuition and math before showing the code —
   the notebooks are not an afterthought, they're the primary evidence of understanding.
2. **Foundations aren't skipped.** `notebooks/00_foundations/` covers the basics of graph representation,
   the message-passing framework, spectral vs. spatial convolutions, over-smoothing, and other concepts
   that a GNN practitioner should be able to explain from first principles — even where a real dataset
   isn't needed to illustrate them.
3. **Apply it to a real problem.** `notebooks/02_project/` and `src/graph_ml/` apply what's learned to the
   original problem: predicting trade-finance/credit outcomes from a buyer/seller transaction network,
   with an honest comparison against classical baselines (see `CONSTITUTION.md` §0).
4. **Software engineering is part of the showcase, not incidental to it.** Production-shaped code in
   `src/graph_ml/` (typed, tested, documented), specs-driven workflow (this `specs/` folder), and a
   deliberate git/GitHub history are all things a reviewer of this portfolio piece should be able to see
   and judge, not just the notebooks.

## Audience

Two audiences, both real: **future-me** relearning or extending this work, and **anyone reviewing it as a
portfolio piece** (recruiters, hiring managers, technical interviewers) who should come away convinced this
person understands graph ML deeply and writes disciplined, well-tested code.

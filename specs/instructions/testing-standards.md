# Instruction: testing standards

This project's code (`src/graph_ml/`) is meant to demonstrate solid engineering practice, so it's held to
a real testing bar — not just "does it run."

## What needs a test

- Every public function/class in `src/graph_ml/` — no exceptions for "it's simple," since simple code
  regresses silently just as easily as complex code.
- Graph construction / data pipeline code: test on small, hand-constructed inputs where the expected
  output can be verified by hand (e.g. a 3-node graph with known edges → known adjacency structure).
- Model code: shape correctness, behavior on edge cases (isolated nodes, self-loops, empty graphs where
  applicable), and a "loss decreases on a tiny overfit case" smoke test for anything trainable.
- Metrics/evaluation code: known input → known output (e.g. a hand-computed AUC on a tiny prediction set).

## What doesn't need a test

- Notebooks themselves (see `notebook-standards.md` — their "test" is that they run top-to-bottom cleanly
  and their conclusions are reproducible).
- One-off exploratory scripts that never made it into `src/graph_ml/`.

## Conventions

- Tests live in `tests/`, mirroring the `src/graph_ml/` package structure (e.g.
  `src/graph_ml/models/gcn.py` → `tests/models/test_gcn.py`).
- Use small, synthetic, in-memory graphs for unit tests — never depend on `data/` (gitignored, not
  guaranteed to exist) for tests to pass.
- Run `pytest` before considering any change to `src/graph_ml/` complete.

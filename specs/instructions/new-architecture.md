# Instruction: adding a new GNN architecture

Follow this every time a new architecture (GCN, GraphSAGE, GAT, GIN, or anything later) is added. The
order matters — understanding comes before applied code.

1. **Explain it first.** Write (or extend) a notebook in `notebooks/01_architectures/` covering:
   - The core idea and what problem it solves relative to prior architectures (e.g. "GAT adds learned
     attention weights over neighbors instead of GCN's fixed degree-based normalization").
   - The math of a single layer's forward pass, in enough detail to reimplement without looking it up.
   - A toy example (small hand-built graph) showing the mechanism concretely — not just prose.
   - Citation of the original paper.
2. **Implement it in `src/graph_ml/models/`**, not in the notebook. The notebook may prototype it first,
   but the version that gets used in the project moves to `src/graph_ml/models/<architecture>.py` with:
   - Type hints on public functions/classes.
   - A docstring that states the layer's inputs/outputs shapes and any assumptions.
3. **Test it.** Add `tests/models/test_<architecture>.py` covering at minimum: output shape correctness,
   behavior on a trivial graph (e.g. a single edge, an isolated node), and that gradients flow (a training
   step reduces loss on a toy overfit case).
4. **Apply it** in `notebooks/02_project/` against the Phase 3 baseline, and record the honest result —
   better, worse, or inconclusive — in that notebook and (if it changes overall conclusions) in
   `specs/roadmap.md` / `BACKLOG.md`.

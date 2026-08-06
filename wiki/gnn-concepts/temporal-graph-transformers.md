# Temporal graph Transformers

## Definition

A temporal graph Transformer applies attention to a node or event's **causally available graph
neighbourhood**. The query is the current event; keys and values are earlier related events; relation and
time encodings tell attention how an event is connected and how old it is. A hard mask must exclude the
current, simultaneous, and future events.

This differs from a plain Transformer over tabular rows. Row attention without entity/role structure does
not represent the seller–buyer graph, while attention over the full event table would leak future events.

## Intuition

The current temporal role GNN compresses all eligible history in each role channel to one exponentially
weighted mean. This assumes importance is mostly a smooth function of age. Attention can learn that an
older but unusually large invoice matters more than several recent routine invoices, conditioned on the
current invoice.

For current invoice `i` and legal historical event `j`, a conceptual attention logit is:

`score(i, j) = query(x_i) · key(x_j, relation_j, time_encoding(t_i - t_j))`.

Softmax is taken only over a bounded set with `t_j < t_i`. The weighted value sum replaces the fixed
decayed mean. Seller/buyer roles can use separate attention heads or learned relation embeddings.

## Why it may help here

- High-degree companies can contain thousands of invoices; learned relevance may avoid washing out a
  small recent warning signal.
- Event importance may depend jointly on amount, terms, role, and age rather than age alone.
- The temporal backtests show relation context can help seen companies, but its value changes across
  periods. Attention is a plausible way to make that context more selective.

## What it cannot solve

A Transformer cannot attend to history that does not exist. Cold-start still requires strong
current-invoice features or transferable company descriptors. Attention also adds seed variance,
parameter count, and leakage surfaces: an incorrect causal mask, future-derived neighbour list, or
padding convention can invalidate the experiment.

## Project decision and result

A TGAT-style candidate was evaluated using the pre-holdout development folds, five seeds, a 39,745
parameter budget, and the same p90 label clock. The reported April–December 2018 holdout was not reused
for selection.

The first candidate should be deliberately small:

- current invoice query;
- strictly-prior seller/buyer role events as keys/values;
- continuous age encoding;
- relation-specific heads or embeddings;
- a fixed recent-neighbour cap for hubs;
- explicit empty-history fallback to the root path.

The model core is now implemented as `TemporalGraphTransformer`. It projects the current invoice to a
query/root state; adds learned relation and log-age encodings to historical event keys/values; performs
multi-head attention only for rows with at least one valid event; and applies residual normalization plus
a feed-forward block. Tests establish that valid weights sum to one, padding weights are zero, changing
masked values cannot change logits, empty history produces a zero attention message, and loss decreases
on a tiny overfit case. The rolling wrapper fits preprocessing on the legal training population, selects
epoch on validation PR-AUC, then refits a fresh model and encoder on train plus validation.

The result is mixed. Fold 1 Transformer PR-AUC is 0.016 overall and 0.034 for seen companies, versus
0.012 and 0.006 for the fixed-decay GNN. In fold 2 it reaches 0.087 overall and 0.146 seen, below
LightGBM (0.120 / 0.157) and the fixed-decay GNN (0.119 / 0.194). Cold-start remains near or below the
best alternatives because attention cannot select absent history. This candidate is therefore retained
but not promoted. Next ablations must remain validation-only: time encoding, capacity/regularization, K,
and root/message fusion.

### Time-treatment ablation

The first validation-only ablation holds histories, relations, architecture, optimization, and paired
seeds fixed. `learned` uses the log-age MLP; `fixed_decay` adds
`-log(2) * age / 180` directly to every legal attention logit; `none` withholds age. Padding remains
negative infinity and causal eligibility is unchanged. Fold-1 mean validation PR-AUC is 0.3016 learned,
0.3046 fixed, and 0.2968 none; fold 2 is 0.0203, 0.0199, and 0.0195. These differences are small relative
to initialization spread and fixed decay does not win both origins. Learned time therefore stays as the
default; coverage-aware root/message fusion became the next hypothesis. See notebook 09 and
`results/temporal_transformer_time_ablation.csv`.

### Root/message fusion ablation

The original residual uses `root + message`. The tested gate uses `root + g * message`, where one scalar
`g` in `[0, 1]` depends on the root representation and the occupied fraction of each relation's K slots.
It raises fold-1 five-seed mean validation PR-AUC from 0.3016 to 0.4272 but lowers fold 2 from 0.0203 to
0.0101. This large sign-reversing effect is evidence that coverage interacts with regime, not evidence
for a generally superior gate. Residual fusion remains default; capacity/regularization became next. The
control preserves PyTorch's random stream around dormant gate initialization so its historical training
trajectory remains exact. Artifact: `results/temporal_transformer_fusion_ablation.csv`.

### Capacity and regularization ablation

A 2×2 validation design compares 64 versus 32 hidden units and current regularization (dropout 0.20,
weight decay `1e-4`) versus a stronger bundle (0.35, `1e-3`). Fold 1 favours the current wide model at
0.3016; the other means are 0.2911–0.2932. In fold 2, wide strong regularization averages 0.0279 versus
0.0203, but a single 0.0941 seed drives the mean and its median is 0.0100. Compact variants fall to
0.0073 and 0.0055. Neither reduced capacity nor the stronger bundle is promoted. The original wide
configuration stays frozen; K became the final planned information-budget control. Artifact:
`results/temporal_transformer_capacity_ablation.csv`.

### Recent-event budget K

K changes retained legal information and tensor memory without changing trainable parameter count. K=2,
4, 8, and 16 use approximately 27.8, 55.7, 111.4, and 222.7 MiB at 12 event features. Fold 1 prefers
short context: means fall from 0.3301 at K=2 to 0.2964 at K=16. Fold 2 needs more history: K=2 falls to
0.0069 and K=8 leads at 0.0203; K=16 reaches 0.0175. At K=8, 31.3% of fold-1 and 43.8% of fold-2
relation histories hit the cap, confirming that the later regime is denser. No K dominates both origins.
K=8 remains the predeclared compromise and the component-ablation sequence stops. Artifacts:
`results/temporal_transformer_k_ablation.csv` and `results/temporal_transformer_k_coverage.csv`.

## Related material

- General event-time principles: [Temporal graphs](temporal-graphs.md)
- Current fixed-decay model: [Temporal role GNN](temporal-role-gnn.md)
- Applied derivation, fitted attention diagnostic, and result: `notebooks/02_project/08_temporal_graph_transformer.ipynb`
- Cross-model scoreboards and paired component ablations: `notebooks/02_project/09_model_comparison_and_time_ablation.ipynb`
- Bounded causal event tensors: `src/graph_ml/data/temporal_graph.py`
- Evaluation contract: `wiki/this-project/evaluation.md`

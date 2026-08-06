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

## Project decision

A TGAT-style candidate is worth implementing **after** a tested bounded recent-event tensor exists. It
must use the pre-holdout development folds in `notebooks/02_project/07_temporal_backtesting.ipynb`, five
seeds, a comparable parameter budget, and the same p90 label clock. Architecture and hyperparameters are
chosen on development-fold aggregates; the reported April–December 2018 holdout is not reused for
selection.

The first candidate should be deliberately small:

- current invoice query;
- strictly-prior seller/buyer role events as keys/values;
- continuous age encoding;
- relation-specific heads or embeddings;
- a fixed recent-neighbour cap for hubs;
- explicit empty-history fallback to the root path.

## Related material

- General event-time principles: [Temporal graphs](temporal-graphs.md)
- Current fixed-decay model: [Temporal role GNN](temporal-role-gnn.md)
- Applied backtest and visual comparison: `notebooks/02_project/07_temporal_backtesting.ipynb`
- Evaluation contract: `wiki/this-project/evaluation.md`

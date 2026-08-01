# Transductive vs. inductive graph learning

## Definition

- **Transductive learning** predicts labels for nodes in a graph whose entities and topology were already
  available during training, even if those labels were hidden.
- **Inductive learning** must apply learned parameters to newly arriving nodes or entirely new graphs.

The distinction is about what is new at inference time, not merely whether labels were held out.

## Intuition

A learned lookup embedding for node ID 42 works only because node 42 existed while fitting. It gives no
principled representation for a new node 9001. An inductive model instead computes representations from
features and local structure using transformations shared across nodes.

## When it matters here

Post-cutoff invoices are new instrument nodes, and 23.76% of the mature test cohort involves at least one
company absent before the cutoff. Company features therefore use observable historical aggregates and an
all-zero no-history representation rather than per-company learned ID embeddings. The inference graph lets
new instruments receive pre-cutoff company context without sending messages back into company states.

See `notebooks/02_project/01_temporal_split_and_metrics.ipynb` for the studybook explanation and
`wiki/this-project/evaluation.md` for the exact split contract.

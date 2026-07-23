# Original project: results

All metrics are ROC AUC on a held-out test set. These are the numbers a GNN approach needs to be honestly
compared against — not beaten by construction, beaten (or not) by evidence.

## Baseline features (Tier 1 only)

| Event | RF shuffle-mode AUC | RF time-mode AUC | SGD shuffle-mode AUC | SGD time-mode AUC |
|---|---|---|---|---|
| Impairment | 0.912 | 0.862 | 0.784 | 0.779 |
| Pastdue90 | 0.946 | 0.844 | 0.780 | 0.772 |
| Pastdue180 | 0.959 | 0.671 | 0.781 | 0.720 |

Note the large shuffle-vs-time-mode gap, especially for pastdue180 (0.959 → 0.671) — direct evidence of
how much a naive (non-time-aware) split overstates real performance.

With event-specific tuning (time-sequential mode): impairment RF 0.862→0.854, pastdue90 RF 0.844→0.856,
pastdue180 RF 0.671→0.741 (the biggest gain, from tuning alone).

## Enriched features (Tier 1 + bond-graph Tier 2)

| Event | RF enriched shuffle-mode AUC | RF enriched time-sequential (leak-prevented) AUC |
|---|---|---|
| Impairment | 0.967 | 0.951 |
| Pastdue90 | 0.960 | 0.848 |
| Pastdue180 | 0.970 | 0.762 |

SGD did not meaningfully benefit from the bond-graph features in any configuration.

## Final results (enriched + event-specific tuning, and neural models for p90/p180)

| Event | Best model | Best test AUC |
|---|---|---|
| Impairment | RF (enriched, tuned) | **0.954** |
| Pastdue90 | RF (enriched, tuned) | **0.861** |
| Pastdue180 | MLP | **0.884** |

Supporting detail: pastdue90 RNN (GRU) ≈0.805-0.806, below enriched RF. Pastdue180 RNN (LSTM/GRU)
≈0.821-0.825 — better than baseline classifiers but below the MLP.

## Headline conclusions (thesis's own framing)

- **Impairment**: the clearest win — bond-graph enrichment took RF from 0.862 → 0.954, "demonstrating the
  potential of networked data in transactions credit scoring."
- **Pastdue90**: enrichment "turned out to be more difficult" — only marginal gains over baseline; neither
  MLP nor RNN offered significant improvement here.
- **Pastdue180**: bond-graph features gave an initial AUC increase, but the larger gain came from neural
  networks — MLP (0.884) and RNN (0.825) substantially outperformed classical models for this event
  specifically.

## Reading these numbers today

Treat these as the **historical reference point**, not the bar to reproduce via the same methods — see
`limitations-and-motivation-for-gnn.md` for why the modelling approach itself (not just the numbers)
needs rethinking for 2026, not just re-implementing in PyTorch.

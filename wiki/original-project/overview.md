# Original project: overview

Source: `networkAnalysisForML` — Davide Mariani's 2019 MSc Data Science thesis at Birkbeck College,
University of London, *"Networked Data and Machine Learning for Supply Chain Predictive Modelling,"*
done in collaboration with **Tradeteq Ltd**, supervised by Prof. George D. Magoulas (Birkbeck) and
Michael Boguslavsky (Tradeteq). Full derivation of this page: `wiki/original-project/source/Report.pdf`
(local-only, see `wiki/README.md`).

## The business problem

**Receivables financing / factoring**: a seller (typically an SME) that has sold goods/services to a
buyer (debtor) gets paid early against its outstanding invoice by a financier, who then collects from the
buyer at maturity. The core question: *given a network of buyer-seller trade transactions, can machine
learning predict which financed instruments (invoices) will become a credit event — impaired, or delayed
by 90/180 days — before it happens?*

**Tradeteq** is a fintech company building "Machine Learning Credit Analytics for Trade Finance" —
software that scores trade-finance receivables (normally an opaque, hard-to-assess asset class) to make
them investable by institutional investors. Tradeteq supplied the anonymized dataset, domain terminology,
and its own pre-existing baseline models (SGD + Random Forest), which the thesis set out to improve on.

## The core hypothesis

Traditional trade-finance credit scoring evaluates each debtor/instrument mostly in isolation. This
thesis's hypothesis: a company's **position in the buyer-seller trade network** carries additional
predictive signal (similar to contagion/systemic risk in financial networks), and this can be captured by
modelling the transaction network with **bond graph theory** (a physics formalism for energy/power flow
in dynamic systems) — extracting "effort/flow/energy" features per node/edge — then feeding those into
classical ML and neural network classifiers, and comparing against Tradeteq's non-network-aware baseline.

## Why this matters for the GNN rework

The thesis's own conclusion states: *"working towards neural ensembles and graph neural networks seems to
be the natural prosecution of the project."* Its literature review already surveyed core GNN literature
(Niepert, Ahmed & Kutzkov; Zhou et al.; Battaglia et al.) but deliberately deferred them in favor of the
hand-engineered bond-graph approach. **This repo picks up exactly where that thesis left off** — this
isn't a speculative reapplication of GNNs to an unrelated problem, it's the author's own stated next step.

See also: `glossary.md`, `data-and-network-construction.md`, `feature-engineering.md`,
`modelling-and-validation.md`, `results.md`, `limitations-and-motivation-for-gnn.md`.

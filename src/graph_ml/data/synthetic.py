"""Synthetic trade-finance instrument dataset generator.

Produces a small, schema-faithful fake dataset — same column names/dtypes as the real pipeline
data (see wiki/this-project/data-availability.md, table `02_instrumentsdf_2`) — so the graph
construction and modelling pipeline can be built, tested, and demoed without the private real
data. Company identity mirrors the real data's most important quirk: `n_hybrids` company *names*
are shared between the seller and buyer pools but assigned distinct IDs per role, because
`customer_id`/`debtor_id` are separate ID spaces in the real data too (see graph-design.md
"Entity resolution: companies are matched by NAME, not ID").
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Matches the real data's observed category values/proportions (see data-availability.md).
CURRENCIES = ("Schweizer Franken", "Euro", "US-Dollar", "Britisches Pfund")
CURRENCY_WEIGHTS = (0.914, 0.079, 0.005, 0.002)
FACTORING_TYPES = ("Full Service", "Reverse Factoring")
FACTORING_WEIGHTS = (0.998, 0.002)


def _zipfian_weights(n: int) -> np.ndarray:
    """A right-tailed weight distribution so a handful of companies act as hubs, like the real data."""
    ranks = np.arange(1, n + 1)
    weights = 1.0 / ranks
    return weights / weights.sum()


def generate_instruments(
    n_instruments: int = 2_000,
    n_sellers: int = 40,
    n_buyers: int = 400,
    n_hybrids: int = 15,
    start_date: str = "2013-07-23",
    end_date: str = "2018-12-18",
    impairment_rate: float = 0.0206,
    pastdue90_rate: float = 0.0701,
    pastdue180_rate: float = 0.0602,
    maturity_days: int = 180,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a fake instruments table matching the real pipeline's schema.

    `n_hybrids` company names are present in both the seller and buyer pools (each with a
    distinct, role-specific ID) and are guaranteed to actually appear in both roles in the
    output -- under Zipfian sampling a low-weight hybrid could otherwise go undrawn in one role
    by chance, which would silently defeat the point of asking for hybrids. The label columns
    respect maturity: an `is_open` instrument never carries a resolved-positive label, mirroring
    the real censoring behaviour the evaluation methodology has to handle (see evaluation.md
    "Label maturity / censoring").
    """
    if n_hybrids > min(n_sellers, n_buyers):
        raise ValueError("n_hybrids cannot exceed n_sellers or n_buyers")
    if n_instruments < 2 * n_hybrids:
        raise ValueError(
            "n_instruments must be at least 2 * n_hybrids to guarantee both roles"
        )

    rng = np.random.default_rng(seed)

    hybrid_names = [f"hybrid-company-{i:03d}" for i in range(n_hybrids)]
    seller_only_names = [
        f"seller-company-{i:03d}" for i in range(n_sellers - n_hybrids)
    ]
    buyer_only_names = [f"buyer-company-{i:03d}" for i in range(n_buyers - n_hybrids)]

    seller_names = np.array(hybrid_names + seller_only_names)
    buyer_names = np.array(hybrid_names + buyer_only_names)
    rng.shuffle(seller_names)
    rng.shuffle(buyer_names)

    # Separate ID spaces per role, even for hybrids -- mirrors customer_id/debtor_id in the real data.
    seller_ids = {name: f"C{i:05d}" for i, name in enumerate(seller_names)}
    buyer_ids = {name: f"D{i:05d}" for i, name in enumerate(buyer_names)}

    chosen_sellers = rng.choice(
        seller_names, size=n_instruments, p=_zipfian_weights(len(seller_names))
    )
    chosen_buyers = rng.choice(
        buyer_names, size=n_instruments, p=_zipfian_weights(len(buyer_names))
    )
    # Force each hybrid into one guaranteed seller-role row and one guaranteed buyer-role row.
    for i, name in enumerate(hybrid_names):
        chosen_sellers[i] = name
        chosen_buyers[n_hybrids + i] = name

    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    offsets = rng.integers(0, (end - start).days + 1, size=n_instruments)
    invoice_dates = start + pd.to_timedelta(offsets, unit="D")
    terms = rng.integers(14, 121, size=n_instruments)
    due_dates = invoice_dates + pd.to_timedelta(terms, unit="D")

    amounts = rng.lognormal(mean=7.7, sigma=1.3, size=n_instruments).round(2)
    currencies = rng.choice(CURRENCIES, size=n_instruments, p=CURRENCY_WEIGHTS)
    factoring_types = rng.choice(
        FACTORING_TYPES, size=n_instruments, p=FACTORING_WEIGHTS
    )

    # Younger instruments are more likely to still be open/unresolved, same shape as the real data.
    age_days = (end - invoice_dates).days.to_numpy()
    open_prob = np.clip(1.0 - age_days / maturity_days, 0.02, 0.95)
    is_open = rng.random(n_instruments) < open_prob

    def resolved_flag(rate: float) -> np.ndarray:
        return (rng.random(n_instruments) < rate) & ~is_open

    return pd.DataFrame(
        {
            "uid": [f"synth-{i:06d}" for i in range(n_instruments)],
            "customer_id": [seller_ids[name] for name in chosen_sellers],
            "customer_name_1": chosen_sellers,
            "debtor_id": [buyer_ids[name] for name in chosen_buyers],
            "debtor_name_1": chosen_buyers,
            "invoice_date": invoice_dates,
            "due_date": due_dates,
            "invoice_amount": amounts,
            "currency": currencies,
            "factoring_type": factoring_types,
            "is_open": is_open,
            "has_impairment1": resolved_flag(impairment_rate),
            "is_pastdue90": resolved_flag(pastdue90_rate),
            "is_pastdue180": resolved_flag(pastdue180_rate),
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out_path",
        type=Path,
        nargs="?",
        default=Path("data/synthetic_instruments.parquet"),
    )
    parser.add_argument("--n-instruments", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = generate_instruments(n_instruments=args.n_instruments, seed=args.seed)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_path, compression="zstd")
    print(f"{args.out_path}  ({len(df)} instruments)")


if __name__ == "__main__":
    main()

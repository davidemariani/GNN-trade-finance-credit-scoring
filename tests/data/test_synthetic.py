import numpy as np
import pandas as pd
import pytest

from graph_ml.data.synthetic import generate_instruments

EXPECTED_COLUMNS = {
    "uid",
    "customer_id",
    "customer_name_1",
    "debtor_id",
    "debtor_name_1",
    "invoice_date",
    "due_date",
    "invoice_amount",
    "currency",
    "factoring_type",
    "is_open",
    "has_impairment1",
    "is_pastdue90",
    "is_pastdue180",
}


def test_has_expected_schema():
    df = generate_instruments(n_instruments=100, seed=0)
    assert set(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 100
    assert df["uid"].is_unique


def test_is_deterministic_given_a_seed():
    a = generate_instruments(n_instruments=200, seed=42)
    b = generate_instruments(n_instruments=200, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_hybrids_appear_by_name_in_both_roles_with_distinct_ids():
    df = generate_instruments(
        n_instruments=5_000, n_sellers=40, n_buyers=400, n_hybrids=15, seed=1
    )

    seller_names = set(df["customer_name_1"])
    buyer_names = set(df["debtor_name_1"])
    hybrids = seller_names & buyer_names
    # Exact, not "roughly": guaranteed by construction so hybrid-handling logic downstream always
    # has something to exercise, regardless of how Zipfian sampling happens to land for a given seed.
    assert len(hybrids) == 15

    # Separate ID spaces per role, even for the same hybrid company name (mirrors the real data).
    seller_ids = set(df["customer_id"])
    buyer_ids = set(df["debtor_id"])
    assert seller_ids.isdisjoint(buyer_ids)


def test_rejects_more_hybrids_than_available_companies():
    with pytest.raises(ValueError):
        generate_instruments(n_sellers=10, n_buyers=400, n_hybrids=15)


def test_rejects_too_few_instruments_to_guarantee_hybrid_coverage():
    with pytest.raises(ValueError):
        generate_instruments(n_instruments=10, n_sellers=40, n_buyers=400, n_hybrids=15)


def test_open_instruments_never_carry_a_resolved_positive_label():
    df = generate_instruments(n_instruments=5_000, seed=2)
    open_rows = df[df["is_open"]]
    assert not open_rows["has_impairment1"].any()
    assert not open_rows["is_pastdue90"].any()
    assert not open_rows["is_pastdue180"].any()


def test_label_rates_are_roughly_on_target():
    df = generate_instruments(n_instruments=20_000, impairment_rate=0.0206, seed=3)
    rate = df["has_impairment1"].mean()
    # Loose tolerance: is_open censoring pulls the observed rate below the input rate by design.
    assert 0.0 < rate < 0.0206 + 0.01


def test_dates_fall_within_the_requested_range():
    df = generate_instruments(
        n_instruments=1_000, start_date="2013-07-23", end_date="2018-12-18", seed=4
    )
    assert df["invoice_date"].min() >= pd.Timestamp("2013-07-23")
    assert df["invoice_date"].max() <= pd.Timestamp("2018-12-18")
    assert (df["due_date"] >= df["invoice_date"]).all()


def test_invoice_amount_is_positive():
    df = generate_instruments(n_instruments=500, seed=5)
    assert (df["invoice_amount"] > 0).all()
    assert np.isfinite(df["invoice_amount"]).all()

from __future__ import annotations

import pandas as pd
import pytest
import torch

from graph_ml.data import (
    GraphBuildConfig,
    build_trade_finance_graph,
    build_trade_finance_graph_from_parquet,
    canonicalize_company_name,
)


@pytest.fixture
def instruments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid": ["post-2", "pre-2", "pre-1", "post-1"],
            "customer_name_1": [
                "New Company",
                "Supplier B",
                "  ACME   GmbH ",
                "Acme GmbH",
            ],
            "debtor_name_1": [
                "Buyer Three",
                "acme gmbh",
                "Buyer One",
                "Buyer Two",
            ],
            "invoice_date": pd.to_datetime(
                ["2020-03-01", "2020-01-20", "2020-01-10", "2020-02-10"]
            ),
            "due_date": pd.to_datetime(
                ["2020-03-31", "2020-02-19", "2020-02-09", "2020-03-11"]
            ),
            "input_date": pd.to_datetime(
                ["2020-03-01", "2020-01-21", "2020-01-10", "2020-02-10"]
            ),
            "invoice_amount": [999_999.0, 200.0, 100.0, 400.0],
            "purchase_amount": [900_000.0, 180.0, 80.0, 360.0],
            "currency": ["Martian Dollar", "EUR", "EUR", "USD"],
            "factoring_type": [
                "Unknown Product",
                "Full Service",
                "Full Service",
                "Full Service",
            ],
            "has_impairment1": [True, False, False, True],
        }
    )


def _build(instruments: pd.DataFrame):
    return build_trade_finance_graph(instruments, GraphBuildConfig(cutoff="2020-02-01"))


def test_canonicalize_company_name_is_conservative_and_stable():
    assert canonicalize_company_name("  ACME\u00a0 GmbH ") == "acme gmbh"
    assert canonicalize_company_name("Müller & Söhne, AG") == "müller & söhne, ag"
    with pytest.raises(ValueError, match="blank"):
        canonicalize_company_name("   ")


def test_builds_expected_node_types_relations_and_reverse_edges(instruments):
    result = _build(instruments)
    graph = result.graph

    assert graph.node_types == ["instrument", "company"]
    assert set(graph.edge_types) == {
        ("instrument", "sold_by", "company"),
        ("company", "sells", "instrument"),
        ("instrument", "owed_by", "company"),
        ("company", "owes", "instrument"),
    }
    assert graph["instrument"].num_nodes == 4
    assert graph["company"].num_nodes == 6
    assert graph["instrument", "sold_by", "company"].num_edges == 4
    assert graph["instrument", "owed_by", "company"].num_edges == 4
    torch.testing.assert_close(
        graph["company", "sells", "instrument"].edge_index,
        graph["instrument", "sold_by", "company"].edge_index.flip(0),
    )
    torch.testing.assert_close(
        graph["company", "owes", "instrument"].edge_index,
        graph["instrument", "owed_by", "company"].edge_index.flip(0),
    )


def test_sorts_instruments_and_unifies_a_hybrid_by_canonical_name(instruments):
    result = _build(instruments)
    metadata = result.metadata
    graph = result.graph

    assert metadata.instrument_uids == ("pre-1", "pre-2", "post-1", "post-2")
    acme_index = metadata.company_index["acme gmbh"]
    pre_1_index = metadata.instrument_index["pre-1"]
    pre_2_index = metadata.instrument_index["pre-2"]

    sold_by = graph["instrument", "sold_by", "company"].edge_index
    owed_by = graph["instrument", "owed_by", "company"].edge_index
    assert sold_by[1, pre_1_index].item() == acme_index
    assert owed_by[1, pre_2_index].item() == acme_index


def test_features_are_fitted_on_pre_cutoff_rows_and_unknowns_are_explicit(instruments):
    result = _build(instruments)
    graph = result.graph
    metadata = result.metadata

    assert graph["instrument"].x.shape[1] == len(metadata.instrument_feature_names)
    assert graph["company"].x.shape[1] == len(metadata.company_feature_names)
    assert metadata.instrument_feature_names[-2:] == (
        "factoring_type=Full Service",
        "factoring_type=__unknown__",
    )

    post_2 = metadata.instrument_index["post-2"]
    currency_unknown = metadata.instrument_feature_names.index("currency=__unknown__")
    factoring_unknown = metadata.instrument_feature_names.index(
        "factoring_type=__unknown__"
    )
    assert graph["instrument"].x[post_2, currency_unknown].item() == 1.0
    assert graph["instrument"].x[post_2, factoring_unknown].item() == 1.0
    assert graph["instrument"].pre_cutoff_mask.tolist() == [True, True, False, False]


def test_post_cutoff_values_do_not_change_company_history_features(instruments):
    original = _build(instruments)
    changed = instruments.copy()
    changed.loc[changed["uid"] == "post-2", "invoice_amount"] = 10.0
    changed.loc[changed["uid"] == "post-2", "purchase_amount"] = 1.0
    rebuilt = _build(changed)

    torch.testing.assert_close(original.graph["company"].x, rebuilt.graph["company"].x)

    new_company = original.metadata.company_index["new company"]
    assert torch.count_nonzero(original.graph["company"].x[new_company]).item() == 0


def test_labels_and_dates_are_in_instrument_order(instruments):
    result = _build(instruments)
    graph = result.graph

    assert graph["instrument"].y.tolist() == [0, 0, 1, 1]
    assert graph["instrument"].invoice_date.dtype == torch.int64
    assert graph["instrument"].invoice_date.tolist() == sorted(
        graph["instrument"].invoice_date.tolist()
    )


def test_build_from_parquet_matches_in_memory_build(instruments, tmp_path):
    path = tmp_path / "instruments.parquet"
    instruments.to_parquet(path)

    direct = _build(instruments)
    loaded = build_trade_finance_graph_from_parquet(
        path, GraphBuildConfig(cutoff="2020-02-01")
    )

    assert loaded.metadata == direct.metadata
    torch.testing.assert_close(
        loaded.graph["instrument"].x, direct.graph["instrument"].x
    )
    torch.testing.assert_close(loaded.graph["company"].x, direct.graph["company"].x)


def test_accepts_legacy_table_with_uid_as_index_and_column(instruments):
    legacy_shaped = instruments.set_index("uid", drop=False)

    result = _build(legacy_shaped)

    assert result.metadata.instrument_uids == ("pre-1", "pre-2", "post-1", "post-2")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda frame: frame.drop(columns="currency"), "Missing required columns"),
        (
            lambda frame: frame.assign(uid=["same"] * len(frame)),
            "Instrument UIDs must be unique",
        ),
        (
            lambda frame: frame.assign(customer_name_1=None),
            "Company names cannot be missing",
        ),
        (
            lambda frame: frame.assign(has_impairment1=2),
            "must contain binary labels",
        ),
    ],
)
def test_rejects_invalid_inputs(instruments, change, message):
    with pytest.raises(ValueError, match=message):
        _build(change(instruments))


def test_requires_history_before_the_cutoff(instruments):
    with pytest.raises(ValueError, match="precede the cutoff"):
        build_trade_finance_graph(instruments, GraphBuildConfig(cutoff="2019-01-01"))

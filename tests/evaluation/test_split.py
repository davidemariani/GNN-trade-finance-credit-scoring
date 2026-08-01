from __future__ import annotations

import pandas as pd
import pytest
import torch

from graph_ml.data import GraphBuildConfig, build_trade_finance_graph
from graph_ml.evaluation import (
    TemporalSplitConfig,
    build_temporal_evaluation_split,
    build_temporal_graph_views,
)


@pytest.fixture
def instruments() -> pd.DataFrame:
    invoice_dates = pd.to_datetime(
        [
            "2020-04-01",
            "2020-02-20",
            "2020-01-20",
            "2020-02-10",
            "2020-01-01",
            "2020-03-01",
            "2020-01-10",
        ]
    )
    return pd.DataFrame(
        {
            "uid": [
                "future",
                "post-cold",
                "pre-censored",
                "post-seen",
                "pre-closed",
                "post-censored",
                "pre-positive-open",
            ],
            "customer_name_1": ["A", "New Co", "D", "A", "A", "B", "C"],
            "debtor_name_1": ["B", "A", "E", "B", "B", "New Two", "A"],
            "invoice_date": invoice_dates,
            "due_date": invoice_dates + pd.Timedelta(days=30),
            "input_date": invoice_dates,
            "invoice_amount": [700.0, 500.0, 300.0, 400.0, 100.0, 600.0, 200.0],
            "purchase_amount": [630.0, 450.0, 270.0, 360.0, 90.0, 540.0, 180.0],
            "currency": ["EUR"] * 7,
            "factoring_type": ["Full Service"] * 7,
            "has_impairment1": [False, True, False, False, False, False, True],
            "is_open": [False, True, True, False, False, True, True],
        }
    )


@pytest.fixture
def graph_result(instruments):
    return build_trade_finance_graph(instruments, GraphBuildConfig(cutoff="2020-02-01"))


@pytest.fixture
def split(instruments, graph_result):
    return build_temporal_evaluation_split(
        instruments,
        graph_result,
        TemporalSplitConfig(analysis_date="2020-03-31"),
    )


def test_split_aligns_unsorted_table_and_applies_target_aware_maturity(
    graph_result, split
):
    index = graph_result.metadata.instrument_index

    assert split.train_mask[index["pre-closed"]]
    assert split.train_mask[index["pre-positive-open"]]
    assert split.censored_mask[index["pre-censored"]]
    assert split.test_mask[index["post-seen"]]
    assert split.test_mask[index["post-cold"]]
    assert split.censored_mask[index["post-censored"]]
    assert not split.observed_mask[index["future"]]
    assert not split.mature_mask[index["future"]]


def test_seen_and_cold_start_masks_partition_the_mature_test_cohort(
    graph_result, split
):
    index = graph_result.metadata.instrument_index

    assert split.seen_test_mask[index["post-seen"]]
    assert split.cold_start_test_mask[index["post-cold"]]
    assert torch.equal(
        split.seen_test_mask | split.cold_start_test_mask, split.test_mask
    )
    assert not torch.any(split.seen_test_mask & split.cold_start_test_mask)


def test_summary_reports_the_cohort_definition(split):
    assert split.summary() == {
        "cutoff": "2020-02-01",
        "analysis_date": "2020-03-31",
        "pre_cutoff_instruments": 3,
        "post_cutoff_instruments": 3,
        "mature_train_instruments": 2,
        "mature_test_instruments": 2,
        "censored_open_negatives": 2,
        "seen_test_instruments": 1,
        "cold_start_test_instruments": 1,
        "cold_start_test_rate": 0.5,
    }


def test_graph_views_block_post_cutoff_messages_into_companies(graph_result, split):
    views = build_temporal_graph_views(graph_result.graph, split)

    for relation in ("sold_by", "owed_by"):
        edge_type = ("instrument", relation, "company")
        assert views.training[edge_type].num_edges == 3
        assert views.inference[edge_type].num_edges == 3
        source_nodes = views.inference[edge_type].edge_index[0]
        assert torch.all(split.pre_cutoff_mask[source_nodes])

    for relation in ("sells", "owes"):
        edge_type = ("company", relation, "instrument")
        assert views.training[edge_type].num_edges == 3
        assert views.inference[edge_type].num_edges == 6
        destination_nodes = views.inference[edge_type].edge_index[1]
        assert torch.all(split.observed_mask[destination_nodes])

    assert views.training["instrument"].num_nodes == 3
    assert views.inference["instrument"].num_nodes == 6
    assert views.training_instrument_indices.tolist() == [0, 1, 2]
    assert views.inference_instrument_indices.tolist() == [0, 1, 2, 3, 4, 5]
    assert views.training_supervision_mask.tolist() == [True, True, False]
    assert views.inference_test_mask.sum().item() == 2
    assert views.inference_seen_test_mask.sum().item() == 1
    assert views.inference_cold_start_test_mask.sum().item() == 1


def test_rejects_uid_mismatch_and_invalid_analysis_date(instruments, graph_result):
    with pytest.raises(ValueError, match="UID mismatch"):
        build_temporal_evaluation_split(
            instruments.iloc[:-1],
            graph_result,
            TemporalSplitConfig(analysis_date="2020-03-31"),
        )

    with pytest.raises(ValueError, match="later than"):
        build_temporal_evaluation_split(
            instruments,
            graph_result,
            TemporalSplitConfig(analysis_date="2020-02-01"),
        )

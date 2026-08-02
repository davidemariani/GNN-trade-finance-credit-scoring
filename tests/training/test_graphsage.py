from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graph_ml.data import GraphBuildConfig, build_trade_finance_graph
from graph_ml.evaluation import TemporalSplitConfig, build_temporal_evaluation_split
from graph_ml.training import (
    GraphSAGETrainingConfig,
    evaluate_graphsage_run,
    fit_hetero_graphsage,
)


@pytest.fixture
def instruments() -> pd.DataFrame:
    count = 48
    dates = pd.date_range("2020-01-01", periods=count, freq="D")
    labels = np.array([index % 4 == 0 for index in range(count)])
    buyers = [f"Buyer {index % 4}" for index in range(count)]
    for index in range(44, count):
        buyers[index] = f"New Buyer {index}"
    return pd.DataFrame(
        {
            "uid": [f"instrument-{index:02d}" for index in range(count)],
            "customer_name_1": [f"Seller {index % 3}" for index in range(count)],
            "debtor_name_1": buyers,
            "invoice_date": dates,
            "due_date": dates + pd.Timedelta(days=30),
            "input_date": dates,
            "invoice_amount": np.where(labels, 1_000.0, 100.0) + np.arange(count),
            "purchase_amount": np.where(labels, 900.0, 80.0) + np.arange(count),
            "currency": np.where(np.arange(count) % 2, "EUR", "CHF"),
            "factoring_type": ["Full Service"] * count,
            "has_impairment1": labels,
            "is_open": [False] * count,
        }
    )


def _setup(instruments: pd.DataFrame):
    graph_result = build_trade_finance_graph(
        instruments, GraphBuildConfig(cutoff="2020-02-10")
    )
    split = build_temporal_evaluation_split(
        instruments,
        graph_result,
        TemporalSplitConfig(analysis_date="2020-02-20"),
    )
    return graph_result, split


def _config() -> GraphSAGETrainingConfig:
    return GraphSAGETrainingConfig(
        hidden_channels=8,
        dropout=0,
        learning_rate=0.01,
        max_epochs=6,
        patience=2,
        minimum_improvement=0,
    )


def test_temporal_training_scores_all_test_cohorts(instruments):
    graph_result, split = _setup(instruments)
    run = fit_hetero_graphsage(graph_result, split, _config())
    report = evaluate_graphsage_run(
        run, graph_result.graph["instrument"].y.numpy(), split, review_fraction=0.2
    )

    assert 1 <= run.best_epoch <= 6
    assert run.validation_start_date == pd.Timestamp("2020-02-02")
    assert run.parameter_count > 0
    assert report.shape[0] == 3
    assert set(report["cohort"]) == {
        "test_all",
        "test_seen",
        "test_cold_start",
    }
    assert np.isfinite(run.scores[split.test_mask.numpy()]).all()
    assert run.instrument_embeddings.shape == (48, 8)


def test_post_cutoff_labels_cannot_change_scores(instruments):
    original_graph, original_split = _setup(instruments)
    original = fit_hetero_graphsage(original_graph, original_split, _config())

    changed = instruments.copy()
    post_cutoff = changed["invoice_date"] >= "2020-02-10"
    changed.loc[post_cutoff, "has_impairment1"] = ~changed.loc[
        post_cutoff, "has_impairment1"
    ]
    changed_graph, changed_split = _setup(changed)
    changed_run = fit_hetero_graphsage(changed_graph, changed_split, _config())

    np.testing.assert_allclose(original.scores, changed_run.scores, equal_nan=True)


def test_rejects_invalid_training_and_evaluation_settings(instruments):
    graph_result, split = _setup(instruments)
    with pytest.raises(ValueError, match="validation_fraction"):
        fit_hetero_graphsage(
            graph_result,
            split,
            GraphSAGETrainingConfig(validation_fraction=0.5),
        )
    run = fit_hetero_graphsage(graph_result, split, _config())
    with pytest.raises(ValueError, match="review_fraction"):
        evaluate_graphsage_run(
            run, graph_result.graph["instrument"].y.numpy(), split, review_fraction=0
        )

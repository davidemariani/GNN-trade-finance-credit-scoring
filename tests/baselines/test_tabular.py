from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graph_ml.baselines import (
    BaselineConfig,
    assemble_tabular_features,
    evaluate_baseline_run,
    fit_tabular_baselines,
)
from graph_ml.data import GraphBuildConfig, build_trade_finance_graph
from graph_ml.evaluation import (
    TemporalSplitConfig,
    build_temporal_evaluation_split,
)


@pytest.fixture
def instruments() -> pd.DataFrame:
    count = 60
    dates = pd.date_range("2020-01-01", periods=count, freq="D")
    labels = np.array([index % 5 == 0 for index in range(count)])
    sellers = [f"Seller {index % 3}" for index in range(count)]
    buyers = [f"Buyer {index % 5}" for index in range(count)]
    for index in range(55, count):
        buyers[index] = f"New Buyer {index}"
    return pd.DataFrame(
        {
            "uid": [f"instrument-{index:02d}" for index in range(count)],
            "customer_name_1": sellers,
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


@pytest.fixture
def setup(instruments):
    graph_result = build_trade_finance_graph(
        instruments, GraphBuildConfig(cutoff="2020-02-20")
    )
    split = build_temporal_evaluation_split(
        instruments,
        graph_result,
        TemporalSplitConfig(analysis_date="2020-02-29"),
    )
    return graph_result, split


def test_assembles_instrument_and_role_specific_company_features(setup):
    graph_result, _ = setup
    features = assemble_tabular_features(graph_result)

    instrument_width = len(graph_result.metadata.instrument_feature_names)
    company_width = len(graph_result.metadata.company_feature_names)
    assert features.instrument_only.shape == (60, instrument_width)
    assert features.instrument_company.shape == (
        60,
        instrument_width + 2 * company_width,
    )
    assert len(features.instrument_feature_names) == instrument_width
    assert features.instrument_company_feature_names[instrument_width].startswith(
        "seller_endpoint__"
    )
    assert features.instrument_company_feature_names[
        instrument_width + company_width
    ].startswith("buyer_endpoint__")
    assert np.isfinite(features.instrument_company).all()


def test_fits_all_baselines_with_temporal_early_stopping(setup):
    graph_result, split = setup
    run = fit_tabular_baselines(
        graph_result,
        split,
        BaselineConfig(max_lightgbm_estimators=20, early_stopping_rounds=3),
    )

    assert set(run.scores) == {
        "base_rate",
        "logistic_instrument",
        "lightgbm_instrument_company",
    }
    assert run.train_prevalence == pytest.approx(0.2)
    assert run.validation_start_date == pd.Timestamp("2020-02-10")
    assert 1 <= run.lightgbm_best_iteration <= 20
    assert len(run.lightgbm_feature_gains) == len(
        assemble_tabular_features(graph_result).instrument_company_feature_names
    )
    for scores in run.scores.values():
        assert scores.shape == (60,)
        assert np.isfinite(scores).all()


def test_test_labels_cannot_change_fitted_scores(instruments):
    config = BaselineConfig(max_lightgbm_estimators=10, early_stopping_rounds=2)

    original_graph = build_trade_finance_graph(
        instruments, GraphBuildConfig(cutoff="2020-02-20")
    )
    original_split = build_temporal_evaluation_split(
        instruments,
        original_graph,
        TemporalSplitConfig(analysis_date="2020-02-29"),
    )
    original_run = fit_tabular_baselines(original_graph, original_split, config)

    changed = instruments.copy()
    changed.loc[
        changed["invoice_date"] >= "2020-02-20", "has_impairment1"
    ] = ~changed.loc[changed["invoice_date"] >= "2020-02-20", "has_impairment1"]
    changed_graph = build_trade_finance_graph(
        changed, GraphBuildConfig(cutoff="2020-02-20")
    )
    changed_split = build_temporal_evaluation_split(
        changed,
        changed_graph,
        TemporalSplitConfig(analysis_date="2020-02-29"),
    )
    changed_run = fit_tabular_baselines(changed_graph, changed_split, config)

    for model_name in original_run.scores:
        np.testing.assert_allclose(
            original_run.scores[model_name], changed_run.scores[model_name]
        )


def test_evaluates_every_model_overall_seen_and_cold_start(setup):
    graph_result, split = setup
    run = fit_tabular_baselines(
        graph_result,
        split,
        BaselineConfig(max_lightgbm_estimators=10, early_stopping_rounds=2),
    )
    report = evaluate_baseline_run(
        run,
        graph_result.graph["instrument"].y.numpy(),
        split,
        review_fraction=0.2,
    )

    assert report.shape[0] == 9
    assert set(report["cohort"]) == {
        "test_all",
        "test_seen",
        "test_cold_start",
    }
    assert (report["sample_count"] > 0).all()
    assert (report["top_k"] >= 1).all()


def test_rejects_invalid_baseline_settings(setup):
    graph_result, split = setup
    with pytest.raises(ValueError, match="validation_fraction"):
        fit_tabular_baselines(
            graph_result, split, BaselineConfig(validation_fraction=0.5)
        )
    with pytest.raises(ValueError, match="review_fraction"):
        evaluate_baseline_run(
            fit_tabular_baselines(
                graph_result,
                split,
                BaselineConfig(max_lightgbm_estimators=5, early_stopping_rounds=1),
            ),
            graph_result.graph["instrument"].y.numpy(),
            split,
            review_fraction=0,
        )

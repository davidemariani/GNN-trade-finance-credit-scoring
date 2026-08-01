from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from graph_ml.viz import (
    plot_baseline_pr_auc,
    plot_binary_ranking_curves,
    plot_feature_gains,
    plot_graph_schema,
    plot_message_passing_steps,
    plot_temporal_cohorts,
)


def test_graph_teaching_figures_have_expected_panels():
    schema = plot_graph_schema()
    message_passing = plot_message_passing_steps()

    assert schema.axes[0].get_title().startswith("One company type")
    assert len(schema.axes[0].get_legend().get_texts()) == 4
    assert len(message_passing.axes) == 3
    plt.close(schema)
    plt.close(message_passing)


def test_temporal_cohort_figure_uses_summary_counts():
    figure = plot_temporal_cohorts(
        {
            "cutoff": "2018-01-01",
            "mature_train_instruments": 80,
            "censored_open_negatives": 10,
            "seen_test_instruments": 7,
            "cold_start_test_instruments": 3,
        }
    )

    assert [patch.get_height() for patch in figure.axes[0].patches] == [80, 10, 7, 3]
    plt.close(figure)


def test_binary_ranking_curves_and_validation():
    figure = plot_binary_ranking_curves([0, 1, 0, 1], [0.1, 0.9, 0.8, 0.7])
    assert [axis.get_title() for axis in figure.axes] == ["Precision–recall", "ROC"]
    plt.close(figure)

    with pytest.raises(ValueError, match="Both classes"):
        plot_binary_ranking_curves([0, 0], [0.1, 0.2])


def test_baseline_and_feature_importance_figures():
    report = pd.DataFrame(
        {
            "model": ["base_rate", "model", "base_rate", "model"],
            "cohort": ["test_all", "test_all", "test_seen", "test_seen"],
            "pr_auc": [0.1, 0.4, 0.08, 0.35],
            "prevalence": [0.1, 0.1, 0.08, 0.08],
        }
    )
    gains = pd.DataFrame({"feature": ["a", "b", "c"], "gain": [1.0, 3.0, 2.0]})

    baseline = plot_baseline_pr_auc(report)
    importance = plot_feature_gains(gains, top_n=2)
    assert baseline.axes[0].get_title().startswith("PR-AUC")
    assert len(importance.axes[0].patches) == 2
    plt.close(baseline)
    plt.close(importance)

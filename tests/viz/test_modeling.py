from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from graph_ml.evaluation import RollingOriginFoldSpec
from graph_ml.viz import (
    embedding_projection_frame,
    plot_backtest_pr_auc,
    plot_embedding_projection,
    plot_expanding_backtest_windows,
    plot_score_distributions,
    plot_seed_variability,
    plot_temporal_decay_curve,
    plot_temporal_attention_schematic,
    plot_temporal_attention_weights,
    plot_temporal_event_slots,
    plot_temporal_message_schematic,
    plot_training_history,
    seed_metric_summary,
)


def test_training_and_score_figures_return_labeled_axes():
    history = plot_training_history([1.0, 0.8, 0.7], [0.2, 0.4, 0.35], best_epoch=2)
    scores = plot_score_distributions([0, 1, 0, 1], [0.1, 0.8, 0.3, 0.7])

    assert history.axes[0].get_title().startswith("Model selection")
    assert len(history.axes) == 2
    assert scores.axes[0].get_xlim() == (0.0, 1.0)
    plt.close(history)
    plt.close(scores)


def test_embedding_projection_preserves_positives_and_is_reproducible():
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(20, 4))
    labels = np.array([0] * 16 + [1] * 4)

    first = embedding_projection_frame(embeddings, labels, max_points=10, seed=3)
    second = embedding_projection_frame(embeddings, labels, max_points=10, seed=3)
    figure = plot_embedding_projection(first)

    assert first.shape == (10, 3)
    assert int(first["label"].sum()) == 4
    np.testing.assert_allclose(first, second)
    assert figure.axes[0].get_title().startswith("PCA")
    plt.close(figure)


def test_modeling_visuals_reject_invalid_inputs():
    with pytest.raises(ValueError, match="best_epoch"):
        plot_training_history([1.0], [0.2], best_epoch=2)
    with pytest.raises(ValueError, match="rank-2"):
        embedding_projection_frame([1, 2], [0, 1])


def test_seed_variability_summary_and_figure():
    metrics = pd.DataFrame(
        {
            "seed": [1, 1, 2, 2],
            "cohort": ["test_all", "test_seen", "test_all", "test_seen"],
            "pr_auc": [0.2, 0.3, 0.4, 0.5],
        }
    )
    summary = seed_metric_summary(metrics)
    figure = plot_seed_variability(
        metrics,
        benchmark_pr_auc={"test_all": 0.6, "test_seen": 0.55},
    )

    assert summary.loc[summary["cohort"] == "test_all", "mean"].item() == pytest.approx(
        0.3
    )
    assert figure.axes[0].get_title().startswith("GraphSAGE varies")
    plt.close(figure)


def test_temporal_teaching_figures_explain_decay_and_relations():
    decay = plot_temporal_decay_curve(half_life_days=10)
    schematic = plot_temporal_message_schematic()

    line = decay.axes[0].lines[0]
    midpoint = np.flatnonzero(np.isclose(line.get_xdata(), 10)).item()
    assert line.get_ydata()[midpoint] == pytest.approx(0.5)
    assert schematic.axes[0].get_title().startswith("One temporal")
    assert len(schematic.axes[0].collections) == 5
    plt.close(decay)
    plt.close(schematic)


def test_temporal_attention_schematic_explains_causal_weighting():
    figure = plot_temporal_attention_schematic()

    assert figure.axes[0].get_title().startswith("Temporal graph attention")
    assert any("time mask" in text.get_text() for text in figure.axes[0].texts)
    plt.close(figure)


def test_temporal_event_slots_show_valid_ages_and_padding():
    figure = plot_temporal_event_slots(
        [[1, 4, 0], [2, 0, 0]],
        [[True, True, False], [True, False, False]],
        ("seller_endpoint__seller_role", "seller_endpoint__buyer_role"),
    )

    assert figure.axes[0].get_title().startswith("Attention-ready")
    assert any(text.get_text() == "padding" for text in figure.axes[0].texts)
    plt.close(figure)


def test_temporal_decay_rejects_nonpositive_half_life():
    with pytest.raises(ValueError, match="positive"):
        plot_temporal_decay_curve(half_life_days=0)


def test_temporal_attention_weights_show_values_ages_and_padding():
    weights = np.array([[0.7, 0.0], [0.3, 0.0]])
    ages = np.array([[2.0, 0.0], [15.0, 0.0]])
    valid = np.array([[True, False], [True, False]])

    figure = plot_temporal_attention_weights(weights, ages, valid, ("seller", "buyer"))

    labels = {text.get_text() for text in figure.axes[0].texts}
    assert {"0.70\n2d", "0.30\n15d", "padding"}.issubset(labels)
    plt.close(figure)


def test_temporal_attention_weights_reject_attention_on_padding():
    with pytest.raises(ValueError, match="padding slots"):
        plot_temporal_attention_weights(
            [[0.8, 0.2]], [[1.0, 0.0]], [[True, False]], ("seller",)
        )


def test_backtest_figures_show_windows_and_model_ranges():
    specs = (
        RollingOriginFoldSpec("2020-01-01", "2021-01-01", "2022-01-01"),
        RollingOriginFoldSpec("2021-01-01", "2022-01-01", "2023-01-01"),
    )
    rows = []
    for fold in (1, 2):
        for model, score in (("tree", 0.2), ("root", 0.15), ("gnn", 0.25)):
            rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "cohort": "test_all",
                    "mean_pr_auc": score,
                    "minimum": score - 0.02,
                    "maximum": score + 0.02,
                    "prevalence": 0.1,
                }
            )
    windows = plot_expanding_backtest_windows(specs, final_holdout_start="2023-01-01")
    comparison = plot_backtest_pr_auc(pd.DataFrame(rows))

    assert windows.axes[0].get_title().startswith("Development windows")
    assert comparison.axes[0].get_title() == "all"
    plt.close(windows)
    plt.close(comparison)


def test_backtest_plot_supports_four_centered_model_families():
    rows = []
    for model in ("lightgbm", "root_only", "temporal_gnn", "transformer"):
        rows.append(
            {
                "fold": 1,
                "model": model,
                "cohort": "test_all",
                "mean_pr_auc": 0.2,
                "minimum": 0.1,
                "maximum": 0.3,
                "prevalence": 0.05,
            }
        )

    figure = plot_backtest_pr_auc(pd.DataFrame(rows))

    centers = sorted(
        patch.get_x() + patch.get_width() / 2 for patch in figure.axes[0].patches
    )
    assert np.mean(centers) == pytest.approx(0.0)
    plt.close(figure)


def test_backtest_window_figure_rejects_holdout_overlap():
    with pytest.raises(ValueError, match="pre-holdout"):
        plot_expanding_backtest_windows(
            (RollingOriginFoldSpec("2020-01-01", "2021-01-01", "2024-01-01"),),
            final_holdout_start="2023-01-01",
        )

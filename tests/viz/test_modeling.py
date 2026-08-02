from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from graph_ml.viz import (
    embedding_projection_frame,
    plot_embedding_projection,
    plot_score_distributions,
    plot_seed_variability,
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

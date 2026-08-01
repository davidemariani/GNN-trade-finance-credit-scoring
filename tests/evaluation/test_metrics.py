import numpy as np
import pytest

from graph_ml.evaluation import compute_binary_metrics


def test_computes_known_rare_event_metrics_and_top_k_operating_point():
    metrics = compute_binary_metrics(
        y_true=[0, 1, 0, 1],
        y_score=[0.1, 0.9, 0.8, 0.7],
        top_k=2,
    )

    assert metrics.sample_count == 4
    assert metrics.positive_count == 2
    assert metrics.prevalence == 0.5
    assert metrics.pr_auc == pytest.approx(5 / 6)
    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.precision_at_k == 0.5
    assert metrics.recall_at_k == 0.5
    assert metrics.as_dict()["top_k"] == 2


def test_reports_undefined_metrics_explicitly_for_single_class_cohort():
    metrics = compute_binary_metrics(
        y_true=[0, 0, 0],
        y_score=[0.3, 0.2, 0.1],
        top_k=1,
    )

    assert metrics.pr_auc is None
    assert metrics.roc_auc is None
    assert metrics.precision_at_k == 0.0
    assert metrics.recall_at_k is None


def test_top_k_ties_follow_stable_input_order():
    metrics = compute_binary_metrics(
        y_true=[1, 0, 1],
        y_score=[0.5, 0.5, 0.1],
        top_k=1,
    )

    assert metrics.precision_at_k == 1.0


@pytest.mark.parametrize(
    ("labels", "scores", "top_k", "message"),
    [
        ([], [], 1, "at least one"),
        ([0, 1], [0.1], 1, "same length"),
        ([0, 2], [0.1, 0.2], 1, "binary labels"),
        ([0, 1], [0.1, np.nan], 1, "finite"),
        ([0, 1], [0.1, 0.2], 0, "between 1"),
        ([0, 1], [0.1, 0.2], 3, "between 1"),
    ],
)
def test_rejects_invalid_metric_inputs(labels, scores, top_k, message):
    with pytest.raises(ValueError, match=message):
        compute_binary_metrics(labels, scores, top_k=top_k)

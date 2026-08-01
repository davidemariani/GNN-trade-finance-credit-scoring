"""Honest binary-classification metrics for rare credit events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class BinaryMetrics:
    """Rare-event metrics plus an explicit top-k review operating point."""

    sample_count: int
    positive_count: int
    prevalence: float
    pr_auc: float | None
    roc_auc: float | None
    top_k: int
    precision_at_k: float
    recall_at_k: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        """Return a serialization-friendly metrics row."""

        return {
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "prevalence": self.prevalence,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "top_k": self.top_k,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
        }


def compute_binary_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    top_k: int,
) -> BinaryMetrics:
    """Compute PR-AUC, ROC AUC, and precision/recall among top-k scores.

    ``y_score`` may be any finite ranking score; it need not be calibrated.
    PR-AUC and recall are returned as ``None`` when there are no positives, and
    ROC AUC is ``None`` unless both classes are present. Ties at the top-k
    boundary follow stable input order, making the operating point deterministic.
    """

    labels = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if labels.size == 0:
        raise ValueError("Metrics require at least one sample")
    if labels.shape != scores.shape:
        raise ValueError("y_true and y_score must have the same length")
    if not np.isfinite(scores).all():
        raise ValueError("y_score must contain only finite values")
    if not np.isin(labels, [0, 1, False, True]).all():
        raise ValueError("y_true must contain binary labels")
    if not 1 <= top_k <= labels.size:
        raise ValueError("top_k must be between 1 and the number of samples")

    labels = labels.astype(np.int64)
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    ranked_indices = np.argsort(-scores, kind="stable")
    selected = labels[ranked_indices[:top_k]]
    selected_positives = int(selected.sum())

    pr_auc = (
        float(average_precision_score(labels, scores)) if positive_count > 0 else None
    )
    roc_auc = (
        float(roc_auc_score(labels, scores))
        if positive_count > 0 and negative_count > 0
        else None
    )
    return BinaryMetrics(
        sample_count=int(labels.size),
        positive_count=positive_count,
        prevalence=positive_count / labels.size,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        top_k=top_k,
        precision_at_k=selected_positives / top_k,
        recall_at_k=(
            selected_positives / positive_count if positive_count > 0 else None
        ),
    )

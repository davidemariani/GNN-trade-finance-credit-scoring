"""Visual diagnostics for graph-model training and learned representations."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.decomposition import PCA


def plot_training_history(
    train_losses: ArrayLike,
    validation_pr_aucs: ArrayLike,
    *,
    best_epoch: int,
) -> Figure:
    """Plot optimization loss and temporal-validation PR-AUC by epoch."""

    losses = np.asarray(train_losses, dtype=float).reshape(-1)
    pr_aucs = np.asarray(validation_pr_aucs, dtype=float).reshape(-1)
    if losses.size == 0 or losses.shape != pr_aucs.shape:
        raise ValueError("Training histories must be non-empty and aligned")
    if not np.isfinite(losses).all() or not np.isfinite(pr_aucs).all():
        raise ValueError("Training histories must be finite")
    if not 1 <= best_epoch <= losses.size:
        raise ValueError("best_epoch must refer to an observed epoch")

    epochs = np.arange(1, losses.size + 1)
    figure, loss_axis = plt.subplots(figsize=(9, 4.5))
    validation_axis = loss_axis.twinx()
    loss_axis.plot(epochs, losses, color="#4C72B0", label="Fit loss")
    validation_axis.plot(
        epochs,
        pr_aucs,
        color="#C44E52",
        label="Validation PR-AUC",
    )
    loss_axis.axvline(
        best_epoch,
        color="#333333",
        linestyle="--",
        label=f"Selected epoch {best_epoch}",
    )
    loss_axis.set(xlabel="Epoch", ylabel="Class-weighted fit loss")
    validation_axis.set(ylabel="Temporal-validation PR-AUC", ylim=(0, 1))
    lines = loss_axis.lines + validation_axis.lines
    loss_axis.legend(lines, [line.get_label() for line in lines], frameon=False)
    loss_axis.set_title("Model selection uses pre-cutoff validation only")
    loss_axis.spines["top"].set_visible(False)
    validation_axis.spines["top"].set_visible(False)
    figure.tight_layout()
    return figure


def plot_score_distributions(y_true: ArrayLike, y_score: ArrayLike) -> Figure:
    """Compare predicted-score densities for negative and positive labels."""

    labels, scores = _validated_binary_arrays(y_true, y_score)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 31)
    axis.hist(
        [scores[labels == 0], scores[labels == 1]],
        bins=bins,
        density=True,
        alpha=0.65,
        label=["Not impaired", "Impaired"],
        color=["#4C72B0", "#C44E52"],
    )
    axis.set(
        title="GraphSAGE score distributions on the mature test cohort",
        xlabel="Predicted impairment score",
        ylabel="Density within class",
        xlim=(0, 1),
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def embedding_projection_frame(
    embeddings: ArrayLike,
    labels: ArrayLike,
    *,
    max_points: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a deterministic, positive-preserving 2D PCA sample of embeddings."""

    matrix = np.asarray(embeddings, dtype=float)
    targets = np.asarray(labels).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != targets.size or matrix.shape[1] < 2:
        raise ValueError("Embeddings must be a rank-2 matrix aligned with labels")
    if matrix.shape[0] < 2 or not np.isfinite(matrix).all():
        raise ValueError("At least two finite embeddings are required")
    if not np.isin(targets, [0, 1, False, True]).all():
        raise ValueError("labels must be binary")
    if max_points < 2:
        raise ValueError("max_points must be at least two")

    targets = targets.astype(int)
    positives = np.flatnonzero(targets == 1)
    negatives = np.flatnonzero(targets == 0)
    if matrix.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        positive_count = min(positives.size, max_points // 2)
        selected_positives = rng.choice(positives, positive_count, replace=False)
        negative_count = max_points - positive_count
        selected_negatives = rng.choice(negatives, negative_count, replace=False)
        selected = np.sort(np.concatenate((selected_positives, selected_negatives)))
    else:
        selected = np.arange(matrix.shape[0])
    coordinates = PCA(n_components=2).fit_transform(matrix[selected])
    return pd.DataFrame(
        {
            "component_1": coordinates[:, 0],
            "component_2": coordinates[:, 1],
            "label": targets[selected],
        }
    )


def plot_embedding_projection(projection: pd.DataFrame) -> Figure:
    """Plot an anonymous two-dimensional learned-embedding projection."""

    required = {"component_1", "component_2", "label"}
    if not required.issubset(projection.columns):
        raise ValueError("Projection is missing required columns")
    figure, axis = plt.subplots(figsize=(7, 5.5))
    for label, color, name in (
        (0, "#4C72B0", "Not impaired"),
        (1, "#C44E52", "Impaired"),
    ):
        rows = projection.loc[projection["label"] == label]
        axis.scatter(
            rows["component_1"],
            rows["component_2"],
            s=12,
            alpha=0.45,
            color=color,
            label=name,
        )
    axis.set(
        title="PCA view of learned test-instrument embeddings",
        xlabel="First principal component",
        ylabel="Second principal component",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def seed_metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize PR-AUC variation across independently initialized runs."""

    required = {"seed", "cohort", "pr_auc"}
    if not required.issubset(metrics.columns):
        raise ValueError("Metrics are missing seed, cohort, or pr_auc")
    if metrics[["seed", "cohort"]].duplicated().any():
        raise ValueError("Each seed/cohort pair must be unique")
    return (
        metrics.groupby("cohort", sort=False)["pr_auc"]
        .agg(
            run_count="count",
            mean="mean",
            sample_std="std",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )


def plot_seed_variability(
    metrics: pd.DataFrame,
    *,
    benchmark_pr_auc: dict[str, float] | None = None,
) -> Figure:
    """Plot every seed's cohort PR-AUC and optional fixed benchmark markers."""

    summary = seed_metric_summary(metrics)
    cohorts = summary["cohort"].tolist()
    positions = {cohort: index for index, cohort in enumerate(cohorts)}
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for seed, rows in metrics.groupby("seed", sort=True):
        ordered = rows.set_index("cohort").loc[cohorts]
        axis.plot(
            range(len(cohorts)),
            ordered["pr_auc"],
            marker="o",
            alpha=0.55,
            linewidth=1,
            label=f"seed {seed}",
        )
    axis.scatter(
        range(len(cohorts)),
        summary["mean"],
        marker="D",
        s=75,
        color="#111111",
        label="GraphSAGE mean",
        zorder=5,
    )
    if benchmark_pr_auc is not None:
        missing = set(cohorts) - set(benchmark_pr_auc)
        if missing:
            raise ValueError("Benchmark is missing one or more cohorts")
        axis.scatter(
            [positions[cohort] for cohort in cohorts],
            [benchmark_pr_auc[cohort] for cohort in cohorts],
            marker="*",
            s=150,
            color="#C44E52",
            label="LightGBM",
            zorder=6,
        )
    axis.set_xticks(
        range(len(cohorts)),
        [cohort.replace("test_", "").replace("_", "\n") for cohort in cohorts],
    )
    axis.set(
        title="GraphSAGE varies materially across random initializations",
        ylabel="PR-AUC",
        ylim=(0, 0.55),
    )
    axis.legend(frameon=False, ncols=2, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def _validated_binary_arrays(
    y_true: ArrayLike, y_score: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_score, dtype=float).reshape(-1)
    if labels.size == 0 or labels.shape != scores.shape:
        raise ValueError("Labels and scores must be non-empty and aligned")
    if not np.isin(labels, [0, 1, False, True]).all():
        raise ValueError("Labels must be binary")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    return labels.astype(int), scores

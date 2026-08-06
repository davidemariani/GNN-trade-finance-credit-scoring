"""Visual diagnostics for graph-model training and learned representations."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.decomposition import PCA

from graph_ml.evaluation import RollingOriginFoldSpec


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
    model_label: str = "GraphSAGE",
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
        label=f"{model_label} mean",
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
        title=f"{model_label} varies across random initializations",
        ylabel="PR-AUC",
        ylim=(0, 0.55),
    )
    axis.legend(frameon=False, ncols=2, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def plot_temporal_decay_curve(*, half_life_days: float = 180.0) -> Figure:
    """Show how exponential message weight decreases with event age."""

    if not np.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("half_life_days must be positive and finite")
    ages = np.linspace(0, 4 * half_life_days, 401)
    weights = np.exp(-np.log(2) * ages / half_life_days)
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    axis.plot(ages, weights, color="#4C72B0", linewidth=2.5)
    axis.scatter([half_life_days], [0.5], color="#C44E52", s=65, zorder=3)
    axis.axvline(half_life_days, color="#C44E52", linestyle="--", alpha=0.7)
    axis.axhline(0.5, color="#C44E52", linestyle="--", alpha=0.7)
    axis.annotate(
        f"half-life = {half_life_days:g} days\nweight = 0.5",
        (half_life_days, 0.5),
        xytext=(1.35 * half_life_days, 0.68),
        arrowprops={"arrowstyle": "->", "color": "#C44E52"},
    )
    axis.set(
        title="Recent historical invoices contribute more to each message",
        xlabel="Age of a strictly-prior invoice (days)",
        ylabel="Relative message weight",
        xlim=(0, 4 * half_life_days),
        ylim=(0, 1.03),
    )
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def plot_temporal_message_schematic() -> Figure:
    """Draw the four role-aware history channels used for one current invoice."""

    figure, axis = plt.subplots(figsize=(11, 5.4))
    current = (0.5, 0.48)
    channels = (
        ((0.08, 0.82), "seller endpoint\nas seller", "#4C72B0"),
        ((0.08, 0.20), "seller endpoint\nas buyer", "#55A868"),
        ((0.92, 0.82), "buyer endpoint\nas seller", "#8172B2"),
        ((0.92, 0.20), "buyer endpoint\nas buyer", "#C44E52"),
    )
    axis.scatter(*current, s=1_250, color="#F0A202", edgecolor="white", zorder=4)
    axis.text(*current, "current\ninvoice", ha="center", va="center", weight="bold")
    for (x, y), label, color in channels:
        axis.scatter(x, y, s=1_050, color=color, alpha=0.9, edgecolor="white", zorder=3)
        axis.text(x, y, label, ha="center", va="center", color="white", weight="bold")
        axis.annotate(
            "",
            xy=(0.43 if x < 0.5 else 0.57, 0.48),
            xytext=(x + 0.07 if x < 0.5 else x - 0.07, y),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2},
        )
    axis.text(
        0.5,
        0.06,
        "Each channel aggregates only invoices with event time < prediction time;\n"
        "count, age, and history-presence metadata gate its learned message.",
        ha="center",
        va="center",
        fontsize=10,
    )
    axis.set(title="One temporal message-passing layer preserves endpoint and role")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.tight_layout()
    return figure


def plot_temporal_attention_schematic() -> Figure:
    """Explain causal graph attention from event selection to weighted message."""

    figure, axis = plt.subplots(figsize=(12, 5.2))
    boxes = (
        (0.08, 0.63, "current invoice\nquery Q", "#F0A202"),
        (0.08, 0.25, "strictly-prior role events\nkeys K and values V", "#4C72B0"),
        (0.38, 0.44, "compatibility scores\nQ · K / √d", "#8172B2"),
        (0.62, 0.44, "causal mask + softmax\nattention weights α", "#55A868"),
        (0.88, 0.44, "weighted message\nΣ αV", "#C44E52"),
    )
    for x, y, label, color in boxes:
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            color="white",
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.7",
                "facecolor": color,
                "edgecolor": "white",
            },
        )
    for start, end in (
        ((0.17, 0.63), (0.29, 0.49)),
        ((0.18, 0.25), (0.29, 0.39)),
        ((0.47, 0.44), (0.53, 0.44)),
        ((0.72, 0.44), (0.79, 0.44)),
    ):
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "lw": 2, "color": "#444444"},
        )
    axis.text(
        0.5,
        0.08,
        "Graph/role chooses candidate neighbours  •  time mask removes tⱼ ≥ tᵢ  •  "
        "attention learns importance among the legal past",
        ha="center",
        va="center",
        fontsize=10,
    )
    axis.text(0.62, 0.75, "future or padded event", ha="center", color="#C44E52")
    axis.plot([0.55, 0.69], [0.72, 0.78], color="#C44E52", linewidth=3)
    axis.plot([0.55, 0.69], [0.78, 0.72], color="#C44E52", linewidth=3)
    axis.set(title="Temporal graph attention replaces fixed neighbour averaging")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.tight_layout()
    return figure


def plot_temporal_event_slots(
    age_days: ArrayLike,
    valid_mask: ArrayLike,
    relation_names: tuple[str, ...],
) -> Figure:
    """Inspect one invoice's bounded newest-first temporal event slots."""

    ages = np.asarray(age_days, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if ages.ndim != 2 or ages.shape != valid.shape:
        raise ValueError("age_days and valid_mask must be aligned [relations, slots]")
    if ages.shape[0] != len(relation_names) or ages.shape[1] < 1:
        raise ValueError("relation_names and at least one slot are required")
    if not np.isfinite(ages).all() or (ages[valid] <= 0).any():
        raise ValueError("Valid event ages must be positive and finite")
    figure, axis = plt.subplots(figsize=(10, 4.4))
    axis.imshow(valid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for relation in range(ages.shape[0]):
        for slot in range(ages.shape[1]):
            label = (
                f"{ages[relation, slot]:.0f}d" if valid[relation, slot] else "padding"
            )
            axis.text(
                slot,
                relation,
                label,
                ha="center",
                va="center",
                color="white" if valid[relation, slot] else "#666666",
                fontsize=8,
            )
    axis.set_xticks(
        range(ages.shape[1]), [f"slot {slot + 1}" for slot in range(ages.shape[1])]
    )
    axis.set_yticks(
        range(ages.shape[0]),
        [
            name.replace("_endpoint__", " → ").replace("_role", "")
            for name in relation_names
        ],
    )
    axis.set(
        title="Attention-ready event slots are newest-first within each role",
        xlabel="bounded historical-event position",
    )
    figure.tight_layout()
    return figure


def plot_temporal_attention_weights(
    weights: ArrayLike,
    age_days: ArrayLike,
    valid_mask: ArrayLike,
    relation_names: Sequence[str],
) -> Figure:
    """Show learned attention and age for one invoice's bounded history."""

    attention = np.asarray(weights, dtype=float)
    ages = np.asarray(age_days, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if (
        attention.ndim != 2
        or attention.shape != ages.shape
        or attention.shape != valid.shape
    ):
        raise ValueError("weights, age_days, and valid_mask must be aligned 2D arrays")
    if attention.shape[0] != len(relation_names):
        raise ValueError("relation_names must align with the first tensor dimension")
    if not np.isfinite(attention).all() or (attention < 0).any():
        raise ValueError("attention weights must be finite and non-negative")
    if not np.allclose(attention[~valid], 0):
        raise ValueError("padding slots must have zero attention")

    displayed = np.ma.masked_where(~valid, attention)
    figure, axis = plt.subplots(
        figsize=(max(7.5, 0.85 * attention.shape[1]), 1.15 * attention.shape[0] + 1.8)
    )
    image = axis.imshow(displayed, cmap="YlGnBu", vmin=0, aspect="auto")
    for relation, slot in np.ndindex(attention.shape):
        label = (
            f"{attention[relation, slot]:.2f}\n{ages[relation, slot]:.0f}d"
            if valid[relation, slot]
            else "padding"
        )
        axis.text(
            slot,
            relation,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="white"
            if valid[relation, slot] and attention[relation, slot] > 0.35
            else "#333333",
        )
    axis.set_xticks(
        range(attention.shape[1]),
        [f"slot {slot + 1}" for slot in range(attention.shape[1])],
    )
    axis.set_yticks(
        range(attention.shape[0]),
        [name.replace("__", " → ") for name in relation_names],
    )
    axis.set(
        title="Where one fitted query places attention",
        xlabel="newest-first event slot (cell: weight / age)",
    )
    figure.colorbar(image, ax=axis, label="attention weight")
    figure.tight_layout()
    return figure


def plot_expanding_backtest_windows(
    specs: tuple[RollingOriginFoldSpec, ...],
    *,
    final_holdout_start: str | pd.Timestamp,
) -> Figure:
    """Draw train, validation, test, and sealed-holdout calendar regions."""

    if not specs:
        raise ValueError("specs must not be empty")
    holdout = pd.Timestamp(final_holdout_start)
    boundaries = [
        tuple(
            pd.Timestamp(value)
            for value in (spec.train_end, spec.validation_end, spec.test_end)
        )
        for spec in specs
    ]
    if any(
        not train < validation < test <= holdout
        for train, validation, test in boundaries
    ):
        raise ValueError("Fold boundaries must be chronological and pre-holdout")
    start = min(train for train, _, _ in boundaries) - pd.DateOffset(years=2)
    figure, axis = plt.subplots(figsize=(11, 1.5 + 1.15 * len(specs)))
    colors = {"train": "#4C72B0", "validation": "#DD8452", "test": "#55A868"}
    for row, (train_end, validation_end, test_end) in enumerate(boundaries):
        axis.barh(
            row,
            train_end - start,
            left=start,
            color=colors["train"],
            label="expanding train" if row == 0 else None,
        )
        axis.barh(
            row,
            validation_end - train_end,
            left=train_end,
            color=colors["validation"],
            label="validation" if row == 0 else None,
        )
        axis.barh(
            row,
            test_end - validation_end,
            left=validation_end,
            color=colors["test"],
            label="development test" if row == 0 else None,
        )
    axis.axvspan(
        holdout,
        holdout + pd.DateOffset(months=8),
        color="#C44E52",
        alpha=0.25,
        label="sealed final holdout",
    )
    axis.set_yticks(
        range(len(specs)), [f"fold {index}" for index in range(1, len(specs) + 1)]
    )
    axis.set(
        title="Development windows end before the sealed final holdout",
        xlabel="calendar time",
    )
    axis.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    axis.spines[["top", "right", "left"]].set_visible(False)
    figure.tight_layout()
    return figure


def plot_backtest_pr_auc(summary: pd.DataFrame) -> Figure:
    """Compare fold-level model PR-AUC with prevalence and neural ranges."""

    required = {
        "fold",
        "model",
        "cohort",
        "mean_pr_auc",
        "minimum",
        "maximum",
        "prevalence",
    }
    if not required.issubset(summary.columns) or summary.empty:
        raise ValueError("summary is empty or missing backtest columns")
    cohorts = summary["cohort"].drop_duplicates().tolist()
    models = summary["model"].drop_duplicates().tolist()
    folds = sorted(summary["fold"].unique())
    figure, axes = plt.subplots(
        1, len(cohorts), figsize=(5 * len(cohorts), 4.6), sharey=True
    )
    axes = np.atleast_1d(axes)
    width = min(0.8 / len(models), 0.24)
    colors = ("#4C72B0", "#8172B2", "#C44E52", "#55A868")
    if len(models) > len(colors):
        raise ValueError("Backtest plot supports at most four model families")
    for axis, cohort in zip(axes, cohorts, strict=True):
        rows = summary.loc[summary["cohort"] == cohort]
        x = np.arange(len(folds), dtype=float)
        for offset, (model, color) in enumerate(zip(models, colors, strict=False)):
            ordered = rows.loc[rows["model"] == model].set_index("fold").loc[folds]
            means = ordered["mean_pr_auc"].to_numpy()
            errors = np.vstack((means - ordered["minimum"], ordered["maximum"] - means))
            axis.bar(
                x + (offset - (len(models) - 1) / 2) * width,
                means,
                width,
                color=color,
                label=model.replace("_", " "),
                yerr=errors,
                capsize=3,
            )
        prevalence = (
            rows.drop_duplicates("fold").set_index("fold").loc[folds, "prevalence"]
        )
        axis.plot(x, prevalence, "k--o", label="prevalence")
        axis.set_xticks(x, [f"fold {fold}" for fold in folds])
        axis.set(
            title=cohort.replace("test_", "").replace("_", " "),
            xlabel="development origin",
        )
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("PR-AUC (bars show min–max across seeds)")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("Model ordering and prevalence change materially through time")
    figure.tight_layout()
    return figure


def plot_paired_time_ablation(results: pd.DataFrame) -> Figure:
    """Compare Transformer time encodings with seeds paired within each fold."""

    return _plot_paired_validation_ablation(
        results,
        treatment="time_encoding",
        variants=("none", "fixed_decay", "learned"),
        labels=("no age", "fixed 180d decay", "learned log-age"),
        xlabel="time treatment",
        title="Changing time encoding does not give a stable validation gain",
    )


def plot_paired_fusion_ablation(results: pd.DataFrame) -> Figure:
    """Compare residual and coverage-gated fusion with paired seeds."""

    return _plot_paired_validation_ablation(
        results,
        treatment="fusion",
        variants=("residual", "coverage_gate"),
        labels=("full residual", "coverage gate"),
        xlabel="root/message fusion",
        title="Coverage gating has a large but regime-dependent effect",
    )


def plot_paired_capacity_ablation(results: pd.DataFrame) -> Figure:
    """Compare the capacity/regularization factorial with paired seeds."""

    return _plot_paired_validation_ablation(
        results,
        treatment="variant",
        variants=(
            "wide_current",
            "wide_strong_reg",
            "compact_current",
            "compact_strong_reg",
        ),
        labels=(
            "wide\ncurrent reg",
            "wide\nstrong reg",
            "compact\ncurrent reg",
            "compact\nstrong reg",
        ),
        xlabel="capacity and regularization treatment",
        title="Compact capacity does not improve validation stability",
    )


def plot_paired_k_ablation(results: pd.DataFrame) -> Figure:
    """Compare recent-event information budgets with paired seeds."""

    return _plot_paired_validation_ablation(
        results,
        treatment="max_events",
        variants=(2, 4, 8, 16),
        labels=("K=2", "K=4", "K=8", "K=16"),
        xlabel="maximum recent events per relation",
        title="The useful history budget changes across temporal regimes",
    )


def plot_paired_relation_ablation(results: pd.DataFrame) -> Figure:
    """Compare role-specific and shared temporal GNN transforms."""

    return _plot_paired_validation_ablation(
        results,
        treatment="relation_mode",
        variants=("separate", "shared"),
        labels=("role-specific", "shared transform"),
        xlabel="relation parameterization",
        title="Relation sharing trades fold-1 mean for fold-2 stability",
    )


def plot_paired_decay_ablation(results: pd.DataFrame) -> Figure:
    """Compare temporal GNN recency priors with paired seeds."""

    return _plot_paired_validation_ablation(
        results,
        treatment="decay",
        variants=("short_60d", "current_180d", "long_365d", "no_decay"),
        labels=("60d", "180d", "365d", "no decay"),
        xlabel="history half-life",
        title="The useful recency prior changes across temporal regimes",
    )


def plot_paired_recent_ablation(results: pd.DataFrame) -> Figure:
    """Compare all-history and bounded recent temporal GNN aggregation."""

    return _plot_paired_validation_ablation(
        results,
        treatment="aggregation",
        variants=("all_history", "recent_k8"),
        labels=("all legal history", "newest K=8"),
        xlabel="relation aggregation budget",
        title="Recent K=8 improves typical seeds but changes the extremes",
    )


def _plot_paired_validation_ablation(
    results: pd.DataFrame,
    *,
    treatment: str,
    variants: tuple[object, ...],
    labels: tuple[str, ...],
    xlabel: str,
    title: str,
) -> Figure:
    required = {"fold", "seed", treatment, "validation_pr_auc"}
    if results.empty or not required.issubset(results.columns):
        raise ValueError("results are empty or missing paired-ablation columns")
    unexpected = set(results[treatment]) - set(variants)
    if unexpected:
        raise ValueError(f"Unknown {treatment} values: {sorted(unexpected)}")
    folds = sorted(results["fold"].unique())
    figure, axes = plt.subplots(1, len(folds), figsize=(5.2 * len(folds), 4.6))
    axes = np.atleast_1d(axes)
    x = np.arange(len(variants))
    for axis, fold in zip(axes, folds, strict=True):
        pivot = results.loc[results["fold"] == fold].pivot(
            index="seed", columns=treatment, values="validation_pr_auc"
        )
        if pivot.reindex(columns=variants).isna().any(axis=None):
            raise ValueError(f"Fold {fold} does not contain paired seed results")
        paired = pivot.loc[:, variants]
        for seed, row in paired.iterrows():
            axis.plot(x, row, color="#9E9E9E", marker="o", alpha=0.65, label=None)
            axis.annotate(
                str(seed),
                (x[-1], row.iloc[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=7,
            )
        means = paired.mean(axis=0)
        axis.plot(
            x, means, color="#C44E52", marker="o", linewidth=3, label="five-seed mean"
        )
        axis.set_xticks(x, labels, rotation=12)
        axis.set(
            title=f"validation fold {fold}",
            xlabel=xlabel,
            ylabel="validation PR-AUC",
        )
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(title)
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

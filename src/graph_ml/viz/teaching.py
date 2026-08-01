"""Concept-first figures used by the project studybooks."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


COMPANY_COLOR = "#4C72B0"
INSTRUMENT_COLOR = "#DD8452"
SELLER_COLOR = "#55A868"
BUYER_COLOR = "#8172B2"
FOCAL_COLOR = "#C44E52"


def plot_graph_schema() -> Figure:
    """Show the v1 node and relation types through a small business example."""

    figure, axis = plt.subplots(figsize=(10, 5.2))
    positions = {
        "seller": (0.08, 0.68),
        "invoice_1": (0.38, 0.68),
        "hybrid": (0.68, 0.68),
        "invoice_2": (0.68, 0.22),
        "buyer": (0.94, 0.22),
    }
    labels = {
        "seller": "Company S\n(seller)",
        "invoice_1": "Instrument I₁\n(prediction unit)",
        "hybrid": "Company H\n(buyer + seller)",
        "invoice_2": "Instrument I₂\n(prediction unit)",
        "buyer": "Company B\n(buyer)",
    }
    node_types = {
        "seller": "company",
        "invoice_1": "instrument",
        "hybrid": "company",
        "invoice_2": "instrument",
        "buyer": "company",
    }
    for node, (x, y) in positions.items():
        color = COMPANY_COLOR if node_types[node] == "company" else INSTRUMENT_COLOR
        axis.scatter(
            x, y, s=2_800, color=color, edgecolor="white", linewidth=2, zorder=3
        )
        axis.text(
            x,
            y,
            labels[node],
            ha="center",
            va="center",
            color="white",
            weight="bold",
            fontsize=10,
            zorder=4,
        )

    _annotated_arrow(
        axis, positions["invoice_1"], positions["seller"], "sold_by", SELLER_COLOR, 0.08
    )
    _annotated_arrow(
        axis, positions["invoice_1"], positions["hybrid"], "owed_by", BUYER_COLOR, -0.08
    )
    _annotated_arrow(
        axis, positions["invoice_2"], positions["hybrid"], "sold_by", SELLER_COLOR, 0.08
    )
    _annotated_arrow(
        axis, positions["invoice_2"], positions["buyer"], "owed_by", BUYER_COLOR, -0.08
    )

    axis.legend(
        handles=[
            Patch(
                color=INSTRUMENT_COLOR,
                label="Instrument node: carries label and invoice features",
            ),
            Patch(
                color=COMPANY_COLOR, label="Company node: shared counterparty context"
            ),
            Line2D([0], [0], color=SELLER_COLOR, lw=3, label="sold_by relation"),
            Line2D([0], [0], color=BUYER_COLOR, lw=3, label="owed_by relation"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=9,
    )
    axis.set_title(
        "One company type; business roles live on typed edges", weight="bold"
    )
    axis.set_xlim(-0.06, 1.08)
    axis.set_ylim(-0.02, 1.0)
    axis.set_axis_off()
    figure.tight_layout()
    return figure


def plot_message_passing_steps() -> Figure:
    """Explain a two-hop instrument→company→instrument receptive field."""

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    titles = (
        "0 hops: own information",
        "1 hop: company aggregates",
        "2 hops: related instruments receive context",
    )
    active_sets = (
        {"target"},
        {"target", "company"},
        {"target", "company", "peer_a", "peer_b"},
    )
    positions = {
        "target": (0.15, 0.5),
        "company": (0.5, 0.5),
        "peer_a": (0.85, 0.75),
        "peer_b": (0.85, 0.25),
    }
    labels = {
        "target": "target\ninstrument",
        "company": "shared\ncompany",
        "peer_a": "related\ninstrument",
        "peer_b": "related\ninstrument",
    }
    for axis, title, active in zip(axes, titles, active_sets, strict=True):
        axis.plot([0.15, 0.5, 0.85], [0.5, 0.5, 0.75], color="#BDBDBD", zorder=1)
        axis.plot([0.5, 0.85], [0.5, 0.25], color="#BDBDBD", zorder=1)
        for node, (x, y) in positions.items():
            base = COMPANY_COLOR if node == "company" else INSTRUMENT_COLOR
            color = base if node in active else "#E6E6E6"
            axis.scatter(
                x, y, s=1_250, color=color, edgecolor="white", linewidth=2, zorder=2
            )
            axis.text(
                x,
                y,
                labels[node],
                ha="center",
                va="center",
                fontsize=8,
                color="white" if node in active else "#777777",
                weight="bold",
            )
        axis.set_title(title, fontsize=10, weight="bold")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_axis_off()
    figure.suptitle(
        "A two-layer GNN expands what the target instrument can know", weight="bold"
    )
    figure.tight_layout()
    return figure


def plot_temporal_cohorts(summary: Mapping[str, int | float | str]) -> Figure:
    """Visualize the train/test/censored cohort counts around the cutoff."""

    labels = [
        "Mature train",
        "Censored\n(no supervision)",
        "Seen test",
        "Cold-start test",
    ]
    values = [
        int(summary["mature_train_instruments"]),
        int(summary["censored_open_negatives"]),
        int(summary["seen_test_instruments"]),
        int(summary["cold_start_test_instruments"]),
    ]
    colors = ["#4C72B0", "#BAB0AC", "#55A868", "#C44E52"]
    figure, axis = plt.subplots(figsize=(9, 4.2))
    bars = axis.bar(labels, values, color=colors)
    axis.bar_label(bars, labels=[f"{value:,}" for value in values], padding=3)
    axis.axvline(1.5, color="#333333", linestyle="--", linewidth=1.5)
    ymax = max(values) * 1.18
    axis.text(
        0.5, ymax * 0.96, f"before {summary['cutoff']}", ha="center", weight="bold"
    )
    axis.text(
        2.5, ymax * 0.96, f"after {summary['cutoff']}", ha="center", weight="bold"
    )
    axis.set(
        title="Only mature labels enter evaluation",
        ylabel="Instruments",
        ylim=(0, ymax),
    )
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def plot_binary_ranking_curves(y_true: ArrayLike, y_score: ArrayLike) -> Figure:
    """Plot precision–recall and ROC curves with their no-skill references."""

    labels = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_score, dtype=float).reshape(-1)
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError("y_true and y_score must be non-empty and aligned")
    if not np.isin(labels, [0, 1, False, True]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores finite")
    if np.unique(labels).size != 2:
        raise ValueError("Both classes are required to draw both curves")

    precision, recall, _ = precision_recall_curve(labels, scores)
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    prevalence = float(labels.mean())
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(recall, precision, color=FOCAL_COLOR, linewidth=2.5)
    axes[0].axhline(
        prevalence,
        color="#777777",
        linestyle="--",
        label=f"No skill = prevalence ({prevalence:.0%})",
    )
    axes[0].set(
        title="Precision–recall",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.03),
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(
        false_positive_rate, true_positive_rate, color=COMPANY_COLOR, linewidth=2.5
    )
    axes[1].plot(
        [0, 1], [0, 1], color="#777777", linestyle="--", label="Random ranking"
    )
    axes[1].set(
        title="ROC",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        xlim=(0, 1),
        ylim=(0, 1.03),
    )
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "The same ranking viewed through two different questions", weight="bold"
    )
    figure.tight_layout()
    return figure


def plot_baseline_pr_auc(report: pd.DataFrame) -> Figure:
    """Compare model PR-AUC with each evaluation cohort's prevalence."""

    required = {"model", "cohort", "pr_auc", "prevalence"}
    missing = required - set(report.columns)
    if missing:
        raise ValueError(f"Report is missing columns: {', '.join(sorted(missing))}")
    cohorts = list(dict.fromkeys(report["cohort"]))
    models = list(dict.fromkeys(report["model"]))
    positions = np.arange(len(cohorts))
    width = 0.8 / len(models)
    figure, axis = plt.subplots(figsize=(10, 4.8))
    for index, model in enumerate(models):
        rows = report.loc[report["model"] == model].set_index("cohort").loc[cohorts]
        offset = (index - (len(models) - 1) / 2) * width
        axis.bar(
            positions + offset,
            rows["pr_auc"],
            width=width,
            label=model.replace("_", " "),
        )
    prevalence = report.groupby("cohort", sort=False)["prevalence"].first().loc[cohorts]
    axis.scatter(
        positions,
        prevalence,
        marker="_",
        s=650,
        linewidth=3,
        color="#222222",
        label="No-skill prevalence",
        zorder=4,
    )
    axis.set_xticks(
        positions,
        [cohort.replace("test_", "").replace("_", "\n") for cohort in cohorts],
    )
    axis.set(
        title="PR-AUC by model and deployment cohort", ylabel="PR-AUC", ylim=(0, 1)
    )
    axis.legend(frameon=False, ncols=2, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def plot_feature_gains(gains: pd.DataFrame, *, top_n: int = 10) -> Figure:
    """Plot the strongest model gain importances as a horizontal ranking."""

    if not {"feature", "gain"}.issubset(gains.columns):
        raise ValueError("gains must contain feature and gain columns")
    if top_n < 1:
        raise ValueError("top_n must be positive")
    selected = gains.nlargest(top_n, "gain").sort_values("gain")
    labels = [
        label.replace("_endpoint__", ": ").replace("_", " ")
        for label in selected["feature"]
    ]
    figure, axis = plt.subplots(figsize=(9, max(4, 0.42 * len(selected))))
    axis.barh(labels, selected["gain"], color=COMPANY_COLOR)
    axis.set(
        title=f"Top {len(selected)} LightGBM gain importances",
        xlabel="Total split gain",
    )
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def _annotated_arrow(axis, start, end, label: str, color: str, curve: float) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.2,
        color=color,
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=38,
        shrinkB=38,
        zorder=2,
    )
    axis.add_patch(arrow)
    midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + curve)
    axis.text(
        *midpoint,
        label,
        color=color,
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )

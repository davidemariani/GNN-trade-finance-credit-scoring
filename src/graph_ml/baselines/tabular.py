"""Classical tabular baselines evaluated on the shared temporal cohort."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import HeteroData

from graph_ml.data import GraphBuildResult
from graph_ml.evaluation import (
    TemporalEvaluationSplit,
    compute_binary_metrics,
)

if TYPE_CHECKING:
    import lightgbm as lgb


@dataclass(frozen=True)
class BaselineConfig:
    """Reproducible settings for the v1 classical baselines."""

    seed: int = 42
    validation_fraction: float = 0.2
    max_lightgbm_estimators: int = 1_000
    early_stopping_rounds: int = 50


@dataclass(frozen=True)
class TabularFeatureSet:
    """Instrument-only and instrument-plus-company feature matrices."""

    instrument_only: np.ndarray
    instrument_company: np.ndarray
    instrument_feature_names: tuple[str, ...]
    instrument_company_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class BaselineRun:
    """Out-of-cohort scores and fit metadata from all baseline models."""

    scores: dict[str, np.ndarray]
    train_prevalence: float
    validation_start_date: pd.Timestamp
    lightgbm_best_iteration: int
    lightgbm_feature_gains: tuple[tuple[str, float], ...]
    seed: int


def assemble_tabular_features(graph_result: GraphBuildResult) -> TabularFeatureSet:
    """Create fair tabular inputs from the same tensors available to the GNN.

    Logistic regression receives instrument features only. LightGBM receives
    instrument features plus the pre-cutoff history vector of both endpoint
    companies, with seller and buyer columns kept distinct.
    """

    graph = graph_result.graph
    metadata = graph_result.metadata
    graph.validate(raise_on_error=True)
    instrument = graph["instrument"].x.detach().cpu().numpy().astype(np.float64)
    company = graph["company"].x.detach().cpu().numpy().astype(np.float64)
    seller_indices = _company_index_per_instrument(
        graph, ("instrument", "sold_by", "company")
    )
    buyer_indices = _company_index_per_instrument(
        graph, ("instrument", "owed_by", "company")
    )
    combined = np.column_stack(
        (instrument, company[seller_indices], company[buyer_indices])
    )

    names = metadata.instrument_feature_names
    combined_names = (
        names
        + tuple(f"seller_endpoint__{name}" for name in metadata.company_feature_names)
        + tuple(f"buyer_endpoint__{name}" for name in metadata.company_feature_names)
    )
    return TabularFeatureSet(
        instrument_only=instrument,
        instrument_company=combined,
        instrument_feature_names=names,
        instrument_company_feature_names=combined_names,
    )


def fit_tabular_baselines(
    graph_result: GraphBuildResult,
    split: TemporalEvaluationSplit,
    config: BaselineConfig | None = None,
) -> BaselineRun:
    """Fit base-rate, logistic, and temporally validated LightGBM models.

    LightGBM's number of estimators is selected by early stopping on the latest
    portion of the mature training period. It is then refitted on the complete
    mature training cohort with that fixed number of trees. Test labels are
    never supplied to fitting or early stopping.
    """

    # Import after PyTorch graph construction. On macOS, eagerly loading both
    # OpenMP runtimes before large torch operations can cause a native crash.
    import lightgbm as lgb

    config = config or BaselineConfig()
    if not 0 < config.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    if config.max_lightgbm_estimators < 1 or config.early_stopping_rounds < 1:
        raise ValueError(
            "LightGBM estimator and early-stopping counts must be positive"
        )

    features = assemble_tabular_features(graph_result)
    graph = graph_result.graph
    labels = graph["instrument"].y.detach().cpu().numpy().astype(np.int64)
    train_mask = split.train_mask.detach().cpu().numpy()
    if train_mask.shape != labels.shape:
        raise ValueError("train_mask must align with graph instrument labels")
    train_indices = np.flatnonzero(train_mask)
    if train_indices.size < 4 or np.unique(labels[train_indices]).size != 2:
        raise ValueError(
            "Baseline training requires at least four rows and both classes"
        )

    fit_indices, validation_indices, validation_start = _temporal_validation_indices(
        graph["instrument"].invoice_date.detach().cpu().numpy(),
        train_indices,
        config.validation_fraction,
    )
    if np.unique(labels[fit_indices]).size != 2:
        raise ValueError("Temporal LightGBM fit partition must contain both classes")

    train_prevalence = float(labels[train_indices].mean())
    scores: dict[str, np.ndarray] = {
        "base_rate": np.full(labels.shape, train_prevalence, dtype=np.float64)
    }

    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=config.seed,
    )
    logistic.fit(features.instrument_only[train_indices], labels[train_indices])
    scores["logistic_instrument"] = logistic.predict_proba(features.instrument_only)[
        :, 1
    ]

    search_model = _lightgbm_model(config, config.max_lightgbm_estimators)
    search_model.fit(
        features.instrument_company[fit_indices],
        labels[fit_indices],
        eval_X=features.instrument_company[validation_indices],
        eval_y=labels[validation_indices],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = int(search_model.best_iteration_ or config.max_lightgbm_estimators)

    final_model = _lightgbm_model(config, best_iteration)
    final_model.fit(features.instrument_company[train_indices], labels[train_indices])
    scores["lightgbm_instrument_company"] = final_model.predict_proba(
        features.instrument_company
    )[:, 1]
    gains = final_model.booster_.feature_importance(importance_type="gain")
    feature_gains = tuple(
        sorted(
            zip(features.instrument_company_feature_names, gains, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    return BaselineRun(
        scores=scores,
        train_prevalence=train_prevalence,
        validation_start_date=_epoch_day_to_timestamp(validation_start),
        lightgbm_best_iteration=best_iteration,
        lightgbm_feature_gains=feature_gains,
        seed=config.seed,
    )


def evaluate_baseline_run(
    run: BaselineRun,
    labels: np.ndarray,
    split: TemporalEvaluationSplit,
    *,
    review_fraction: float = 0.05,
) -> pd.DataFrame:
    """Evaluate each baseline overall and by seen/cold-start test cohort."""

    if not 0 < review_fraction <= 1:
        raise ValueError("review_fraction must be in (0, 1]")
    labels = np.asarray(labels).reshape(-1)
    cohorts = {
        "test_all": split.test_mask.detach().cpu().numpy(),
        "test_seen": split.seen_test_mask.detach().cpu().numpy(),
        "test_cold_start": split.cold_start_test_mask.detach().cpu().numpy(),
    }
    rows = []
    for model_name, scores in run.scores.items():
        scores = np.asarray(scores).reshape(-1)
        if scores.shape != labels.shape:
            raise ValueError(f"Scores for {model_name} do not align with labels")
        for cohort_name, mask in cohorts.items():
            cohort_count = int(mask.sum())
            if cohort_count == 0:
                raise ValueError(f"Cohort {cohort_name} is empty")
            top_k = max(1, ceil(review_fraction * cohort_count))
            metrics = compute_binary_metrics(labels[mask], scores[mask], top_k=top_k)
            rows.append(
                {
                    "model": model_name,
                    "cohort": cohort_name,
                    "review_fraction": review_fraction,
                    **metrics.as_dict(),
                }
            )
    return pd.DataFrame(rows)


def _company_index_per_instrument(
    graph: HeteroData, edge_type: tuple[str, str, str]
) -> np.ndarray:
    edge_index = graph[edge_type].edge_index.detach().cpu().numpy()
    num_instruments = graph["instrument"].num_nodes
    if edge_index.shape[1] != num_instruments:
        raise ValueError(f"Relation {edge_type} must have one edge per instrument")
    order = np.argsort(edge_index[0], kind="stable")
    if not np.array_equal(edge_index[0, order], np.arange(num_instruments)):
        raise ValueError(
            f"Relation {edge_type} must cover every instrument exactly once"
        )
    return edge_index[1, order]


def _temporal_validation_indices(
    invoice_days: np.ndarray,
    train_indices: np.ndarray,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    ordered = train_indices[np.argsort(invoice_days[train_indices], kind="stable")]
    candidate = int((1 - validation_fraction) * ordered.size)
    candidate = min(max(candidate, 1), ordered.size - 1)
    validation_start = int(invoice_days[ordered[candidate]])
    fit_indices = train_indices[invoice_days[train_indices] < validation_start]
    validation_indices = train_indices[invoice_days[train_indices] >= validation_start]
    if fit_indices.size == 0 or validation_indices.size == 0:
        raise ValueError("Temporal validation produced an empty partition")
    return fit_indices, validation_indices, validation_start


def _lightgbm_model(config: BaselineConfig, n_estimators: int) -> lgb.LGBMClassifier:
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=config.seed,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _epoch_day_to_timestamp(epoch_day: int) -> pd.Timestamp:
    return pd.Timestamp(np.datetime64(epoch_day, "D"))

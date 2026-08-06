"""Multi-window causal backtesting for temporal tabular and graph models."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Sequence

import numpy as np
import pandas as pd

from graph_ml.baselines import (
    PointInTimeLightGBMConfig,
    build_point_in_time_feature_frame,
    evaluate_point_in_time_run,
    fit_point_in_time_lightgbm,
    point_in_time_cold_start_mask,
)
from graph_ml.evaluation import (
    RollingOriginFoldSpec,
    build_horizon_label_availability,
    build_point_in_time_fold,
)
from graph_ml.training import (
    TemporalGNNTrainingConfig,
    evaluate_temporal_gnn_run,
    fit_temporal_role_gnn,
)


@dataclass(frozen=True)
class TemporalBacktestConfig:
    """Frozen model and reporting settings for development-window backtests."""

    target_column: str = "is_pastdue90"
    horizon_days: int = 90
    seeds: tuple[int, ...] = (7, 19, 42, 73, 101)
    review_fraction: float = 0.05
    minimum_class_count: int = 10
    lightgbm: PointInTimeLightGBMConfig = field(
        default_factory=PointInTimeLightGBMConfig
    )
    temporal_gnn: TemporalGNNTrainingConfig = field(
        default_factory=TemporalGNNTrainingConfig
    )


def run_temporal_backtest(
    instruments: pd.DataFrame,
    fold_specs: Sequence[RollingOriginFoldSpec],
    config: TemporalBacktestConfig | None = None,
) -> pd.DataFrame:
    """Evaluate fixed model families over causal pre-holdout development folds.

    Feature histories are built once because every row's history is intrinsically
    query-time causal. Encoders, model selection, refits, and cold-start identity
    are recomputed inside each fold. The function never chooses a configuration;
    it returns the evidence required to do so outside a sealed final holdout.
    """

    config = config or TemporalBacktestConfig()
    specs = tuple(fold_specs)
    _validate_backtest_config(config, specs)
    availability = build_horizon_label_availability(
        instruments,
        target_column=config.target_column,
        horizon_days=config.horizon_days,
    )
    features = build_point_in_time_feature_frame(instruments)
    frames: list[pd.DataFrame] = []
    for fold_index, spec in enumerate(specs, start=1):
        fold = build_point_in_time_fold(
            instruments["invoice_date"], availability, spec
        )
        _require_minimum_class_counts(
            availability.labels, fold, config.minimum_class_count
        )
        cold = point_in_time_cold_start_mask(
            instruments, cutoff=fold.validation_end
        )
        cohorts = _nonempty_test_cohorts(fold.test_mask, cold)

        tabular_run = fit_point_in_time_lightgbm(
            features, availability, fold, config.lightgbm
        )
        tabular_metrics = evaluate_point_in_time_run(
            tabular_run,
            availability,
            cohorts,
            review_fraction=config.review_fraction,
        )
        frames.append(
            _annotate_metrics(
                tabular_metrics,
                fold_index=fold_index,
                fold=fold,
                seed=tabular_run.seed,
                selection_step=tabular_run.best_iteration,
                validation_pr_auc=tabular_run.best_validation_pr_auc,
            )
        )

        for use_context in (False, True):
            for seed in config.seeds:
                run_config = replace(
                    config.temporal_gnn,
                    seed=seed,
                    use_relation_context=use_context,
                )
                neural_run = fit_temporal_role_gnn(
                    instruments, features, availability, fold, run_config
                )
                neural_metrics = evaluate_temporal_gnn_run(
                    neural_run,
                    availability,
                    cohorts,
                    review_fraction=config.review_fraction,
                )
                frames.append(
                    _annotate_metrics(
                        neural_metrics,
                        fold_index=fold_index,
                        fold=fold,
                        seed=seed,
                        selection_step=neural_run.best_epoch,
                        validation_pr_auc=neural_run.best_validation_pr_auc,
                    )
                )
    return pd.concat(frames, ignore_index=True)


def _nonempty_test_cohorts(
    test_mask: np.ndarray, cold_mask: np.ndarray
) -> dict[str, np.ndarray]:
    cohorts = {"test_all": test_mask}
    for name, mask in (
        ("test_seen", test_mask & ~cold_mask),
        ("test_cold_start", test_mask & cold_mask),
    ):
        if mask.any():
            cohorts[name] = mask
    return cohorts


def _annotate_metrics(
    metrics: pd.DataFrame,
    *,
    fold_index: int,
    fold,
    seed: int,
    selection_step: int,
    validation_pr_auc: float,
) -> pd.DataFrame:
    output = metrics.copy()
    output.insert(0, "fold", fold_index)
    output.insert(1, "train_end", fold.train_end.date().isoformat())
    output.insert(2, "validation_end", fold.validation_end.date().isoformat())
    output.insert(3, "test_end", fold.test_end.date().isoformat())
    output.insert(4, "seed", seed)
    output.insert(5, "selection_step", selection_step)
    output.insert(6, "validation_pr_auc", validation_pr_auc)
    return output


def _validate_backtest_config(
    config: TemporalBacktestConfig, specs: tuple[RollingOriginFoldSpec, ...]
) -> None:
    if not specs:
        raise ValueError("fold_specs must not be empty")
    if config.horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be non-empty and unique")
    if not 0 < config.review_fraction <= 1:
        raise ValueError("review_fraction must be in (0, 1]")
    if config.minimum_class_count < 1:
        raise ValueError("minimum_class_count must be positive")


def _require_minimum_class_counts(labels, fold, minimum: int) -> None:
    for name, mask in (
        ("train", fold.train_mask),
        ("validation", fold.validation_mask),
        ("test", fold.test_mask),
    ):
        positives = int(labels[mask].sum())
        negatives = int(mask.sum()) - positives
        if min(positives, negatives) < minimum:
            raise ValueError(
                f"Fold {name} must contain at least {minimum} rows of each class"
            )

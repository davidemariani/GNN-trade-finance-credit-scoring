"""Leakage-safe point-in-time tabular features and LightGBM baseline."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING
from collections.abc import Mapping

import numpy as np
import pandas as pd

from graph_ml.data import canonicalize_company_name, query_strictly_prior_histories
from graph_ml.evaluation import (
    LabelAvailability,
    PointInTimeFold,
    compute_binary_metrics,
)

if TYPE_CHECKING:
    import lightgbm as lgb

_VALUE_COLUMNS = (
    "log_invoice_amount",
    "purchase_invoice_ratio",
    "payment_term_days",
    "input_lag_days",
)
_CATEGORICAL_COLUMNS = ("currency", "factoring_type")


@dataclass(frozen=True)
class PointInTimeFeatureFrame:
    """Raw causal numeric/history features plus origination categories."""

    frame: pd.DataFrame
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]


@dataclass(frozen=True)
class PointInTimeFeatureEncoder:
    """Fold-fitted medians and categorical vocabularies."""

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_medians: tuple[float, ...]
    categorical_values: tuple[tuple[str, ...], ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return transformed feature names in matrix-column order."""

        categorical_names = tuple(
            f"{column}={value}"
            for column, values in zip(
                self.categorical_columns, self.categorical_values, strict=True
            )
            for value in (*values, "__unknown__")
        )
        return self.numeric_columns + categorical_names

    def transform(self, features: PointInTimeFeatureFrame) -> np.ndarray:
        """Transform rows using training-fold statistics only."""

        if features.numeric_columns != self.numeric_columns:
            raise ValueError("Numeric feature contract does not match fitted encoder")
        if features.categorical_columns != self.categorical_columns:
            raise ValueError("Categorical feature contract does not match fitted encoder")
        numeric = features.frame.loc[:, self.numeric_columns].to_numpy(dtype=np.float64)
        medians = np.asarray(self.numeric_medians, dtype=np.float64)
        missing = ~np.isfinite(numeric)
        if missing.any():
            numeric[missing] = np.broadcast_to(medians, numeric.shape)[missing]

        encoded = [numeric]
        for column, values in zip(
            self.categorical_columns, self.categorical_values, strict=True
        ):
            source = features.frame[column].fillna("__missing__").astype(str)
            known = source.isin(values)
            for value in values:
                encoded.append(source.eq(value).to_numpy(dtype=np.float64)[:, None])
            encoded.append((~known).to_numpy(dtype=np.float64)[:, None])
        return np.column_stack(encoded)


@dataclass(frozen=True)
class PointInTimeLightGBMConfig:
    """Fixed settings for the first causal p90 tabular benchmark."""

    seed: int = 42
    max_estimators: int = 1_000
    early_stopping_rounds: int = 50


@dataclass(frozen=True)
class PointInTimeLightGBMRun:
    """Scores and fitted metadata from one rolling-origin baseline run."""

    scores: np.ndarray
    best_iteration: int
    best_validation_pr_auc: float
    feature_gains: tuple[tuple[str, float], ...]
    encoder: PointInTimeFeatureEncoder
    seed: int


def build_point_in_time_feature_frame(
    instruments: pd.DataFrame,
) -> PointInTimeFeatureFrame:
    """Build origination features and full role-aware, strictly-prior histories.

    Each endpoint receives the company's earlier seller-role and buyer-role
    histories. This preserves hybrid-company information without allowing the
    current invoice, a same-time sibling, or any future instrument into context.
    """

    required = {
        "customer_name_1",
        "debtor_name_1",
        "invoice_date",
        "due_date",
        "input_date",
        "invoice_amount",
        "purchase_amount",
        *_CATEGORICAL_COLUMNS,
    }
    missing = sorted(required - set(instruments.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not instruments.index.is_unique:
        raise ValueError("Instrument index must be unique")

    invoice_date = pd.to_datetime(instruments["invoice_date"], errors="coerce")
    due_date = pd.to_datetime(instruments["due_date"], errors="coerce")
    input_date = pd.to_datetime(instruments["input_date"], errors="coerce")
    if invoice_date.isna().any():
        raise ValueError("invoice_date must contain valid timestamps")
    invoice_amount = pd.to_numeric(instruments["invoice_amount"], errors="coerce")
    purchase_amount = pd.to_numeric(instruments["purchase_amount"], errors="coerce")
    if (invoice_amount.dropna() < 0).any() or (purchase_amount.dropna() < 0).any():
        raise ValueError("Invoice and purchase amounts cannot be negative")

    derived = pd.DataFrame(index=instruments.index)
    derived["log_invoice_amount"] = np.log1p(invoice_amount)
    derived["purchase_invoice_ratio"] = purchase_amount.div(
        invoice_amount.where(invoice_amount > 0)
    )
    derived["payment_term_days"] = (due_date - invoice_date).dt.days.astype(float)
    derived["input_lag_days"] = (input_date - invoice_date).dt.days.astype(float)
    events = pd.concat(
        (
            instruments[["customer_name_1", "debtor_name_1", "invoice_date"]],
            derived,
        ),
        axis=1,
    )
    output = pd.concat((derived, instruments[list(_CATEGORICAL_COLUMNS)]), axis=1)
    history_columns: list[str] = []
    endpoint_columns = {
        "seller_endpoint": "customer_name_1",
        "buyer_endpoint": "debtor_name_1",
    }
    history_roles = {
        "seller_role": "customer_name_1",
        "buyer_role": "debtor_name_1",
    }
    for endpoint_name, query_column in endpoint_columns.items():
        for role_name, event_column in history_roles.items():
            history = query_strictly_prior_histories(
                events,
                event_entity_column=event_column,
                event_timestamp_column="invoice_date",
                value_columns=_VALUE_COLUMNS,
                query_entities=instruments[query_column],
                query_timestamps=invoice_date,
            )
            prefix = f"{endpoint_name}__{role_name}__"
            renamed = history.rename(columns=lambda column: f"{prefix}{column}")
            output = pd.concat((output, renamed), axis=1)
            history_columns.extend(renamed.columns.tolist())

    return PointInTimeFeatureFrame(
        frame=output,
        numeric_columns=_VALUE_COLUMNS + tuple(history_columns),
        categorical_columns=_CATEGORICAL_COLUMNS,
    )


def fit_point_in_time_encoder(
    features: PointInTimeFeatureFrame, fit_mask: np.ndarray
) -> PointInTimeFeatureEncoder:
    """Fit imputation and category vocabularies on one legal training mask."""

    fit_mask = np.asarray(fit_mask, dtype=bool)
    if fit_mask.shape != (len(features.frame),) or not fit_mask.any():
        raise ValueError("fit_mask must select aligned training rows")
    fit = features.frame.loc[fit_mask]
    medians = (
        fit.loc[:, features.numeric_columns]
        .median(axis=0, skipna=True)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    categorical_values = tuple(
        tuple(sorted(fit[column].dropna().astype(str).unique().tolist()))
        for column in features.categorical_columns
    )
    return PointInTimeFeatureEncoder(
        numeric_columns=features.numeric_columns,
        categorical_columns=features.categorical_columns,
        numeric_medians=tuple(medians.tolist()),
        categorical_values=categorical_values,
    )


def transform_point_in_time_instruments(
    features: PointInTimeFeatureFrame,
    encoder: PointInTimeFeatureEncoder,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return only current-instrument inputs from a fold-fitted feature contract."""

    matrix = encoder.transform(features)
    names = encoder.feature_names
    selected = tuple(
        index
        for index, name in enumerate(names)
        if name in _VALUE_COLUMNS
        or any(name.startswith(f"{column}=") for column in _CATEGORICAL_COLUMNS)
    )
    return matrix[:, selected], tuple(names[index] for index in selected)


def fit_point_in_time_lightgbm(
    features: PointInTimeFeatureFrame,
    availability: LabelAvailability,
    fold: PointInTimeFold,
    config: PointInTimeLightGBMConfig | None = None,
) -> PointInTimeLightGBMRun:
    """Select tree count on rolling validation, refit legally, and score rows."""

    import lightgbm as lgb

    config = config or PointInTimeLightGBMConfig()
    if config.max_estimators < 1 or config.early_stopping_rounds < 1:
        raise ValueError("Estimator and early-stopping counts must be positive")
    labels = availability.labels.astype(np.int64)
    for name, mask in (
        ("train", fold.train_mask),
        ("validation", fold.validation_mask),
        ("refit", fold.refit_mask),
    ):
        if mask.shape != labels.shape or np.unique(labels[mask]).size != 2:
            raise ValueError(f"{name} cohort must align and contain both classes")

    search_encoder = fit_point_in_time_encoder(features, fold.train_mask)
    search_matrix = search_encoder.transform(features)
    search_model = _lightgbm_model(config, config.max_estimators)
    search_model.fit(
        search_matrix[fold.train_mask],
        labels[fold.train_mask],
        eval_X=search_matrix[fold.validation_mask],
        eval_y=labels[fold.validation_mask],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = int(search_model.best_iteration_ or config.max_estimators)
    best_validation_pr_auc = float(
        search_model.best_score_["valid_0"]["average_precision"]
    )

    final_encoder = fit_point_in_time_encoder(features, fold.refit_mask)
    final_matrix = final_encoder.transform(features)
    final_model = _lightgbm_model(config, best_iteration)
    final_model.fit(final_matrix[fold.refit_mask], labels[fold.refit_mask])
    scores = final_model.predict_proba(final_matrix)[:, 1]
    gains = final_model.booster_.feature_importance(importance_type="gain")
    feature_gains = tuple(
        sorted(
            zip(final_encoder.feature_names, gains, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    return PointInTimeLightGBMRun(
        scores=scores,
        best_iteration=best_iteration,
        best_validation_pr_auc=best_validation_pr_auc,
        feature_gains=feature_gains,
        encoder=final_encoder,
        seed=config.seed,
    )


def point_in_time_cold_start_mask(
    instruments: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp,
) -> np.ndarray:
    """Flag rows with a seller or buyer unseen before the deployment cutoff."""

    required = {"invoice_date", "customer_name_1", "debtor_name_1"}
    missing = sorted(required - set(instruments.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    dates = pd.to_datetime(instruments["invoice_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("invoice_date must contain valid timestamps")
    boundary = pd.Timestamp(cutoff).normalize()
    historical = dates.lt(boundary)
    known = set(
        instruments.loc[historical, "customer_name_1"].map(
            canonicalize_company_name
        )
    ) | set(
        instruments.loc[historical, "debtor_name_1"].map(canonicalize_company_name)
    )
    seller_seen = instruments["customer_name_1"].map(
        canonicalize_company_name
    ).isin(known)
    buyer_seen = instruments["debtor_name_1"].map(canonicalize_company_name).isin(
        known
    )
    return (~seller_seen | ~buyer_seen).to_numpy()


def evaluate_point_in_time_run(
    run: PointInTimeLightGBMRun,
    availability: LabelAvailability,
    cohorts: Mapping[str, np.ndarray],
    *,
    review_fraction: float = 0.05,
) -> pd.DataFrame:
    """Evaluate the causal baseline on named, row-aligned cohorts."""

    if not 0 < review_fraction <= 1:
        raise ValueError("review_fraction must be in (0, 1]")
    labels = availability.labels.astype(np.int64)
    rows = []
    for cohort, mask in cohorts.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != labels.shape or not mask.any():
            raise ValueError(f"Cohort {cohort} must select aligned rows")
        metrics = compute_binary_metrics(
            labels[mask],
            run.scores[mask],
            top_k=max(1, ceil(review_fraction * int(mask.sum()))),
        )
        rows.append(
            {
                "model": "point_in_time_lightgbm",
                "cohort": cohort,
                "review_fraction": review_fraction,
                **metrics.as_dict(),
            }
        )
    return pd.DataFrame(rows)


def _lightgbm_model(
    config: PointInTimeLightGBMConfig, n_estimators: int
) -> lgb.LGBMClassifier:
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

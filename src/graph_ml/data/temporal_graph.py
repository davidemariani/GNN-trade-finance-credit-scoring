"""Causal relation tensors for temporal trade-finance message passing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from graph_ml.data.temporal import query_time_decayed_histories


TEMPORAL_RELATIONS = (
    "seller_endpoint__seller_role",
    "seller_endpoint__buyer_role",
    "buyer_endpoint__seller_role",
    "buyer_endpoint__buyer_role",
)


@dataclass(frozen=True)
class TemporalRelationContext:
    """Time-decayed neighbor values and age/count metadata per relation."""

    values: np.ndarray
    metadata: np.ndarray
    relation_names: tuple[str, ...] = TEMPORAL_RELATIONS


def build_temporal_relation_context(
    instruments: pd.DataFrame,
    event_features: np.ndarray,
    *,
    half_life_days: float = 180.0,
) -> TemporalRelationContext:
    """Aggregate strictly-prior invoice neighbors through four typed channels.

    ``values`` has shape ``[N, 4, F]``. ``metadata`` has shape ``[N, 4, 3]``
    and stores ``log1p(count)``, ``log1p(age_days)``, and a history-present flag.
    No target or post-origination value enters either tensor.
    """

    required = {"customer_name_1", "debtor_name_1", "invoice_date"}
    missing = sorted(required - set(instruments.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    event_features = np.asarray(event_features, dtype=np.float64)
    if event_features.ndim != 2 or event_features.shape[0] != len(instruments):
        raise ValueError("event_features must have shape [num_instruments, features]")
    if not np.isfinite(event_features).all():
        raise ValueError("event_features must be finite")

    feature_columns = tuple(
        f"event_feature_{index}" for index in range(event_features.shape[1])
    )
    events = instruments[
        ["customer_name_1", "debtor_name_1", "invoice_date"]
    ].copy()
    for index, column in enumerate(feature_columns):
        events[column] = event_features[:, index]

    values = np.zeros(
        (len(instruments), len(TEMPORAL_RELATIONS), event_features.shape[1]),
        dtype=np.float32,
    )
    metadata = np.zeros(
        (len(instruments), len(TEMPORAL_RELATIONS), 3), dtype=np.float32
    )
    endpoint_columns = ("customer_name_1", "customer_name_1", "debtor_name_1", "debtor_name_1")
    event_columns = ("customer_name_1", "debtor_name_1", "customer_name_1", "debtor_name_1")
    for relation, (query_column, event_column) in enumerate(
        zip(endpoint_columns, event_columns, strict=True)
    ):
        history = query_time_decayed_histories(
            events,
            event_entity_column=event_column,
            event_timestamp_column="invoice_date",
            value_columns=feature_columns,
            query_entities=instruments[query_column],
            query_timestamps=instruments["invoice_date"],
            half_life_days=half_life_days,
        )
        mean_columns = [f"history_decay_mean__{column}" for column in feature_columns]
        values[:, relation] = history[mean_columns].fillna(0).to_numpy(np.float32)
        counts = history["history_count"].to_numpy(np.float32)
        ages = history["history_age_days"].to_numpy(np.float32)
        metadata[:, relation, 0] = np.log1p(counts)
        metadata[:, relation, 1] = np.log1p(ages)
        metadata[:, relation, 2] = counts > 0
    return TemporalRelationContext(values, metadata)

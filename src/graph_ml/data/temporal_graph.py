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


@dataclass(frozen=True)
class TemporalEventSequences:
    """Bounded newest-first causal event tensors for temporal attention.

    ``values`` has shape ``[N, R, K, F]``; ``age_days``, ``valid_mask``, and
    ``event_indices`` have shape ``[N, R, K]``. Invalid padding slots contain
    zero values/ages and source index ``-1``.
    """

    values: np.ndarray
    age_days: np.ndarray
    valid_mask: np.ndarray
    event_indices: np.ndarray
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
    events = instruments[["customer_name_1", "debtor_name_1", "invoice_date"]].copy()
    for index, column in enumerate(feature_columns):
        events[column] = event_features[:, index]

    values = np.zeros(
        (len(instruments), len(TEMPORAL_RELATIONS), event_features.shape[1]),
        dtype=np.float32,
    )
    metadata = np.zeros(
        (len(instruments), len(TEMPORAL_RELATIONS), 3), dtype=np.float32
    )
    endpoint_columns = (
        "customer_name_1",
        "customer_name_1",
        "debtor_name_1",
        "debtor_name_1",
    )
    event_columns = (
        "customer_name_1",
        "debtor_name_1",
        "customer_name_1",
        "debtor_name_1",
    )
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


def build_temporal_event_sequences(
    instruments: pd.DataFrame,
    event_features: np.ndarray,
    *,
    max_events: int,
) -> TemporalEventSequences:
    """Retain the newest ``K`` strictly-prior events in each typed relation.

    Events are newest-first within each relation. A current or same-timestamp
    event is never eligible because lookup uses a strict left boundary. This
    function performs no learned preprocessing; ``event_features`` must already
    come from the appropriate fold-fitted encoder.
    """

    required = {"customer_name_1", "debtor_name_1", "invoice_date"}
    missing = sorted(required - set(instruments.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_events < 1:
        raise ValueError("max_events must be positive")
    event_features = np.asarray(event_features, dtype=np.float64)
    if event_features.ndim != 2 or event_features.shape[0] != len(instruments):
        raise ValueError("event_features must have shape [num_instruments, features]")
    if not np.isfinite(event_features).all():
        raise ValueError("event_features must be finite")
    times = pd.to_datetime(instruments["invoice_date"], errors="coerce")
    if times.isna().any():
        raise ValueError("invoice_date must contain valid timestamps")
    if instruments[["customer_name_1", "debtor_name_1"]].isna().any(axis=None):
        raise ValueError("Company endpoints cannot be missing")

    size, feature_count = event_features.shape
    relation_count = len(TEMPORAL_RELATIONS)
    values = np.zeros(
        (size, relation_count, max_events, feature_count), dtype=np.float32
    )
    ages = np.zeros((size, relation_count, max_events), dtype=np.float32)
    valid = np.zeros((size, relation_count, max_events), dtype=bool)
    indices = np.full((size, relation_count, max_events), -1, dtype=np.int64)
    endpoint_columns = (
        "customer_name_1",
        "customer_name_1",
        "debtor_name_1",
        "debtor_name_1",
    )
    event_columns = (
        "customer_name_1",
        "debtor_name_1",
        "customer_name_1",
        "debtor_name_1",
    )
    time_values = times.to_numpy(dtype="datetime64[ns]")
    for relation, (query_column, event_column) in enumerate(
        zip(endpoint_columns, event_columns, strict=True)
    ):
        _fill_relation_sequences(
            instruments[query_column],
            instruments[event_column],
            time_values,
            event_features,
            max_events,
            values[:, relation],
            ages[:, relation],
            valid[:, relation],
            indices[:, relation],
        )
    return TemporalEventSequences(values, ages, valid, indices)


def _fill_relation_sequences(
    query_entities: pd.Series,
    event_entities: pd.Series,
    times: np.ndarray,
    features: np.ndarray,
    max_events: int,
    output_values: np.ndarray,
    output_ages: np.ndarray,
    output_valid: np.ndarray,
    output_indices: np.ndarray,
) -> None:
    from graph_ml.data.graph import canonicalize_company_name

    query_keys = query_entities.map(canonicalize_company_name).to_numpy()
    event_keys = event_entities.map(canonicalize_company_name).to_numpy()
    positions_by_entity: dict[str, list[int]] = {}
    for position, key in enumerate(event_keys):
        positions_by_entity.setdefault(key, []).append(position)
    histories = {}
    for key, positions in positions_by_entity.items():
        ordered = np.asarray(
            sorted(positions, key=lambda position: (times[position], position)),
            dtype=np.int64,
        )
        histories[key] = (times[ordered], ordered)

    nanoseconds_per_day = 86_400_000_000_000.0
    for query_position, (key, query_time) in enumerate(
        zip(query_keys, times, strict=True)
    ):
        history = histories.get(key)
        if history is None:
            continue
        history_times, history_positions = history
        end = int(np.searchsorted(history_times, query_time, side="left"))
        start = max(0, end - max_events)
        selected = history_positions[start:end][::-1]
        count = selected.size
        if count == 0:
            continue
        output_values[query_position, :count] = features[selected]
        output_ages[query_position, :count] = (query_time - times[selected]).astype(
            "timedelta64[ns]"
        ).astype(np.float64) / nanoseconds_per_day
        output_valid[query_position, :count] = True
        output_indices[query_position, :count] = selected

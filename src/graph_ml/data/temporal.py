"""Point-in-time feature primitives for temporal trade-finance models."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from graph_ml.data.graph import canonicalize_company_name


_FINAL_STATE_COLUMNS = frozenset(
    {
        "has_impairment1",
        "has_prosecution",
        "is_pastdue",
        "is_pastdue30",
        "is_pastdue90",
        "is_pastdue180",
        "is_open",
        "is_due",
        "has_discharge",
        "purchase_amount_open",
        "discharge_amount",
        "discharge_loss",
        "last_payment_amount",
        "total_repayment",
        "total_impairment",
    }
)
_POST_ORIGINATION_COLUMNS = frozenset(
    {
        "debt_collection_date",
        "last_payment_date",
        "reminder_date",
        "cancellation_date",
        "discharge_date",
        "payment_date",
        "posting_date",
        "first_posting_date",
        "last_posting_date",
        "payment_date_mismatch",
    }
)
_OUTCOME_AGGREGATE = re.compile(
    r"^(?:cd|d|c)_(?:repaid|impaired1|pastdue(?:30|90|180)?|pd_mismatch|we_payment_share)(?:_|$)"
)
_BOND_GRAPH = re.compile(r"^(?:imp|p90|p180)_(?:edge|d_node|c_node|node|energy)|^flow_shock_")


@dataclass(frozen=True)
class FeatureLeakageAudit:
    """Columns grouped by why they are unsafe without event-time reconstruction."""

    final_state: tuple[str, ...]
    post_origination: tuple[str, ...]
    outcome_aggregate: tuple[str, ...]
    bond_graph: tuple[str, ...]

    @property
    def prohibited(self) -> tuple[str, ...]:
        """Return every flagged column once, in deterministic order."""

        return tuple(
            sorted(
                set(self.final_state)
                | set(self.post_origination)
                | set(self.outcome_aggregate)
                | set(self.bond_graph)
            )
        )

    @property
    def is_safe(self) -> bool:
        """Whether no known point-in-time risk was found."""

        return not self.prohibited


def audit_point_in_time_columns(columns: Iterable[str]) -> FeatureLeakageAudit:
    """Classify known final-state, lifecycle, aggregate, and bond-graph columns.

    This is a schema guard, not proof that an unflagged column is safe. Provenance
    and availability timestamps still have to be established for every input.
    """

    names = tuple(dict.fromkeys(str(column) for column in columns))
    return FeatureLeakageAudit(
        final_state=tuple(sorted(name for name in names if name in _FINAL_STATE_COLUMNS)),
        post_origination=tuple(
            sorted(
                name
                for name in names
                if name in _POST_ORIGINATION_COLUMNS or name.startswith("dd_")
            )
        ),
        outcome_aggregate=tuple(
            sorted(name for name in names if _OUTCOME_AGGREGATE.match(name))
        ),
        bond_graph=tuple(sorted(name for name in names if _BOND_GRAPH.match(name))),
    )


def build_strictly_prior_histories(
    events: pd.DataFrame,
    *,
    entity_column: str,
    timestamp_column: str,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    """Build count/mean histories using only events at strictly earlier times.

    Rows sharing a timestamp never see one another. Output rows preserve the input
    order and index. ``history_count`` counts prior events, while each mean ignores
    missing values in its own source column. This primitive intentionally rejects
    known outcome, lifecycle, and precomputed bond-graph columns: those require an
    explicit observation timestamp before they may enter temporal state.
    """

    value_columns = tuple(dict.fromkeys(value_columns))
    required = {entity_column, timestamp_column, *value_columns}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not value_columns:
        raise ValueError("At least one value column is required")

    audit = audit_point_in_time_columns(value_columns)
    if not audit.is_safe:
        raise ValueError(
            "Point-in-time histories cannot use unverified outcome/lifecycle/bond "
            f"columns: {list(audit.prohibited)}"
        )

    timestamps = pd.to_datetime(events[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{timestamp_column} must contain valid timestamps")
    if events[entity_column].isna().any():
        raise ValueError(f"{entity_column} cannot contain missing entities")

    numeric = events.loc[:, value_columns].apply(pd.to_numeric, errors="coerce")
    invalid = numeric.isna() & events.loc[:, value_columns].notna()
    if invalid.any(axis=None):
        bad = sorted(invalid.columns[invalid.any()].tolist())
        raise ValueError(f"History value columns must be numeric: {bad}")

    working = numeric.copy()
    working["__entity"] = events[entity_column].map(canonicalize_company_name).to_numpy()
    working["__timestamp"] = timestamps.to_numpy()
    working["__row"] = np.arange(len(events), dtype=np.int64)

    group_keys = ["__entity", "__timestamp"]
    buckets = working.groupby(group_keys, sort=True, observed=True)
    bucket_size = buckets.size().rename("__bucket_size")
    bucket_sum = buckets[list(value_columns)].sum(min_count=1).add_suffix("__sum")
    bucket_valid = buckets[list(value_columns)].count().add_suffix("__valid")
    aggregate = pd.concat((bucket_size, bucket_sum, bucket_valid), axis=1).reset_index()
    aggregate = aggregate.sort_values(group_keys, kind="stable")

    aggregate["history_count"] = (
        aggregate.groupby("__entity", sort=False)["__bucket_size"].cumsum()
        - aggregate["__bucket_size"]
    )
    output_columns = ["history_count"]
    for column in value_columns:
        sum_column = f"{column}__sum"
        valid_column = f"{column}__valid"
        bucket_values = aggregate[sum_column].fillna(0)
        prior_sum = (
            bucket_values.groupby(aggregate["__entity"], sort=False).cumsum()
            - bucket_values
        )
        prior_valid = (
            aggregate.groupby("__entity", sort=False)[valid_column].cumsum()
            - aggregate[valid_column]
        )
        output = f"history_mean__{column}"
        aggregate[output] = prior_sum.div(prior_valid.where(prior_valid > 0))
        output_columns.append(output)

    joined = working[["__entity", "__timestamp", "__row"]].merge(
        aggregate[group_keys + output_columns],
        on=group_keys,
        how="left",
        validate="many_to_one",
    )
    joined = joined.sort_values("__row", kind="stable")
    result = joined[output_columns].reset_index(drop=True)
    result.index = events.index
    return result

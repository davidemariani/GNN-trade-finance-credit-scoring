from __future__ import annotations

import pandas as pd
import pytest

from graph_ml.data import (
    audit_point_in_time_columns,
    build_strictly_prior_histories,
    query_strictly_prior_histories,
    query_time_decayed_histories,
)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company": ["ACME", " acme ", "ACME", "Other", "ACME"],
            "time": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-02", "2020-01-03"]
            ),
            "amount": [10.0, 20.0, 40.0, 7.0, 50.0],
            "term": [1.0, None, 3.0, 5.0, 5.0],
        },
        index=[10, 11, 12, 13, 14],
    )


def test_histories_use_strictly_earlier_timestamps_and_preserve_rows():
    histories = build_strictly_prior_histories(
        _events(),
        entity_column="company",
        timestamp_column="time",
        value_columns=("amount", "term"),
    )

    assert histories.index.tolist() == [10, 11, 12, 13, 14]
    assert histories["history_count"].tolist() == [0, 1, 1, 0, 3]
    assert pd.isna(histories.loc[10, "history_mean__amount"])
    assert histories.loc[11, "history_mean__amount"] == 10.0
    assert histories.loc[12, "history_mean__amount"] == 10.0
    assert histories.loc[14, "history_mean__amount"] == pytest.approx(70 / 3)
    assert histories.loc[14, "history_mean__term"] == 2.0


def test_future_changes_cannot_modify_earlier_histories():
    events = _events()
    original = build_strictly_prior_histories(
        events,
        entity_column="company",
        timestamp_column="time",
        value_columns=("amount",),
    )
    changed = events.copy()
    changed.loc[14, "amount"] = 1_000_000.0
    rebuilt = build_strictly_prior_histories(
        changed,
        entity_column="company",
        timestamp_column="time",
        value_columns=("amount",),
    )

    pd.testing.assert_frame_equal(original.loc[:13], rebuilt.loc[:13])


def test_all_missing_bucket_does_not_erase_older_numeric_history():
    events = pd.DataFrame(
        {
            "company": ["A", "A", "A"],
            "time": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "amount": [10.0, None, 30.0],
        }
    )

    histories = build_strictly_prior_histories(
        events,
        entity_column="company",
        timestamp_column="time",
        value_columns=("amount",),
    )

    assert histories.loc[2, "history_count"] == 2
    assert histories.loc[2, "history_mean__amount"] == 10.0


def test_arbitrary_queries_support_cross_role_history_and_strict_ties():
    events = _events()
    query_entities = pd.Series(["acme", "ACME", "Other", "Never"], index=[4, 3, 2, 1])
    query_times = pd.Series(
        pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-03", "2020-01-03"]),
        index=query_entities.index,
    )

    histories = query_strictly_prior_histories(
        events,
        event_entity_column="company",
        event_timestamp_column="time",
        value_columns=("amount",),
        query_entities=query_entities,
        query_timestamps=query_times,
    )

    assert histories.index.tolist() == [4, 3, 2, 1]
    assert histories["history_count"].tolist() == [1, 3, 1, 0]
    assert histories["history_mean__amount"].iloc[0] == 10.0
    assert histories["history_mean__amount"].iloc[1] == pytest.approx(70 / 3)
    assert histories["history_mean__amount"].iloc[2] == 7.0
    assert pd.isna(histories["history_mean__amount"].iloc[3])


def test_time_decay_weights_recent_events_more_and_keeps_strict_order():
    events = pd.DataFrame(
        {
            "company": ["A", "A"],
            "time": pd.to_datetime(["2020-01-01", "2020-01-11"]),
            "value": [0.0, 10.0],
        }
    )
    queries = pd.Series(["A", "A"])
    times = pd.Series(pd.to_datetime(["2020-01-11", "2020-01-21"]))

    history = query_time_decayed_histories(
        events,
        event_entity_column="company",
        event_timestamp_column="time",
        value_columns=("value",),
        query_entities=queries,
        query_timestamps=times,
        half_life_days=10,
    )

    assert history["history_count"].tolist() == [1, 2]
    assert history["history_age_days"].tolist() == [10.0, 10.0]
    assert history.loc[0, "history_decay_mean__value"] == 0.0
    assert history.loc[1, "history_decay_mean__value"] == pytest.approx(20 / 3)


def test_infinite_half_life_is_exact_no_decay_mean():
    events = pd.DataFrame(
        {
            "company": ["A", "A"],
            "time": pd.to_datetime(["2020-01-01", "2020-01-11"]),
            "value": [0.0, 10.0],
        }
    )

    history = query_time_decayed_histories(
        events,
        event_entity_column="company",
        event_timestamp_column="time",
        value_columns=("value",),
        query_entities=pd.Series(["A"]),
        query_timestamps=pd.Series(pd.to_datetime(["2020-01-21"])),
        half_life_days=float("inf"),
    )

    assert history.loc[0, "history_decay_mean__value"] == 5.0


def test_schema_audit_flags_outcomes_lifecycle_aggregates_and_bond_features():
    audit = audit_point_in_time_columns(
        [
            "invoice_amount",
            "has_impairment1",
            "last_payment_date",
            "cd_impaired1_r",
            "p90_edge_flow",
            "flow_shock_imp1",
        ]
    )

    assert audit.final_state == ("has_impairment1",)
    assert audit.post_origination == ("last_payment_date",)
    assert audit.outcome_aggregate == ("cd_impaired1_r",)
    assert audit.bond_graph == ("flow_shock_imp1", "p90_edge_flow")
    assert not audit.is_safe


@pytest.mark.parametrize(
    "column",
    ["has_impairment1", "dd_last_payment_date", "d_pastdue90_r", "imp_energy"],
)
def test_history_builder_rejects_unverified_temporal_columns(column):
    events = _events().assign(**{column: 1.0})

    with pytest.raises(ValueError, match="unverified outcome/lifecycle/bond"):
        build_strictly_prior_histories(
            events,
            entity_column="company",
            timestamp_column="time",
            value_columns=(column,),
        )


def test_history_builder_rejects_non_numeric_values():
    events = _events().assign(category=["a", "b", "c", "d", "e"])

    with pytest.raises(ValueError, match="must be numeric"):
        build_strictly_prior_histories(
            events,
            entity_column="company",
            timestamp_column="time",
            value_columns=("category",),
        )

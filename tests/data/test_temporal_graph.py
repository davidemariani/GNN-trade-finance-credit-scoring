from __future__ import annotations

import numpy as np
import pandas as pd

from graph_ml.data import (
    TEMPORAL_RELATIONS,
    build_temporal_event_sequences,
    build_temporal_relation_context,
)


def test_temporal_context_is_typed_strict_and_future_invariant():
    frame = pd.DataFrame(
        {
            "customer_name_1": ["A", "B", "A", "C"],
            "debtor_name_1": ["D", "A", "E", "A"],
            "invoice_date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"]
            ),
        }
    )
    features = np.array([[1.0], [2.0], [4.0], [8.0]])

    context = build_temporal_relation_context(frame, features, half_life_days=10)

    assert context.values.shape == (4, 4, 1)
    assert context.metadata.shape == (4, 4, 3)
    assert context.relation_names == TEMPORAL_RELATIONS
    assert context.values[1, 1, 0] == 0
    assert context.values[1, 2, 0] == 1
    assert context.values[3, 3, 0] == 2
    assert context.metadata[3, 3, 2] == 1

    changed = features.copy()
    changed[3] = 1_000_000
    rebuilt = build_temporal_relation_context(frame, changed, half_life_days=10)
    np.testing.assert_allclose(context.values[:3], rebuilt.values[:3])
    np.testing.assert_allclose(context.metadata[:3], rebuilt.metadata[:3])


def test_event_sequences_are_newest_first_typed_and_strict():
    frame = pd.DataFrame(
        {
            "customer_name_1": ["A", "A", "B", "A", "C"],
            "debtor_name_1": ["D", "E", "A", "F", "A"],
            "invoice_date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03", "2020-01-04"]
            ),
        }
    )
    features = np.arange(1.0, 6.0)[:, None]

    sequences = build_temporal_event_sequences(frame, features, max_events=2)

    assert sequences.values.shape == (5, 4, 2, 1)
    assert sequences.age_days.shape == (5, 4, 2)
    assert sequences.valid_mask.dtype == bool
    assert sequences.relation_names == TEMPORAL_RELATIONS
    # A's seller-role history at 2020-01-03 contains the two latest earlier A events.
    assert sequences.event_indices[3, 0].tolist() == [1, 0]
    assert sequences.values[3, 0, :, 0].tolist() == [2.0, 1.0]
    assert sequences.age_days[3, 0].tolist() == [1.0, 2.0]
    # At 2020-01-02, neither same-timestamp row can see the other.
    assert sequences.event_indices[2, 2].tolist() == [0, -1]
    # A's buyer-role history is kept separate from A's seller-role history.
    assert sequences.event_indices[4, 3].tolist() == [2, -1]


def test_event_sequences_are_future_invariant_and_padding_is_zero():
    frame = pd.DataFrame(
        {
            "customer_name_1": ["A", "A", "A"],
            "debtor_name_1": ["B", "B", "B"],
            "invoice_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        }
    )
    features = np.array([[1.0], [2.0], [3.0]])
    original = build_temporal_event_sequences(frame, features, max_events=3)
    changed = features.copy()
    changed[2] = 999_999
    rebuilt = build_temporal_event_sequences(frame, changed, max_events=3)

    np.testing.assert_allclose(original.values[:2], rebuilt.values[:2])
    assert not original.valid_mask[0].any()
    assert np.count_nonzero(original.values[0]) == 0
    assert np.count_nonzero(original.age_days[0]) == 0
    assert (original.event_indices[0] == -1).all()


def test_event_sequences_reject_invalid_capacity():
    frame = pd.DataFrame(
        {
            "customer_name_1": ["A"],
            "debtor_name_1": ["B"],
            "invoice_date": pd.to_datetime(["2020-01-01"]),
        }
    )
    with np.testing.assert_raises_regex(ValueError, "max_events"):
        build_temporal_event_sequences(frame, np.ones((1, 1)), max_events=0)

from __future__ import annotations

import numpy as np
import pandas as pd

from graph_ml.data import TEMPORAL_RELATIONS, build_temporal_relation_context


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

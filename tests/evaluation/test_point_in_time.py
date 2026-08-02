from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graph_ml.evaluation import (
    RollingOriginFoldSpec,
    build_event_label_availability,
    build_horizon_label_availability,
    build_point_in_time_fold,
)


def test_horizon_labels_are_not_known_until_due_date_plus_horizon():
    frame = pd.DataFrame(
        {
            "due_date": pd.to_datetime(["2020-01-01", "2020-02-01", None]),
            "is_pastdue90": [False, True, True],
        }
    )

    availability = build_horizon_label_availability(
        frame, target_column="is_pastdue90", horizon_days=90
    )

    assert availability.available_at.iloc[0] == pd.Timestamp("2020-03-31")
    assert availability.known_by("2020-03-30").tolist() == [False, False, False]
    assert availability.known_by("2020-05-01").tolist() == [True, True, False]


def test_event_labels_require_the_timestamp_matching_their_class():
    frame = pd.DataFrame(
        {
            "target": [True, False, True, False],
            "event_date": pd.to_datetime(["2020-02-01", None, None, None]),
            "resolution_date": pd.to_datetime(
                [None, "2020-03-01", "2020-02-15", None]
            ),
        }
    )

    availability = build_event_label_availability(
        frame,
        target_column="target",
        positive_event_date_column="event_date",
        negative_resolution_date_column="resolution_date",
    )

    assert availability.valid_mask.tolist() == [True, True, False, False]
    assert availability.known_by("2020-02-15").tolist() == [True, False, False, False]
    assert availability.known_by("2020-03-01").tolist() == [True, True, False, False]


def test_rolling_fold_respects_origin_windows_and_label_availability():
    prediction_times = pd.Series(
        pd.to_datetime(
            ["2020-01-01", "2020-02-20", "2020-03-10", "2020-04-10", "2020-05-01"]
        )
    )
    frame = pd.DataFrame(
        {
            "target": [False, True, False, True, False],
            "event": pd.to_datetime([None, "2020-04-01", None, "2020-05-20", None]),
            "resolution": pd.to_datetime(
                ["2020-02-01", None, "2020-04-20", None, "2020-06-01"]
            ),
        }
    )
    availability = build_event_label_availability(
        frame,
        target_column="target",
        positive_event_date_column="event",
        negative_resolution_date_column="resolution",
    )

    fold = build_point_in_time_fold(
        prediction_times,
        availability,
        RollingOriginFoldSpec(
            train_end="2020-03-01",
            validation_end="2020-05-01",
            test_end="2020-07-01",
        ),
    )

    assert fold.train_mask.tolist() == [True, False, False, False, False]
    assert fold.validation_mask.tolist() == [False, False, True, False, False]
    assert fold.refit_mask.tolist() == [True, True, True, False, False]
    assert fold.test_mask.tolist() == [False, False, False, False, True]
    assert fold.summary() == {
        "train_end": "2020-03-01",
        "validation_end": "2020-05-01",
        "test_end": "2020-07-01",
        "train_rows": 1,
        "validation_rows": 1,
        "refit_rows": 3,
        "test_rows": 1,
    }


def test_fold_masks_are_disjoint():
    prediction_times = pd.Series(pd.date_range("2020-01-01", periods=6, freq="MS"))
    frame = pd.DataFrame(
        {
            "due_date": prediction_times,
            "target": [False, True, False, True, False, True],
        }
    )
    availability = build_horizon_label_availability(
        frame, target_column="target", horizon_days=1
    )
    fold = build_point_in_time_fold(
        prediction_times,
        availability,
        RollingOriginFoldSpec("2020-03-01", "2020-05-01", "2020-07-01"),
    )

    assert not np.any(fold.train_mask & fold.validation_mask)
    assert not np.any(fold.refit_mask & fold.test_mask)
    assert np.all(fold.train_mask <= fold.refit_mask)


@pytest.mark.parametrize(
    "spec",
    [
        RollingOriginFoldSpec("2020-03-01", "2020-03-01", "2020-04-01"),
        RollingOriginFoldSpec("2020-05-01", "2020-04-01", "2020-06-01"),
    ],
)
def test_rejects_non_chronological_boundaries(spec):
    frame = pd.DataFrame(
        {"due_date": pd.to_datetime(["2020-01-01"]), "target": [False]}
    )
    availability = build_horizon_label_availability(
        frame, target_column="target", horizon_days=1
    )

    with pytest.raises(ValueError, match="train_end < validation_end < test_end"):
        build_point_in_time_fold(frame["due_date"], availability, spec)

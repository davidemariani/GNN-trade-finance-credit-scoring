from __future__ import annotations

import numpy as np
import pandas as pd

from graph_ml.baselines import (
    PointInTimeLightGBMConfig,
    build_point_in_time_feature_frame,
    evaluate_point_in_time_run,
    fit_point_in_time_encoder,
    fit_point_in_time_lightgbm,
    point_in_time_cold_start_mask,
)
from graph_ml.evaluation import (
    RollingOriginFoldSpec,
    build_horizon_label_availability,
    build_point_in_time_fold,
)


def _instruments() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    return pd.DataFrame(
        {
            "customer_name_1": ["A", "B", "A", "C"] * 3,
            "debtor_name_1": ["D", "A", "E", "A"] * 3,
            "invoice_date": dates,
            "due_date": dates + pd.Timedelta(days=30),
            "input_date": dates + pd.Timedelta(days=1),
            "invoice_amount": np.arange(100.0, 1_300.0, 100.0),
            "purchase_amount": np.arange(90.0, 1_170.0, 90.0),
            "currency": ["EUR"] * 4 + ["USD"] * 8,
            "factoring_type": ["Full Service"] * 12,
            "is_pastdue90": [False, True] * 6,
        }
    )


def _protocol(frame: pd.DataFrame):
    availability = build_horizon_label_availability(
        frame, target_column="is_pastdue90", horizon_days=1
    )
    fold = build_point_in_time_fold(
        frame["invoice_date"],
        availability,
        RollingOriginFoldSpec("2020-05-01", "2020-09-01", "2021-01-01"),
    )
    return availability, fold


def test_feature_frame_contains_role_aware_strictly_prior_hybrid_history():
    frame = _instruments()
    features = build_point_in_time_feature_frame(frame)

    assert len(features.numeric_columns) == 24
    assert features.categorical_columns == ("currency", "factoring_type")
    assert (
        features.frame.loc[
            1, "buyer_endpoint__seller_role__history_count"
        ]
        == 1
    )
    assert features.frame.loc[0, "seller_endpoint__seller_role__history_count"] == 0


def test_future_amount_change_cannot_modify_earlier_feature_rows():
    frame = _instruments()
    original = build_point_in_time_feature_frame(frame).frame
    changed = frame.copy()
    changed.loc[11, "invoice_amount"] = 99_999_999.0
    rebuilt = build_point_in_time_feature_frame(changed).frame

    pd.testing.assert_frame_equal(original.iloc[:11], rebuilt.iloc[:11])


def test_encoder_fits_categories_on_training_fold_only():
    frame = _instruments()
    features = build_point_in_time_feature_frame(frame)
    _, fold = _protocol(frame)
    encoder = fit_point_in_time_encoder(features, fold.train_mask)
    matrix = encoder.transform(features)

    assert encoder.categorical_values[0] == ("EUR",)
    unknown = encoder.feature_names.index("currency=__unknown__")
    assert matrix[0, unknown] == 0
    assert matrix[4, unknown] == 1


def test_fits_causal_lightgbm_with_rolling_validation_and_refit():
    frame = _instruments()
    availability, fold = _protocol(frame)
    features = build_point_in_time_feature_frame(frame)

    run = fit_point_in_time_lightgbm(
        features,
        availability,
        fold,
        PointInTimeLightGBMConfig(max_estimators=10, early_stopping_rounds=3),
    )

    assert run.scores.shape == (len(frame),)
    assert np.isfinite(run.scores).all()
    assert 1 <= run.best_iteration <= 10
    assert len(run.feature_gains) == len(run.encoder.feature_names)

    cold = point_in_time_cold_start_mask(frame, cutoff="2020-09-01")
    metrics = evaluate_point_in_time_run(
        run,
        availability,
        {"test_all": fold.test_mask, "test_seen": fold.test_mask & ~cold},
    )
    assert metrics["cohort"].tolist() == ["test_all", "test_seen"]
    assert metrics["sample_count"].tolist() == [3, 3]

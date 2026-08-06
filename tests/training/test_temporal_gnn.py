from __future__ import annotations

import numpy as np
import pandas as pd

from graph_ml.baselines import (
    build_point_in_time_feature_frame,
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


def _setup():
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    frame = pd.DataFrame(
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
    availability = build_horizon_label_availability(
        frame, target_column="is_pastdue90", horizon_days=1
    )
    fold = build_point_in_time_fold(
        frame["invoice_date"],
        availability,
        RollingOriginFoldSpec("2020-05-01", "2020-09-01", "2021-01-01"),
    )
    return frame, availability, fold


def test_temporal_gnn_trains_scores_and_evaluates_causal_cohorts():
    frame, availability, fold = _setup()
    features = build_point_in_time_feature_frame(frame)

    run = fit_temporal_role_gnn(
        frame,
        features,
        availability,
        fold,
        TemporalGNNTrainingConfig(
            hidden_channels=8,
            dropout=0,
            max_epochs=5,
            patience=2,
            half_life_days=30,
        ),
    )

    assert run.scores.shape == (12,)
    assert run.embeddings.shape == (12, 8)
    assert np.isfinite(run.scores).all()
    assert 1 <= run.best_epoch <= 5
    assert run.parameter_count > 0
    cold = point_in_time_cold_start_mask(frame, cutoff="2020-09-01")
    metrics = evaluate_temporal_gnn_run(
        run,
        availability,
        {"test_all": fold.test_mask, "test_seen": fold.test_mask & ~cold},
    )
    assert metrics["sample_count"].tolist() == [3, 3]


def test_root_only_training_run_is_labeled_as_control():
    frame, availability, fold = _setup()
    features = build_point_in_time_feature_frame(frame)
    run = fit_temporal_role_gnn(
        frame,
        features,
        availability,
        fold,
        TemporalGNNTrainingConfig(
            hidden_channels=4,
            dropout=0,
            max_epochs=2,
            patience=2,
            seed=3,
            use_relation_context=False,
        ),
    )

    metrics = evaluate_temporal_gnn_run(
        run, availability, {"test": fold.test_mask}, review_fraction=0.5
    )

    assert metrics["model"].item() == "root_only_neural"


def test_shared_relation_training_run_has_distinct_label():
    frame, availability, fold = _setup()
    run = fit_temporal_role_gnn(
        frame,
        build_point_in_time_feature_frame(frame),
        availability,
        fold,
        TemporalGNNTrainingConfig(
            hidden_channels=4,
            dropout=0,
            max_epochs=2,
            patience=2,
            relation_mode="shared",
        ),
    )

    metrics = evaluate_temporal_gnn_run(
        run, availability, {"test": fold.test_mask}, review_fraction=0.5
    )

    assert metrics["model"].item() == "temporal_role_gnn_shared"


def test_recent_event_training_run_has_distinct_label():
    frame, availability, fold = _setup()
    run = fit_temporal_role_gnn(
        frame,
        build_point_in_time_feature_frame(frame),
        availability,
        fold,
        TemporalGNNTrainingConfig(
            hidden_channels=4,
            dropout=0,
            max_epochs=2,
            patience=2,
            max_recent_events=2,
        ),
    )

    metrics = evaluate_temporal_gnn_run(
        run, availability, {"test": fold.test_mask}, review_fraction=0.5
    )

    assert run.max_recent_events == 2
    assert metrics["model"].item() == "temporal_role_gnn_recent_k2"

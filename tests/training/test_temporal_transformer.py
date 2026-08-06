from __future__ import annotations

import numpy as np
import pandas as pd

from graph_ml.baselines import build_point_in_time_feature_frame
from graph_ml.evaluation import (
    RollingOriginFoldSpec,
    build_horizon_label_availability,
    build_point_in_time_fold,
)
from graph_ml.training import (
    TemporalTransformerTrainingConfig,
    evaluate_temporal_transformer_run,
    fit_temporal_graph_transformer,
)


def test_transformer_selects_refits_scores_and_evaluates():
    dates = pd.date_range("2020-01-01", periods=16, freq="MS")
    frame = pd.DataFrame(
        {
            "customer_name_1": ["A", "B", "A", "C"] * 4,
            "debtor_name_1": ["D", "A", "E", "A"] * 4,
            "invoice_date": dates,
            "due_date": dates + pd.Timedelta(days=30),
            "input_date": dates + pd.Timedelta(days=1),
            "invoice_amount": np.arange(100.0, 1_700.0, 100.0),
            "purchase_amount": np.arange(90.0, 1_530.0, 90.0),
            "currency": ["EUR", "USD"] * 8,
            "factoring_type": ["Full Service"] * 16,
            "target": [False, True] * 8,
        }
    )
    availability = build_horizon_label_availability(
        frame, target_column="target", horizon_days=1
    )
    fold = build_point_in_time_fold(
        frame["invoice_date"],
        availability,
        RollingOriginFoldSpec("2020-07-01", "2020-12-01", "2021-05-01"),
    )
    run = fit_temporal_graph_transformer(
        frame,
        build_point_in_time_feature_frame(frame),
        availability,
        fold,
        TemporalTransformerTrainingConfig(
            hidden_channels=8,
            attention_heads=2,
            max_events=2,
            dropout=0,
            batch_size=4,
            max_epochs=2,
            patience=2,
            seed=3,
        ),
    )
    metrics = evaluate_temporal_transformer_run(
        run, availability, {"test_all": fold.test_mask}, review_fraction=0.5
    )

    assert run.scores.shape == (16,)
    assert np.isfinite(run.scores).all()
    assert 1 <= run.best_epoch <= 2
    assert metrics["model"].item() == "temporal_graph_transformer"
    assert metrics["sample_count"].item() == int(fold.test_mask.sum())

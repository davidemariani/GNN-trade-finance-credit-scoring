from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graph_ml.backtesting import TemporalBacktestConfig, run_temporal_backtest
from graph_ml.baselines import PointInTimeLightGBMConfig
from graph_ml.evaluation import RollingOriginFoldSpec
from graph_ml.training import TemporalGNNTrainingConfig
from graph_ml.training import TemporalTransformerTrainingConfig


def _instruments() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    return pd.DataFrame(
        {
            "customer_name_1": ["A", "B", "A", "C"] * 6,
            "debtor_name_1": ["D", "A", "E", "A"] * 6,
            "invoice_date": dates,
            "due_date": dates + pd.Timedelta(days=30),
            "input_date": dates + pd.Timedelta(days=1),
            "invoice_amount": np.arange(100.0, 2_500.0, 100.0),
            "purchase_amount": np.arange(90.0, 2_170.0, 90.0),
            "currency": ["EUR", "USD"] * 12,
            "factoring_type": ["Full Service"] * 24,
            "target": [False, True] * 12,
        }
    )


def test_temporal_backtest_runs_all_fixed_model_families():
    metrics = run_temporal_backtest(
        _instruments(),
        [RollingOriginFoldSpec("2020-09-01", "2021-05-01", "2022-01-01")],
        TemporalBacktestConfig(
            target_column="target",
            horizon_days=1,
            seeds=(3,),
            review_fraction=0.5,
            minimum_class_count=1,
            lightgbm=PointInTimeLightGBMConfig(
                seed=3, max_estimators=3, early_stopping_rounds=1
            ),
            temporal_gnn=TemporalGNNTrainingConfig(
                hidden_channels=4,
                dropout=0,
                max_epochs=2,
                patience=2,
                seed=3,
            ),
            temporal_transformer=TemporalTransformerTrainingConfig(
                hidden_channels=4,
                attention_heads=1,
                max_events=2,
                dropout=0,
                batch_size=4,
                max_epochs=2,
                patience=2,
                seed=3,
            ),
        ),
    )

    assert set(metrics["model"]) == {
        "point_in_time_lightgbm",
        "root_only_neural",
        "temporal_role_gnn",
        "temporal_graph_transformer",
    }
    assert metrics["fold"].eq(1).all()
    assert metrics["test_end"].eq("2022-01-01").all()
    assert metrics["validation_pr_auc"].notna().all()
    assert metrics["sample_count"].gt(0).all()


@pytest.mark.parametrize(
    "config,message",
    [
        (TemporalBacktestConfig(seeds=()), "seeds"),
        (TemporalBacktestConfig(review_fraction=0), "review_fraction"),
    ],
)
def test_temporal_backtest_rejects_invalid_config(config, message):
    with pytest.raises(ValueError, match=message):
        run_temporal_backtest(
            _instruments(),
            [RollingOriginFoldSpec("2020-09-01", "2021-05-01", "2022-01-01")],
            config,
        )


def test_temporal_backtest_rejects_statistically_thin_partition():
    with pytest.raises(ValueError, match="at least 10 rows of each class"):
        run_temporal_backtest(
            _instruments(),
            [RollingOriginFoldSpec("2020-09-01", "2021-05-01", "2022-01-01")],
            TemporalBacktestConfig(target_column="target", horizon_days=1),
        )

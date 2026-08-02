"""Classical reference models that set the bar for graph models."""

from graph_ml.baselines.tabular import (
    BaselineConfig,
    BaselineRun,
    TabularFeatureSet,
    assemble_tabular_features,
    evaluate_baseline_run,
    fit_tabular_baselines,
)
from graph_ml.baselines.point_in_time import (
    PointInTimeFeatureEncoder,
    PointInTimeFeatureFrame,
    PointInTimeLightGBMConfig,
    PointInTimeLightGBMRun,
    build_point_in_time_feature_frame,
    evaluate_point_in_time_run,
    fit_point_in_time_encoder,
    fit_point_in_time_lightgbm,
    point_in_time_cold_start_mask,
    transform_point_in_time_instruments,
)

__all__ = [
    "BaselineConfig",
    "BaselineRun",
    "PointInTimeFeatureEncoder",
    "PointInTimeFeatureFrame",
    "PointInTimeLightGBMConfig",
    "PointInTimeLightGBMRun",
    "TabularFeatureSet",
    "assemble_tabular_features",
    "build_point_in_time_feature_frame",
    "evaluate_point_in_time_run",
    "evaluate_baseline_run",
    "fit_tabular_baselines",
    "fit_point_in_time_encoder",
    "fit_point_in_time_lightgbm",
    "point_in_time_cold_start_mask",
    "transform_point_in_time_instruments",
]

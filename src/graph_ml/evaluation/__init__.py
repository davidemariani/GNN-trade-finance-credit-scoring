"""Temporal evaluation cohorts, graph views, and rare-event metrics."""

from graph_ml.evaluation.metrics import BinaryMetrics, compute_binary_metrics
from graph_ml.evaluation.point_in_time import (
    ExpandingWindowPlan,
    LabelAvailability,
    PointInTimeFold,
    RollingOriginFoldSpec,
    build_event_label_availability,
    build_expanding_window_specs,
    build_horizon_label_availability,
    build_point_in_time_fold,
)
from graph_ml.evaluation.split import (
    TemporalEvaluationSplit,
    TemporalGraphViews,
    TemporalSplitConfig,
    build_temporal_evaluation_split,
    build_temporal_graph_views,
)

__all__ = [
    "BinaryMetrics",
    "ExpandingWindowPlan",
    "LabelAvailability",
    "PointInTimeFold",
    "RollingOriginFoldSpec",
    "TemporalEvaluationSplit",
    "TemporalGraphViews",
    "TemporalSplitConfig",
    "build_temporal_evaluation_split",
    "build_temporal_graph_views",
    "build_event_label_availability",
    "build_expanding_window_specs",
    "build_horizon_label_availability",
    "build_point_in_time_fold",
    "compute_binary_metrics",
]

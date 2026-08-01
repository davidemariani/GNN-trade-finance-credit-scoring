"""Classical reference models that set the bar for graph models."""

from graph_ml.baselines.tabular import (
    BaselineConfig,
    BaselineRun,
    TabularFeatureSet,
    assemble_tabular_features,
    evaluate_baseline_run,
    fit_tabular_baselines,
)

__all__ = [
    "BaselineConfig",
    "BaselineRun",
    "TabularFeatureSet",
    "assemble_tabular_features",
    "evaluate_baseline_run",
    "fit_tabular_baselines",
]

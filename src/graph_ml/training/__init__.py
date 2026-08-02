"""Training protocols for graph models."""

from graph_ml.training.graphsage import (
    GraphSAGETrainingConfig,
    GraphSAGETrainingRun,
    evaluate_graphsage_run,
    fit_hetero_graphsage,
)

__all__ = [
    "GraphSAGETrainingConfig",
    "GraphSAGETrainingRun",
    "evaluate_graphsage_run",
    "fit_hetero_graphsage",
]

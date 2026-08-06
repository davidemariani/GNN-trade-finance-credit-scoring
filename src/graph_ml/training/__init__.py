"""Training protocols for graph models."""

from graph_ml.training.graphsage import (
    GraphSAGETrainingConfig,
    GraphSAGETrainingRun,
    evaluate_graphsage_run,
    fit_hetero_graphsage,
)
from graph_ml.training.temporal_gnn import (
    TemporalGNNTrainingConfig,
    TemporalGNNTrainingRun,
    evaluate_temporal_gnn_run,
    fit_temporal_role_gnn,
)
from graph_ml.training.temporal_transformer import (
    TemporalTransformerTrainingConfig,
    TemporalTransformerTrainingRun,
    evaluate_temporal_transformer_run,
    fit_temporal_graph_transformer,
)

__all__ = [
    "GraphSAGETrainingConfig",
    "GraphSAGETrainingRun",
    "TemporalGNNTrainingConfig",
    "TemporalGNNTrainingRun",
    "TemporalTransformerTrainingConfig",
    "TemporalTransformerTrainingRun",
    "evaluate_graphsage_run",
    "fit_hetero_graphsage",
    "evaluate_temporal_gnn_run",
    "fit_temporal_role_gnn",
    "evaluate_temporal_transformer_run",
    "fit_temporal_graph_transformer",
]

"""Reusable aggregate EDA and graph-topology visualizations."""

from graph_ml.viz.eda import (
    class_balance_frame,
    plot_class_balance,
    plot_temporal_volume,
    temporal_volume_frame,
)
from graph_ml.viz.modeling import (
    embedding_projection_frame,
    plot_embedding_projection,
    plot_score_distributions,
    plot_seed_variability,
    plot_training_history,
    seed_metric_summary,
)
from graph_ml.viz.topology import (
    anonymous_company_ego_graph,
    build_interactive_ego_network,
    company_degree_frame,
    component_size_frame,
    hybrid_footprint,
    plot_anonymous_ego_graph,
    plot_company_degree_distributions,
)
from graph_ml.viz.teaching import (
    plot_baseline_pr_auc,
    plot_binary_ranking_curves,
    plot_feature_gains,
    plot_graph_schema,
    plot_message_passing_steps,
    plot_temporal_cohorts,
)

__all__ = [
    "anonymous_company_ego_graph",
    "build_interactive_ego_network",
    "class_balance_frame",
    "company_degree_frame",
    "component_size_frame",
    "embedding_projection_frame",
    "hybrid_footprint",
    "plot_anonymous_ego_graph",
    "plot_baseline_pr_auc",
    "plot_binary_ranking_curves",
    "plot_class_balance",
    "plot_company_degree_distributions",
    "plot_embedding_projection",
    "plot_feature_gains",
    "plot_graph_schema",
    "plot_message_passing_steps",
    "plot_score_distributions",
    "plot_seed_variability",
    "plot_temporal_cohorts",
    "plot_temporal_volume",
    "plot_training_history",
    "seed_metric_summary",
    "temporal_volume_frame",
]

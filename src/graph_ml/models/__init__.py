"""Graph neural network architectures used by the project."""

from graph_ml.models.hetero_graphsage import HeteroGraphSAGE
from graph_ml.models.temporal_role_gnn import TemporalRoleGNN

__all__ = ["HeteroGraphSAGE", "TemporalRoleGNN"]

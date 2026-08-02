"""Relation-aware GraphSAGE for the company/instrument graph."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch_geometric.nn import HeteroConv, SAGEConv

NodeType = str
EdgeType = tuple[str, str, str]

TRADE_FINANCE_EDGE_TYPES: tuple[EdgeType, ...] = (
    ("instrument", "sold_by", "company"),
    ("company", "sells", "instrument"),
    ("instrument", "owed_by", "company"),
    ("company", "owes", "instrument"),
)


class HeteroGraphSAGE(nn.Module):
    """Two-layer, relation-aware GraphSAGE instrument classifier.

    Args:
        instrument_channels: Width of ``x_dict["instrument"]``.
        company_channels: Width of ``x_dict["company"]``.
        hidden_channels: Width of both learned node representations.
        dropout: Dropout probability applied after each hidden layer.

    Inputs:
        ``x_dict`` maps ``instrument`` to ``[num_instruments,
        instrument_channels]`` and ``company`` to ``[num_companies,
        company_channels]``. ``edge_index_dict`` contains all four
        :data:`TRADE_FINANCE_EDGE_TYPES`, each shaped ``[2, num_edges]``.

    Returns:
        One unbounded impairment logit per instrument, shaped
        ``[num_instruments]``.

    Each relation owns independent GraphSAGE parameters. Relation outputs
    arriving at the same node type are summed by :class:`HeteroConv`. The
    root transformation built into :class:`SAGEConv` supplies self information;
    explicit homogeneous self-loop edge types are neither expected nor needed.
    """

    def __init__(
        self,
        instrument_channels: int,
        company_channels: int,
        *,
        hidden_channels: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if instrument_channels < 1 or company_channels < 1:
            raise ValueError("Input channel counts must be positive")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        input_channels = {
            "instrument": instrument_channels,
            "company": company_channels,
        }
        self.first_convolution = self._heterogeneous_layer(
            input_channels, hidden_channels
        )
        self.first_normalizations = nn.ModuleDict(
            {
                "instrument": nn.LayerNorm(hidden_channels),
                "company": nn.LayerNorm(hidden_channels),
            }
        )
        self.instrument_convolutions = nn.ModuleDict(
            {
                relation: SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                    aggr="mean",
                    normalize=False,
                    root_weight=True,
                )
                for relation in ("sells", "owes")
            }
        )
        self.instrument_normalization = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, 1)

    def encode(
        self,
        x_dict: Mapping[NodeType, Tensor],
        edge_index_dict: Mapping[EdgeType, Tensor],
    ) -> dict[NodeType, Tensor]:
        """Return hidden states for both node types after two message-passing layers."""

        self._validate_inputs(x_dict, edge_index_dict)
        first = self.first_convolution(dict(x_dict), edge_index_dict)
        first = {
            node_type: self.dropout(
                torch.relu(self.first_normalizations[node_type](node_features))
            )
            for node_type, node_features in first.items()
        }
        second = sum(
            convolution(
                (first["company"], first["instrument"]),
                edge_index_dict[("company", relation, "instrument")],
            )
            for relation, convolution in self.instrument_convolutions.items()
        )
        instrument = self.dropout(torch.relu(self.instrument_normalization(second)))
        return {"instrument": instrument, "company": first["company"]}

    def forward(
        self,
        x_dict: Mapping[NodeType, Tensor],
        edge_index_dict: Mapping[EdgeType, Tensor],
    ) -> Tensor:
        """Return one impairment logit for every instrument node."""

        hidden = self.encode(x_dict, edge_index_dict)
        return self.classifier(hidden["instrument"]).squeeze(-1)

    @staticmethod
    def _heterogeneous_layer(
        channels: Mapping[NodeType, int],
        hidden_channels: int,
    ) -> HeteroConv:
        return HeteroConv(
            {
                edge_type: SAGEConv(
                    (channels[edge_type[0]], channels[edge_type[2]]),
                    hidden_channels,
                    aggr="mean",
                    normalize=False,
                    root_weight=True,
                )
                for edge_type in TRADE_FINANCE_EDGE_TYPES
            },
            aggr="sum",
        )

    @staticmethod
    def _validate_inputs(
        x_dict: Mapping[NodeType, Tensor],
        edge_index_dict: Mapping[EdgeType, Tensor],
    ) -> None:
        missing_nodes = {"instrument", "company"} - set(x_dict)
        if missing_nodes:
            raise ValueError(
                f"Missing node features: {', '.join(sorted(missing_nodes))}"
            )
        missing_edges = set(TRADE_FINANCE_EDGE_TYPES) - set(edge_index_dict)
        if missing_edges:
            relations = ", ".join(edge_type[1] for edge_type in sorted(missing_edges))
            raise ValueError(f"Missing edge relations: {relations}")
        for node_type in ("instrument", "company"):
            if x_dict[node_type].ndim != 2:
                raise ValueError(f"{node_type} features must be a rank-2 tensor")
        for edge_type in TRADE_FINANCE_EDGE_TYPES:
            edge_index = edge_index_dict[edge_type]
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(f"Relation {edge_type} must have shape [2, num_edges]")

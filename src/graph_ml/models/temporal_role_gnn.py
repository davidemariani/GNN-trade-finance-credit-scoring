"""Role-aware temporal message passing over causal invoice histories."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from graph_ml.data import TEMPORAL_RELATIONS


class TemporalRoleGNN(nn.Module):
    """Classify invoices from self features and four causal relation channels.

    Historical instrument neighbors are aggregated at company endpoints before
    this module is called. Each relation owns a learned message transform and a
    gate conditioned on log-count, age, and history presence. This is one
    temporal bipartite message-passing layer with explicit relation semantics.
    """

    def __init__(
        self,
        instrument_channels: int,
        *,
        hidden_channels: int = 64,
        dropout: float = 0.2,
        use_relation_context: bool = True,
        relation_mode: Literal["separate", "shared"] = "separate",
    ) -> None:
        super().__init__()
        if instrument_channels < 1 or hidden_channels < 1:
            raise ValueError("Channel counts must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if relation_mode not in {"separate", "shared"}:
            raise ValueError("relation_mode must be separate or shared")
        self.instrument_channels = instrument_channels
        self.use_relation_context = use_relation_context
        self.relation_mode = relation_mode
        relation_layer_count = (
            len(TEMPORAL_RELATIONS)
            if use_relation_context and relation_mode == "separate"
            else int(use_relation_context)
        )
        self.root = nn.Linear(instrument_channels, hidden_channels)
        self.relation_messages = nn.ModuleList(
            nn.Linear(instrument_channels, hidden_channels, bias=False)
            for _ in range(relation_layer_count)
        )
        self.temporal_messages = nn.ModuleList(
            nn.Linear(3, hidden_channels, bias=False)
            for _ in range(relation_layer_count)
        )
        self.temporal_gates = nn.ModuleList(
            nn.Linear(3, hidden_channels) for _ in range(relation_layer_count)
        )
        self.normalization = nn.LayerNorm(hidden_channels)
        self.refinement = nn.Linear(hidden_channels, hidden_channels)
        self.refinement_normalization = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, 1)

    def encode(
        self, instrument_features: Tensor, context: Tensor, metadata: Tensor
    ) -> Tensor:
        """Return one hidden representation per current instrument."""

        self._validate_inputs(instrument_features, context, metadata)
        hidden = self.root(instrument_features)
        for relation in range(len(TEMPORAL_RELATIONS)):
            if not self.use_relation_context:
                break
            layer = relation if self.relation_mode == "separate" else 0
            message_layer = self.relation_messages[layer]
            time_layer = self.temporal_messages[layer]
            gate_layer = self.temporal_gates[layer]
            temporal = metadata[:, relation]
            message = message_layer(context[:, relation]) + time_layer(temporal)
            hidden = hidden + torch.sigmoid(gate_layer(temporal)) * message
        hidden = self.dropout(torch.relu(self.normalization(hidden)))
        refined = self.refinement(hidden)
        return self.dropout(torch.relu(self.refinement_normalization(refined + hidden)))

    def forward(
        self, instrument_features: Tensor, context: Tensor, metadata: Tensor
    ) -> Tensor:
        """Return one unbounded p90 logit per instrument."""

        return self.classifier(
            self.encode(instrument_features, context, metadata)
        ).squeeze(-1)

    def _validate_inputs(
        self, instrument_features: Tensor, context: Tensor, metadata: Tensor
    ) -> None:
        if instrument_features.ndim != 2:
            raise ValueError("instrument_features must have shape [N, F]")
        expected_context = (
            instrument_features.shape[0],
            len(TEMPORAL_RELATIONS),
            self.instrument_channels,
        )
        if tuple(context.shape) != expected_context:
            raise ValueError(f"context must have shape {expected_context}")
        expected_metadata = (
            instrument_features.shape[0],
            len(TEMPORAL_RELATIONS),
            3,
        )
        if tuple(metadata.shape) != expected_metadata:
            raise ValueError(f"metadata must have shape {expected_metadata}")

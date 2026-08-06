"""Causally masked attention over bounded role-specific invoice histories."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from graph_ml.data import TEMPORAL_RELATIONS


class TemporalGraphTransformer(nn.Module):
    """Classify current invoices using attention over legal historical events."""

    def __init__(
        self,
        instrument_channels: int,
        *,
        hidden_channels: int = 64,
        attention_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if instrument_channels < 1 or hidden_channels < 1 or attention_heads < 1:
            raise ValueError("Channel and attention-head counts must be positive")
        if hidden_channels % attention_heads:
            raise ValueError("hidden_channels must be divisible by attention_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.instrument_channels = instrument_channels
        self.hidden_channels = hidden_channels
        self.root = nn.Linear(instrument_channels, hidden_channels)
        self.event_projection = nn.Linear(instrument_channels, hidden_channels)
        self.time_projection = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.relation_embedding = nn.Embedding(len(TEMPORAL_RELATIONS), hidden_channels)
        self.attention = nn.MultiheadAttention(
            hidden_channels,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_normalization = nn.LayerNorm(hidden_channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_channels, 2 * hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_channels, hidden_channels),
        )
        self.output_normalization = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, 1)

    def attend(
        self,
        instrument_features: Tensor,
        event_values: Tensor,
        age_days: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return attention messages and one averaged weight per event slot."""

        self._validate_inputs(instrument_features, event_values, age_days, valid_mask)
        size, relations, slots, _ = event_values.shape
        root = self.root(instrument_features)
        message = torch.zeros_like(root)
        weights = torch.zeros(
            size, relations * slots, dtype=root.dtype, device=root.device
        )
        flattened_mask = valid_mask.reshape(size, relations * slots)
        active = flattened_mask.any(dim=1)
        if active.any():
            relation_ids = torch.arange(
                relations, device=root.device
            ).repeat_interleave(slots)
            event_hidden = self.event_projection(
                event_values.reshape(size, relations * slots, self.instrument_channels)
            )
            time_hidden = self.time_projection(
                torch.log1p(age_days).reshape(size, relations * slots, 1)
            )
            keys = event_hidden + time_hidden + self.relation_embedding(relation_ids)
            attended, active_weights = self.attention(
                root[active, None, :],
                keys[active],
                keys[active],
                key_padding_mask=~flattened_mask[active],
                need_weights=True,
                average_attn_weights=True,
            )
            message[active] = attended[:, 0]
            weights[active] = active_weights[:, 0]
        return message, weights.reshape(size, relations, slots)

    def encode(
        self,
        instrument_features: Tensor,
        event_values: Tensor,
        age_days: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Return one residual temporal-attention representation per invoice."""

        root = self.root(instrument_features)
        message, _ = self.attend(
            instrument_features, event_values, age_days, valid_mask
        )
        hidden = self.attention_normalization(root + self.dropout(message))
        return self.output_normalization(
            hidden + self.dropout(self.feed_forward(hidden))
        )

    def forward(
        self,
        instrument_features: Tensor,
        event_values: Tensor,
        age_days: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Return one unbounded p90 logit per current invoice."""

        return self.classifier(
            self.encode(instrument_features, event_values, age_days, valid_mask)
        ).squeeze(-1)

    def _validate_inputs(
        self,
        instrument_features: Tensor,
        event_values: Tensor,
        age_days: Tensor,
        valid_mask: Tensor,
    ) -> None:
        if instrument_features.ndim != 2:
            raise ValueError("instrument_features must have shape [N, F]")
        if event_values.ndim != 4:
            raise ValueError("event_values must have shape [N, R, K, F]")
        size, relations, slots, channels = event_values.shape
        if (size, channels) != (
            instrument_features.shape[0],
            self.instrument_channels,
        ) or instrument_features.shape[1] != self.instrument_channels:
            raise ValueError("Current and event feature shapes must align")
        expected = (size, relations, slots)
        if relations != len(TEMPORAL_RELATIONS):
            raise ValueError("event_values must contain all temporal relations")
        if tuple(age_days.shape) != expected or tuple(valid_mask.shape) != expected:
            raise ValueError("age_days and valid_mask must align with event slots")
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean")
        if (age_days[valid_mask] <= 0).any():
            raise ValueError("Valid event ages must be positive")

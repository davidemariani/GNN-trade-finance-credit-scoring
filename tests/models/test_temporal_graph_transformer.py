from __future__ import annotations

import torch
from torch.nn import functional as F

from graph_ml.models import TemporalGraphTransformer


def _inputs():
    torch.manual_seed(4)
    current = torch.randn(5, 3)
    events = torch.randn(5, 4, 3, 3)
    ages = torch.ones(5, 4, 3)
    valid = torch.tensor(
        [
            [[False, False, False]] * 4,
            [[True, False, False]] * 4,
            [[True, True, False]] * 4,
            [[True, True, True]] * 4,
            [[True, False, False]] * 4,
        ]
    )
    return current, events, ages, valid


def test_attention_weights_normalize_and_empty_history_falls_back():
    model = TemporalGraphTransformer(3, hidden_channels=8, attention_heads=2, dropout=0)
    model.eval()
    current, events, ages, valid = _inputs()

    message, weights = model.attend(current, events, ages, valid)
    hidden = model.encode(current, events, ages, valid)

    assert message.shape == (5, 8)
    assert weights.shape == (5, 4, 3)
    assert hidden.shape == (5, 8)
    assert weights[0].sum() == 0
    torch.testing.assert_close(message[0], torch.zeros(8))
    torch.testing.assert_close(weights[1:].sum(dim=(1, 2)), torch.ones(4))
    assert (weights[~valid] == 0).all()


def test_masked_padding_values_cannot_change_logits():
    model = TemporalGraphTransformer(3, hidden_channels=8, attention_heads=2, dropout=0)
    model.eval()
    current, events, ages, valid = _inputs()
    changed = events.clone()
    changed[~valid] = 1_000_000

    torch.testing.assert_close(
        model(current, events, ages, valid),
        model(current, changed, ages, valid),
    )


def test_temporal_graph_transformer_loss_decreases_on_tiny_case():
    model = TemporalGraphTransformer(3, hidden_channels=8, attention_heads=2, dropout=0)
    current, events, ages, valid = _inputs()
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            model(current, events, ages, valid), labels
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0] * 0.5

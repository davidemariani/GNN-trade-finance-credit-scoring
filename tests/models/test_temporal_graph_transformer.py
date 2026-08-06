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


def test_fixed_decay_bias_prefers_the_newer_of_equal_events():
    model = TemporalGraphTransformer(
        3,
        hidden_channels=8,
        attention_heads=2,
        dropout=0,
        time_encoding="fixed_decay",
        fixed_half_life_days=10,
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    current = torch.zeros(1, 3)
    events = torch.zeros(1, 4, 2, 3)
    ages = torch.ones(1, 4, 2)
    ages[0, 0] = torch.tensor([1.0, 11.0])
    valid = torch.zeros(1, 4, 2, dtype=torch.bool)
    valid[0, 0] = True

    _, weights = model.attend(current, events, ages, valid)

    assert weights[0, 0, 0] == torch.tensor(2 / 3)
    assert weights[0, 0, 1] == torch.tensor(1 / 3)
    assert (weights[~valid] == 0).all()


def test_no_time_ablation_is_invariant_to_valid_event_ages():
    model = TemporalGraphTransformer(
        3, hidden_channels=8, attention_heads=2, dropout=0, time_encoding="none"
    )
    model.eval()
    current, events, ages, valid = _inputs()
    changed_ages = ages.clone()
    changed_ages[valid] *= 100

    torch.testing.assert_close(
        model(current, events, ages, valid),
        model(current, events, changed_ages, valid),
    )


def test_coverage_gate_returns_one_learned_scalar_per_invoice():
    model = TemporalGraphTransformer(
        3,
        hidden_channels=8,
        attention_heads=2,
        dropout=0,
        fusion="coverage_gate",
    )
    model.coverage_gate.weight.data.zero_()
    model.coverage_gate.bias.data.fill_(torch.logit(torch.tensor(0.25)))
    current, _, _, valid = _inputs()

    gates = model.fusion_weights(current, valid)

    assert gates.shape == (5, 1)
    torch.testing.assert_close(gates, torch.full((5, 1), 0.25))


def test_residual_fusion_is_exactly_full_strength():
    model = TemporalGraphTransformer(3, hidden_channels=8, attention_heads=2)
    current, _, _, valid = _inputs()

    torch.testing.assert_close(model.fusion_weights(current, valid), torch.ones(5, 1))


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

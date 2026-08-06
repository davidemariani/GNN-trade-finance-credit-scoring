from __future__ import annotations

import pytest
import torch

from graph_ml.models import TemporalRoleGNN


def test_temporal_role_gnn_shapes_and_gradients():
    torch.manual_seed(3)
    model = TemporalRoleGNN(5, hidden_channels=8, dropout=0)
    features = torch.randn(6, 5)
    context = torch.randn(6, 4, 5)
    metadata = torch.randn(6, 4, 3)

    hidden = model.encode(features, context, metadata)
    logits = model(features, context, metadata)
    logits.sum().backward()

    assert hidden.shape == (6, 8)
    assert logits.shape == (6,)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_relation_channels_have_independent_parameters():
    model = TemporalRoleGNN(3, hidden_channels=4)

    assert len(model.relation_messages) == 4
    assert (
        model.relation_messages[0].weight.data_ptr()
        != model.relation_messages[1].weight.data_ptr()
    )


def test_shared_relation_model_is_invariant_to_role_permutation():
    model = TemporalRoleGNN(3, hidden_channels=4, dropout=0, relation_mode="shared")
    model.eval()
    features = torch.randn(5, 3)
    context = torch.randn(5, 4, 3)
    metadata = torch.randn(5, 4, 3)
    permutation = torch.tensor([2, 0, 3, 1])

    original = model(features, context, metadata)
    permuted = model(features, context[:, permutation], metadata[:, permutation])

    torch.testing.assert_close(original, permuted)
    assert len(model.relation_messages) == 1


def test_empty_history_has_exact_zero_message_fallback():
    model = TemporalRoleGNN(3, hidden_channels=4, dropout=0)
    model.eval()
    features = torch.randn(5, 3)
    empty_context = torch.zeros(5, 4, 3)
    empty_metadata = torch.zeros(5, 4, 3)

    encoded = model.encode(features, empty_context, empty_metadata)
    root = model.root(features)
    hidden = torch.relu(model.normalization(root))
    expected = torch.relu(
        model.refinement_normalization(model.refinement(hidden) + hidden)
    )

    torch.testing.assert_close(encoded, expected)


@pytest.mark.parametrize(
    ("context_shape", "metadata_shape", "message"),
    [
        ((2, 3, 5), (2, 4, 3), "context"),
        ((2, 4, 5), (2, 4, 2), "metadata"),
    ],
)
def test_rejects_misaligned_temporal_tensors(context_shape, metadata_shape, message):
    model = TemporalRoleGNN(5)

    with pytest.raises(ValueError, match=message):
        model(
            torch.zeros(2, 5),
            torch.zeros(context_shape),
            torch.zeros(metadata_shape),
        )


def test_root_only_control_is_independent_of_relation_context():
    model = TemporalRoleGNN(2, hidden_channels=4, dropout=0, use_relation_context=False)
    model.eval()
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    metadata = torch.zeros(2, 4, 3)

    first = model(features, torch.zeros(2, 4, 2), metadata)
    second = model(features, torch.full((2, 4, 2), 999.0), metadata)

    torch.testing.assert_close(first, second)
    assert len(model.relation_messages) == 0

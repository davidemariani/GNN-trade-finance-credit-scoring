from __future__ import annotations

import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData

from graph_ml.models import HeteroGraphSAGE


def _toy_graph() -> HeteroData:
    graph = HeteroData()
    graph["instrument"].x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]]
    )
    graph["company"].x = torch.tensor([[1.0], [0.0], [0.5]])
    sold_by = torch.tensor([[0, 1, 2], [0, 1, 0]])
    owed_by = torch.tensor([[0, 1, 2], [1, 2, 2]])
    graph["instrument", "sold_by", "company"].edge_index = sold_by
    graph["company", "sells", "instrument"].edge_index = sold_by.flip(0)
    graph["instrument", "owed_by", "company"].edge_index = owed_by
    graph["company", "owes", "instrument"].edge_index = owed_by.flip(0)
    return graph


def test_output_and_embedding_shapes_include_an_isolated_instrument():
    graph = _toy_graph()
    model = HeteroGraphSAGE(2, 1, hidden_channels=8, dropout=0)

    embeddings = model.encode(graph.x_dict, graph.edge_index_dict)
    logits = model(graph.x_dict, graph.edge_index_dict)

    assert embeddings["instrument"].shape == (4, 8)
    assert embeddings["company"].shape == (3, 8)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_gradients_flow_and_tiny_graph_loss_decreases():
    torch.manual_seed(3)
    graph = _toy_graph()
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    model = HeteroGraphSAGE(2, 1, hidden_channels=8, dropout=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    initial = float(
        F.binary_cross_entropy_with_logits(
            model(graph.x_dict, graph.edge_index_dict), labels
        ).detach()
    )
    for _ in range(50):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            model(graph.x_dict, graph.edge_index_dict), labels
        )
        loss.backward()
        optimizer.step()

    final = float(
        F.binary_cross_entropy_with_logits(
            model(graph.x_dict, graph.edge_index_dict), labels
        ).detach()
    )
    assert final < initial * 0.5
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_rejects_missing_relations_and_invalid_configuration():
    graph = _toy_graph()
    model = HeteroGraphSAGE(2, 1)
    incomplete_edges = dict(graph.edge_index_dict)
    incomplete_edges.pop(("company", "owes", "instrument"))

    try:
        model(graph.x_dict, incomplete_edges)
    except ValueError as error:
        assert "Missing edge relations" in str(error)
    else:
        raise AssertionError("Expected missing relations to be rejected")

    try:
        HeteroGraphSAGE(2, 1, dropout=1.0)
    except ValueError as error:
        assert "dropout" in str(error)
    else:
        raise AssertionError("Expected invalid dropout to be rejected")

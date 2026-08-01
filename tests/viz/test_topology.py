from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pyvis.network import Network

from graph_ml.data import GraphBuildConfig, build_trade_finance_graph
from graph_ml.viz import (
    anonymous_company_ego_graph,
    build_interactive_ego_network,
    company_degree_frame,
    component_size_frame,
    hybrid_footprint,
    plot_anonymous_ego_graph,
    plot_company_degree_distributions,
)


@pytest.fixture
def graph_result():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    frame = pd.DataFrame(
        {
            "uid": ["i1", "i2", "i3"],
            "customer_name_1": ["A", "C", "D"],
            "debtor_name_1": ["B", "A", "E"],
            "invoice_date": dates,
            "due_date": dates + pd.Timedelta(days=30),
            "input_date": dates,
            "invoice_amount": [100.0, 200.0, 300.0],
            "purchase_amount": [90.0, 180.0, 270.0],
            "currency": ["EUR"] * 3,
            "factoring_type": ["Full Service"] * 3,
            "has_impairment1": [False, True, False],
        }
    )
    return build_trade_finance_graph(frame, GraphBuildConfig(cutoff="2020-01-04"))


def test_degree_component_and_hybrid_summaries(graph_result):
    graph = graph_result.graph
    degrees = company_degree_frame(graph)
    components = component_size_frame(graph)
    footprint = hybrid_footprint(graph)

    a_index = graph_result.metadata.company_index["a"]
    a = degrees.loc[degrees["company_index"] == a_index].iloc[0]
    assert a["seller_degree"] == 1
    assert a["buyer_degree"] == 1
    assert bool(a["is_hybrid"])
    assert components["total_nodes"].tolist() == [5, 3]
    assert footprint == {
        "hybrid_companies": 1,
        "instruments_touching_hybrid": 2,
        "instrument_share_touching_hybrid": pytest.approx(2 / 3),
    }


def test_anonymous_ego_graph_and_visual_builders(graph_result):
    graph = graph_result.graph
    a_index = graph_result.metadata.company_index["a"]
    ego = anonymous_company_ego_graph(graph, a_index)

    assert len(ego) == 5
    assert all(node.startswith(("company:", "instrument:")) for node in ego.nodes)
    assert {data["relation"] for _, _, data in ego.edges(data=True)} == {
        "sold_by",
        "owed_by",
    }

    degree_figure = plot_company_degree_distributions(company_degree_frame(graph))
    ego_figure = plot_anonymous_ego_graph(ego)
    interactive = build_interactive_ego_network(ego)
    assert degree_figure.axes[0].get_xscale() == "log"
    assert ego_figure.axes[0].get_title().startswith("Anonymous")
    assert isinstance(interactive, Network)
    assert len(interactive.nodes) == len(ego.nodes)
    plt.close(degree_figure)
    plt.close(ego_figure)


def test_ego_graph_rejects_invalid_company(graph_result):
    with pytest.raises(ValueError, match="out of range"):
        anonymous_company_ego_graph(graph_result.graph, 999)

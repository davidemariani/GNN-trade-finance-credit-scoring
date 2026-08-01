"""Topology summaries and anonymous graph visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
import torch
from torch_geometric.data import HeteroData


def company_degree_frame(graph: HeteroData) -> pd.DataFrame:
    """Return role-specific instrument degrees for every company node."""

    graph.validate(raise_on_error=True)
    seller = graph["instrument", "sold_by", "company"].edge_index[1]
    buyer = graph["instrument", "owed_by", "company"].edge_index[1]
    count = graph["company"].num_nodes
    seller_degree = torch.bincount(seller, minlength=count).cpu().numpy()
    buyer_degree = torch.bincount(buyer, minlength=count).cpu().numpy()
    return pd.DataFrame(
        {
            "company_index": np.arange(count),
            "seller_degree": seller_degree,
            "buyer_degree": buyer_degree,
            "total_degree": seller_degree + buyer_degree,
            "is_hybrid": (seller_degree > 0) & (buyer_degree > 0),
        }
    )


def component_size_frame(graph: HeteroData) -> pd.DataFrame:
    """Return connected-component sizes in the bipartite graph."""

    network = _full_anonymous_networkx(graph)
    instrument_count = graph["instrument"].num_nodes
    rows = []
    components = sorted(nx.connected_components(network), key=len, reverse=True)
    for rank, component in enumerate(components, start=1):
        instruments = sum(node < instrument_count for node in component)
        rows.append(
            {
                "component_rank": rank,
                "total_nodes": len(component),
                "instrument_nodes": instruments,
                "company_nodes": len(component) - instruments,
            }
        )
    return pd.DataFrame(rows)


def hybrid_footprint(graph: HeteroData) -> dict[str, int | float]:
    """Summarize hybrid companies and instruments touching at least one."""

    degrees = company_degree_frame(graph)
    hybrid = torch.from_numpy(degrees["is_hybrid"].to_numpy(copy=True))
    touched = torch.zeros(graph["instrument"].num_nodes, dtype=torch.bool)
    for edge_type in (
        ("instrument", "sold_by", "company"),
        ("instrument", "owed_by", "company"),
    ):
        edge_index = graph[edge_type].edge_index
        hybrid_edges = hybrid[edge_index[1]]
        touched[edge_index[0, hybrid_edges]] = True
    touched_count = int(touched.sum())
    return {
        "hybrid_companies": int(hybrid.sum()),
        "instruments_touching_hybrid": touched_count,
        "instrument_share_touching_hybrid": touched_count
        / graph["instrument"].num_nodes,
    }


def anonymous_company_ego_graph(
    graph: HeteroData,
    company_index: int,
    *,
    max_instruments_per_role: int = 12,
) -> nx.Graph:
    """Build a bounded two-hop ego graph without business identifiers."""

    if not 0 <= company_index < graph["company"].num_nodes:
        raise ValueError("company_index is out of range")
    if max_instruments_per_role < 1:
        raise ValueError("max_instruments_per_role must be positive")

    sold_by = graph["instrument", "sold_by", "company"].edge_index.cpu().numpy()
    owed_by = graph["instrument", "owed_by", "company"].edge_index.cpu().numpy()
    selected: set[int] = set()
    for edge_index in (sold_by, owed_by):
        candidates = edge_index[0, edge_index[1] == company_index]
        selected.update(sorted(candidates.tolist())[:max_instruments_per_role])

    ego = nx.Graph()
    focal = f"company:{company_index}"
    ego.add_node(focal, node_type="company", focal=True)
    for instrument_index in sorted(selected):
        instrument_node = f"instrument:{instrument_index}"
        ego.add_node(instrument_node, node_type="instrument", focal=False)
        for relation, edge_index in (("sold_by", sold_by), ("owed_by", owed_by)):
            matching = edge_index[1, edge_index[0] == instrument_index]
            company_node = f"company:{int(matching[0])}"
            ego.add_node(
                company_node,
                node_type="company",
                focal=int(matching[0]) == company_index,
            )
            ego.add_edge(instrument_node, company_node, relation=relation)
    return ego


def plot_company_degree_distributions(degrees: pd.DataFrame) -> Figure:
    """Plot seller- and buyer-role degree distributions on log axes."""

    figure, axis = plt.subplots(figsize=(7, 4.5))
    bins = np.logspace(0, np.log10(max(2, degrees["total_degree"].max())), 30)
    axis.hist(
        [
            degrees.loc[degrees["seller_degree"] > 0, "seller_degree"],
            degrees.loc[degrees["buyer_degree"] > 0, "buyer_degree"],
        ],
        bins=bins,
        label=["Seller-role degree", "Buyer-role degree"],
        alpha=0.7,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(
        title="Company role-degree distributions",
        xlabel="Connected instruments",
        ylabel="Companies",
    )
    axis.legend()
    figure.tight_layout()
    return figure


def plot_anonymous_ego_graph(ego: nx.Graph, *, seed: int = 42) -> Figure:
    """Plot a deterministic anonymous company/instrument ego network."""

    positions = nx.spring_layout(ego, seed=seed)
    figure, axis = plt.subplots(figsize=(8, 6))
    node_colors = [
        "#C44E52"
        if data.get("focal")
        else "#4C72B0"
        if data["node_type"] == "company"
        else "#DD8452"
        for _, data in ego.nodes(data=True)
    ]
    edge_colors = [
        "#55A868" if data["relation"] == "sold_by" else "#8172B2"
        for _, _, data in ego.edges(data=True)
    ]
    company_nodes = [
        node for node, data in ego.nodes(data=True) if data["node_type"] == "company"
    ]
    instrument_nodes = [
        node for node, data in ego.nodes(data=True) if data["node_type"] == "instrument"
    ]
    labels = {node: f"C{index}" for index, node in enumerate(company_nodes)} | {
        node: f"I{index}" for index, node in enumerate(instrument_nodes)
    }
    nx.draw_networkx(
        ego,
        positions,
        ax=axis,
        labels=labels,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=430,
        width=1.2,
        font_size=7,
        font_color="white",
        font_weight="bold",
    )
    axis.set_title("Anonymous two-hop ego graph around a hybrid company")
    axis.legend(
        handles=[
            Patch(color="#C44E52", label="Focal hybrid company"),
            Patch(color="#4C72B0", label="Other company"),
            Patch(color="#DD8452", label="Instrument"),
            Line2D([0], [0], color="#55A868", lw=2, label="sold_by"),
            Line2D([0], [0], color="#8172B2", lw=2, label="owed_by"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    axis.set_axis_off()
    figure.tight_layout()
    return figure


def build_interactive_ego_network(ego: nx.Graph) -> Network:
    """Create a pyvis network with anonymous labels and role-colored edges."""

    network = Network(height="650px", width="100%", bgcolor="#ffffff", directed=False)
    for node, data in ego.nodes(data=True):
        node_type = data["node_type"]
        color = (
            "#C44E52"
            if data.get("focal")
            else "#4C72B0"
            if node_type == "company"
            else "#DD8452"
        )
        network.add_node(
            node,
            label="",
            title=node_type,
            color=color,
            size=22 if data.get("focal") else 10,
        )
    for source, destination, data in ego.edges(data=True):
        relation = data["relation"]
        network.add_edge(
            source,
            destination,
            title=relation,
            color="#55A868" if relation == "sold_by" else "#8172B2",
        )
    network.set_options('{"layout":{"randomSeed":42},"physics":{"solver":"barnesHut"}}')
    return network


def _full_anonymous_networkx(graph: HeteroData) -> nx.Graph:
    instrument_count = graph["instrument"].num_nodes
    company_count = graph["company"].num_nodes
    network = nx.Graph()
    network.add_nodes_from(range(instrument_count + company_count))
    for edge_type in (
        ("instrument", "sold_by", "company"),
        ("instrument", "owed_by", "company"),
    ):
        edge_index = graph[edge_type].edge_index.cpu().numpy()
        network.add_edges_from(
            zip(edge_index[0], edge_index[1] + instrument_count, strict=True)
        )
    return network

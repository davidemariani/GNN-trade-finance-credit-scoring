"""Temporal cohorts and leakage-safe message-passing views."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from graph_ml.data import GraphBuildResult


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Configuration for the v1 impairment evaluation cohort.

    Args:
        analysis_date: Last date on which outcomes are observed. Instruments
            after this date are outside the evaluation snapshot.
        uid_column: Stable identifier used to align the source table to graph
            tensor rows.
        open_column: Boolean final-snapshot open/closed state. An open negative
            is right-censored; a positive event is observed even if still open.
    """

    analysis_date: str | pd.Timestamp = "2018-12-18"
    uid_column: str = "uid"
    open_column: str = "is_open"


@dataclass(frozen=True)
class TemporalEvaluationSplit:
    """Boolean masks aligned to instrument-node rows in a graph.

    ``train_mask`` and ``test_mask`` contain only label-mature instruments.
    ``seen_test_mask`` and ``cold_start_test_mask`` partition ``test_mask``.
    """

    cutoff: pd.Timestamp
    analysis_date: pd.Timestamp
    pre_cutoff_mask: Tensor
    observed_mask: Tensor
    mature_mask: Tensor
    censored_mask: Tensor
    train_mask: Tensor
    test_mask: Tensor
    seen_test_mask: Tensor
    cold_start_test_mask: Tensor

    def summary(self) -> dict[str, int | float | str]:
        """Return aggregate cohort counts and rates for reporting."""

        pre_count = int(self.pre_cutoff_mask.sum())
        post_count = int((self.observed_mask & ~self.pre_cutoff_mask).sum())
        test_count = int(self.test_mask.sum())
        cold_count = int(self.cold_start_test_mask.sum())
        return {
            "cutoff": self.cutoff.date().isoformat(),
            "analysis_date": self.analysis_date.date().isoformat(),
            "pre_cutoff_instruments": pre_count,
            "post_cutoff_instruments": post_count,
            "mature_train_instruments": int(self.train_mask.sum()),
            "mature_test_instruments": test_count,
            "censored_open_negatives": int(self.censored_mask.sum()),
            "seen_test_instruments": int(self.seen_test_mask.sum()),
            "cold_start_test_instruments": cold_count,
            "cold_start_test_rate": cold_count / test_count if test_count else 0.0,
        }


@dataclass(frozen=True)
class TemporalGraphViews:
    """Training and inference graphs with different temporal edge visibility.

    The training graph physically contains only pre-cutoff instruments and
    their companies, preventing post-cutoff tensors from affecting global
    operations such as batch normalization. The inference graph contains all
    observed instruments, but only pre-cutoff instruments can send messages
    into company representations. Index tensors map local instrument rows back
    to the original graph.
    """

    training: HeteroData
    inference: HeteroData
    training_instrument_indices: Tensor
    inference_instrument_indices: Tensor
    training_supervision_mask: Tensor
    inference_test_mask: Tensor
    inference_seen_test_mask: Tensor
    inference_cold_start_test_mask: Tensor


def build_temporal_evaluation_split(
    instruments: pd.DataFrame,
    graph_result: GraphBuildResult,
    config: TemporalSplitConfig | None = None,
) -> TemporalEvaluationSplit:
    """Create mature temporal and cold-start masks aligned to graph nodes.

    For impairment, maturity means that the event is observed (positive label)
    or the instrument is closed. Open negatives remain right-censored because
    they may still impair after the dataset snapshot.
    """

    config = config or TemporalSplitConfig()
    graph = graph_result.graph
    metadata = graph_result.metadata
    analysis_date = _normalize_date(config.analysis_date, "analysis_date")
    if analysis_date <= metadata.cutoff:
        raise ValueError("analysis_date must be later than the graph cutoff")

    _validate_graph(graph)
    is_open = _aligned_boolean_column(
        instruments,
        uids=metadata.instrument_uids,
        uid_column=config.uid_column,
        value_column=config.open_column,
    )
    labels = graph["instrument"].y.to(dtype=torch.bool)
    invoice_dates = graph["instrument"].invoice_date
    analysis_day = _timestamp_to_epoch_day(analysis_date)

    observed = invoice_dates <= analysis_day
    pre_cutoff = graph["instrument"].pre_cutoff_mask.to(dtype=torch.bool)
    if torch.any(pre_cutoff & ~observed):
        raise ValueError("Pre-cutoff instruments cannot fall after analysis_date")

    mature = observed & (labels | ~is_open)
    censored = observed & is_open & ~labels
    train = pre_cutoff & mature
    test = ~pre_cutoff & mature

    cold_start = _cold_start_instruments(graph, pre_cutoff)
    cold_start_test = test & cold_start
    seen_test = test & ~cold_start

    return TemporalEvaluationSplit(
        cutoff=metadata.cutoff,
        analysis_date=analysis_date,
        pre_cutoff_mask=pre_cutoff,
        observed_mask=observed,
        mature_mask=mature,
        censored_mask=censored,
        train_mask=train,
        test_mask=test,
        seen_test_mask=seen_test,
        cold_start_test_mask=cold_start_test,
    )


def build_temporal_graph_views(
    graph: HeteroData,
    split: TemporalEvaluationSplit,
) -> TemporalGraphViews:
    """Build edge-filtered graphs for leakage-safe training and inference.

    Each view is node-induced for its time range. The returned index tensors
    preserve the mapping to original graph rows, while local masks are ready
    for model training and subgroup evaluation.
    """

    _validate_graph(graph)
    num_instruments = graph["instrument"].num_nodes
    for name, mask in (
        ("pre_cutoff_mask", split.pre_cutoff_mask),
        ("observed_mask", split.observed_mask),
    ):
        if mask.dtype != torch.bool or mask.numel() != num_instruments:
            raise ValueError(f"{name} must be a boolean mask for every instrument")

    training_instruments = torch.where(split.pre_cutoff_mask)[0]
    inference_instruments = torch.where(split.observed_mask)[0]
    training_companies = _companies_connected_to(graph, split.pre_cutoff_mask)
    inference_companies = _companies_connected_to(graph, split.observed_mask)

    training = graph.subgraph(
        {"instrument": training_instruments, "company": training_companies}
    )
    inference = graph.subgraph(
        {"instrument": inference_instruments, "company": inference_companies}
    )

    local_pre_cutoff = split.pre_cutoff_mask[inference_instruments]
    for edge_type in inference.edge_types:
        source_type, _, destination_type = edge_type
        if source_type == "instrument":
            edge_index = inference[edge_type].edge_index
            inference[edge_type].edge_index = edge_index[
                :, local_pre_cutoff[edge_index[0]]
            ]
        elif destination_type != "instrument":
            raise ValueError(f"Expected an instrument endpoint in relation {edge_type}")

    training.validate(raise_on_error=True)
    inference.validate(raise_on_error=True)
    return TemporalGraphViews(
        training=training,
        inference=inference,
        training_instrument_indices=training_instruments,
        inference_instrument_indices=inference_instruments,
        training_supervision_mask=split.train_mask[training_instruments],
        inference_test_mask=split.test_mask[inference_instruments],
        inference_seen_test_mask=split.seen_test_mask[inference_instruments],
        inference_cold_start_test_mask=split.cold_start_test_mask[
            inference_instruments
        ],
    )


def _normalize_date(value: str | pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is not None:
        raise ValueError(f"{name} must be timezone-naive")
    return timestamp.normalize()


def _timestamp_to_epoch_day(timestamp: pd.Timestamp) -> int:
    return int(timestamp.to_datetime64().astype("datetime64[D]").astype(int))


def _validate_graph(graph: HeteroData) -> None:
    graph.validate(raise_on_error=True)
    required_attributes = {"y", "invoice_date", "pre_cutoff_mask"}
    missing = required_attributes - set(graph["instrument"].keys())
    if missing:
        raise ValueError(
            f"Instrument nodes are missing required attributes: {', '.join(sorted(missing))}"
        )


def _aligned_boolean_column(
    instruments: pd.DataFrame,
    *,
    uids: tuple[str, ...],
    uid_column: str,
    value_column: str,
) -> Tensor:
    missing_columns = {uid_column, value_column} - set(instruments.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing_columns))}"
        )

    frame = instruments[[uid_column, value_column]].copy().reset_index(drop=True)
    if (
        frame[uid_column].isna().any()
        or frame[uid_column].astype(str).duplicated().any()
    ):
        raise ValueError("Instrument UIDs must be present and unique")
    if (
        frame[value_column].isna().any()
        or not frame[value_column].isin([False, True, 0, 1]).all()
    ):
        raise ValueError(f"{value_column} must contain boolean values")

    values_by_uid = frame.set_index(frame[uid_column].astype(str))[value_column]
    graph_uid_set = set(uids)
    source_uid_set = set(values_by_uid.index)
    if graph_uid_set != source_uid_set:
        missing = len(graph_uid_set - source_uid_set)
        extra = len(source_uid_set - graph_uid_set)
        raise ValueError(
            f"Instrument UID mismatch between table and graph: {missing} missing, {extra} extra"
        )
    aligned = values_by_uid.loc[list(uids)].to_numpy(dtype=bool, copy=True)
    return torch.from_numpy(aligned)


def _cold_start_instruments(graph: HeteroData, pre_cutoff_mask: Tensor) -> Tensor:
    sold_by = graph["instrument", "sold_by", "company"].edge_index
    owed_by = graph["instrument", "owed_by", "company"].edge_index
    seen_companies = torch.zeros(graph["company"].num_nodes, dtype=torch.bool)
    for edge_index in (sold_by, owed_by):
        historical_edges = pre_cutoff_mask[edge_index[0]]
        seen_companies[edge_index[1, historical_edges]] = True

    cold_start = torch.zeros(graph["instrument"].num_nodes, dtype=torch.bool)
    for edge_index in (sold_by, owed_by):
        cold_edges = ~seen_companies[edge_index[1]]
        cold_start[edge_index[0, cold_edges]] = True
    return cold_start


def _companies_connected_to(graph: HeteroData, instrument_mask: Tensor) -> Tensor:
    company_indices = []
    for edge_type in (
        ("instrument", "sold_by", "company"),
        ("instrument", "owed_by", "company"),
    ):
        edge_index = graph[edge_type].edge_index
        company_indices.append(edge_index[1, instrument_mask[edge_index[0]]])
    return torch.unique(torch.cat(company_indices), sorted=True)

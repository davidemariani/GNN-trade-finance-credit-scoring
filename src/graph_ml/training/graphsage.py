"""Deterministic temporal training protocol for relation-aware GraphSAGE."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import random

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import HeteroData

from graph_ml.data import GraphBuildResult
from graph_ml.evaluation import (
    TemporalEvaluationSplit,
    build_temporal_graph_views,
    compute_binary_metrics,
)
from graph_ml.models import HeteroGraphSAGE


@dataclass(frozen=True)
class GraphSAGETrainingConfig:
    """Reproducible settings for the first full-batch GraphSAGE experiment."""

    hidden_channels: int = 64
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    max_epochs: int = 250
    patience: int = 30
    minimum_improvement: float = 1e-4
    seed: int = 42


@dataclass(frozen=True)
class GraphSAGETrainingRun:
    """Scores, model, and fit metadata from one temporal GraphSAGE run."""

    model: HeteroGraphSAGE
    scores: np.ndarray
    instrument_embeddings: np.ndarray
    best_epoch: int
    best_validation_pr_auc: float
    validation_start_date: pd.Timestamp
    search_train_losses: tuple[float, ...]
    search_validation_pr_aucs: tuple[float, ...]
    parameter_count: int
    seed: int


def fit_hetero_graphsage(
    graph_result: GraphBuildResult,
    split: TemporalEvaluationSplit,
    config: GraphSAGETrainingConfig | None = None,
) -> GraphSAGETrainingRun:
    """Select epoch count temporally, refit on all training labels, and score test nodes.

    The search model fits the earliest portion of mature pre-cutoff labels and
    selects an epoch using average precision on the latest portion. A new model
    is then initialized with the same seed and fitted for that fixed number of
    epochs on every mature training label. Final inference uses the leakage-safe
    view in which post-cutoff instruments cannot send messages into companies.
    All reported runs use CPU for deterministic reproducibility.
    """

    config = config or GraphSAGETrainingConfig()
    _validate_config(config)
    views = build_temporal_graph_views(graph_result.graph, split)
    training_graph = views.training.cpu()
    inference_graph = views.inference.cpu()
    labels = training_graph["instrument"].y.float()
    fit_mask, validation_mask, validation_start = _temporal_validation_masks(
        training_graph["instrument"].invoice_date,
        views.training_supervision_mask,
        config.validation_fraction,
    )
    _require_both_classes(labels, fit_mask, "fit")
    _require_both_classes(labels, validation_mask, "validation")

    search_model = _new_model(graph_result, config)
    optimizer = torch.optim.Adam(
        search_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    positive_weight = _positive_weight(labels[fit_mask])
    best_pr_auc = -np.inf
    best_epoch = 1
    stale_epochs = 0
    losses: list[float] = []
    validation_pr_aucs: list[float] = []
    for epoch in range(1, config.max_epochs + 1):
        loss = _training_step(
            search_model,
            training_graph,
            labels,
            fit_mask,
            optimizer,
            positive_weight,
        )
        validation_pr_auc = _average_precision(
            search_model, training_graph, labels, validation_mask
        )
        losses.append(loss)
        validation_pr_aucs.append(validation_pr_auc)
        if validation_pr_auc > best_pr_auc + config.minimum_improvement:
            best_pr_auc = validation_pr_auc
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    final_model = _new_model(graph_result, config)
    final_optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    supervision_mask = views.training_supervision_mask.cpu()
    final_positive_weight = _positive_weight(labels[supervision_mask])
    for _ in range(best_epoch):
        _training_step(
            final_model,
            training_graph,
            labels,
            supervision_mask,
            final_optimizer,
            final_positive_weight,
        )

    final_model.eval()
    with torch.no_grad():
        hidden = final_model.encode(
            inference_graph.x_dict, inference_graph.edge_index_dict
        )["instrument"]
        local_scores = torch.sigmoid(final_model.classifier(hidden).squeeze(-1))
    instrument_count = graph_result.graph["instrument"].num_nodes
    scores = np.full(instrument_count, np.nan, dtype=np.float64)
    embeddings = np.full(
        (instrument_count, config.hidden_channels), np.nan, dtype=np.float32
    )
    inference_indices = views.inference_instrument_indices.cpu().numpy()
    scores[inference_indices] = local_scores.cpu().numpy()
    embeddings[inference_indices] = hidden.cpu().numpy()

    return GraphSAGETrainingRun(
        model=final_model,
        scores=scores,
        instrument_embeddings=embeddings,
        best_epoch=best_epoch,
        best_validation_pr_auc=float(best_pr_auc),
        validation_start_date=_epoch_day_to_timestamp(validation_start),
        search_train_losses=tuple(losses),
        search_validation_pr_aucs=tuple(validation_pr_aucs),
        parameter_count=sum(
            parameter.numel() for parameter in final_model.parameters()
        ),
        seed=config.seed,
    )


def evaluate_graphsage_run(
    run: GraphSAGETrainingRun,
    labels: np.ndarray,
    split: TemporalEvaluationSplit,
    *,
    review_fraction: float = 0.05,
) -> pd.DataFrame:
    """Evaluate GraphSAGE overall and on seen/cold-start test cohorts."""

    if not 0 < review_fraction <= 1:
        raise ValueError("review_fraction must be in (0, 1]")
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(run.scores).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError("Run scores must align with labels")
    cohorts = {
        "test_all": split.test_mask.detach().cpu().numpy(),
        "test_seen": split.seen_test_mask.detach().cpu().numpy(),
        "test_cold_start": split.cold_start_test_mask.detach().cpu().numpy(),
    }
    rows = []
    for cohort_name, mask in cohorts.items():
        if not np.isfinite(scores[mask]).all():
            raise ValueError(f"Scores for {cohort_name} must be finite")
        cohort_count = int(mask.sum())
        if cohort_count == 0:
            raise ValueError(f"Cohort {cohort_name} is empty")
        metrics = compute_binary_metrics(
            labels[mask],
            scores[mask],
            top_k=max(1, ceil(review_fraction * cohort_count)),
        )
        rows.append(
            {
                "model": "hetero_graphsage",
                "cohort": cohort_name,
                "review_fraction": review_fraction,
                **metrics.as_dict(),
            }
        )
    return pd.DataFrame(rows)


def _new_model(
    graph_result: GraphBuildResult, config: GraphSAGETrainingConfig
) -> HeteroGraphSAGE:
    _seed_everything(config.seed)
    graph = graph_result.graph
    return HeteroGraphSAGE(
        graph["instrument"].x.shape[1],
        graph["company"].x.shape[1],
        hidden_channels=config.hidden_channels,
        dropout=config.dropout,
    ).cpu()


def _training_step(
    model: HeteroGraphSAGE,
    graph: HeteroData,
    labels: Tensor,
    mask: Tensor,
    optimizer: torch.optim.Optimizer,
    positive_weight: Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(graph.x_dict, graph.edge_index_dict)
    loss = F.binary_cross_entropy_with_logits(
        logits[mask], labels[mask], pos_weight=positive_weight
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _average_precision(
    model: HeteroGraphSAGE,
    graph: HeteroData,
    labels: Tensor,
    mask: Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(graph.x_dict, graph.edge_index_dict)[mask])
    return float(
        average_precision_score(labels[mask].cpu().numpy(), scores.cpu().numpy())
    )


def _temporal_validation_masks(
    invoice_days: Tensor,
    supervision_mask: Tensor,
    validation_fraction: float,
) -> tuple[Tensor, Tensor, int]:
    days = invoice_days.cpu().numpy()
    supervised = np.flatnonzero(supervision_mask.cpu().numpy())
    ordered = supervised[np.argsort(days[supervised], kind="stable")]
    candidate = int((1 - validation_fraction) * ordered.size)
    candidate = min(max(candidate, 1), ordered.size - 1)
    validation_start = int(days[ordered[candidate]])
    fit = supervision_mask.cpu() & torch.from_numpy(days < validation_start)
    validation = supervision_mask.cpu() & torch.from_numpy(days >= validation_start)
    if not fit.any() or not validation.any():
        raise ValueError("Temporal validation produced an empty partition")
    return fit, validation, validation_start


def _positive_weight(labels: Tensor) -> Tensor:
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("Class-weighted loss requires both classes")
    return (negatives / positives).reshape(())


def _require_both_classes(labels: Tensor, mask: Tensor, name: str) -> None:
    if torch.unique(labels[mask]).numel() != 2:
        raise ValueError(f"Temporal {name} partition must contain both classes")


def _validate_config(config: GraphSAGETrainingConfig) -> None:
    if config.hidden_channels < 1:
        raise ValueError("hidden_channels must be positive")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    if not 0 < config.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    if config.max_epochs < 1 or config.patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    if config.minimum_improvement < 0:
        raise ValueError("minimum_improvement must be non-negative")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _epoch_day_to_timestamp(epoch_day: int) -> pd.Timestamp:
    return pd.Timestamp(np.datetime64(epoch_day, "D"))

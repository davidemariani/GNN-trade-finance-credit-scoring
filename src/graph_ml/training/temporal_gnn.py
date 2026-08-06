"""Deterministic rolling-origin training for the causal temporal p90 GNN."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
import random
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
from torch import Tensor
from torch.nn import functional as F

from graph_ml.baselines import (
    PointInTimeFeatureFrame,
    fit_point_in_time_encoder,
    transform_point_in_time_instruments,
)
from graph_ml.data import (
    build_bounded_temporal_relation_context,
    build_temporal_relation_context,
)
from graph_ml.evaluation import (
    LabelAvailability,
    PointInTimeFold,
    compute_binary_metrics,
)
from graph_ml.models import TemporalRoleGNN


@dataclass(frozen=True)
class TemporalGNNTrainingConfig:
    """Frozen settings for the first role-aware temporal message-passing model."""

    hidden_channels: int = 64
    dropout: float = 0.2
    half_life_days: float = 180.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 25
    minimum_improvement: float = 1e-4
    seed: int = 42
    use_relation_context: bool = True
    relation_mode: Literal["separate", "shared"] = "separate"
    max_recent_events: int | None = None


@dataclass(frozen=True)
class TemporalGNNTrainingRun:
    """Scores, embeddings, and validation trace from one temporal GNN run."""

    model: TemporalRoleGNN
    scores: np.ndarray
    embeddings: np.ndarray
    best_epoch: int
    best_validation_pr_auc: float
    search_train_losses: tuple[float, ...]
    search_validation_pr_aucs: tuple[float, ...]
    parameter_count: int
    instrument_feature_names: tuple[str, ...]
    seed: int
    max_recent_events: int | None


def fit_temporal_role_gnn(
    instruments: pd.DataFrame,
    features: PointInTimeFeatureFrame,
    availability: LabelAvailability,
    fold: PointInTimeFold,
    config: TemporalGNNTrainingConfig | None = None,
) -> TemporalGNNTrainingRun:
    """Select epochs on rolling validation and refit on all legal p90 labels."""

    config = config or TemporalGNNTrainingConfig()
    _validate_config(config)
    labels = torch.from_numpy(availability.labels.astype(np.float32))
    train_mask = torch.from_numpy(fold.train_mask)
    validation_mask = torch.from_numpy(fold.validation_mask)
    refit_mask = torch.from_numpy(fold.refit_mask)
    _require_both_classes(labels, train_mask, "train")
    _require_both_classes(labels, validation_mask, "validation")
    _require_both_classes(labels, refit_mask, "refit")

    search_encoder = fit_point_in_time_encoder(features, fold.train_mask)
    search_values, _ = transform_point_in_time_instruments(features, search_encoder)
    search_tensors = _temporal_tensors(
        instruments,
        search_values,
        config.half_life_days,
        config.max_recent_events,
    )
    search_model = _new_model(search_values.shape[1], config)
    optimizer = torch.optim.Adam(
        search_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    positive_weight = _positive_weight(labels[train_mask])
    best_pr_auc = -np.inf
    best_epoch = 1
    stale_epochs = 0
    losses: list[float] = []
    validation_scores: list[float] = []
    for epoch in range(1, config.max_epochs + 1):
        loss = _training_step(
            search_model,
            search_tensors,
            labels,
            train_mask,
            optimizer,
            positive_weight,
        )
        score = _average_precision(
            search_model, search_tensors, labels, validation_mask
        )
        losses.append(loss)
        validation_scores.append(score)
        if score > best_pr_auc + config.minimum_improvement:
            best_pr_auc = score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    final_encoder = fit_point_in_time_encoder(features, fold.refit_mask)
    final_values, final_feature_names = transform_point_in_time_instruments(
        features, final_encoder
    )
    final_tensors = _temporal_tensors(
        instruments,
        final_values,
        config.half_life_days,
        config.max_recent_events,
    )
    final_model = _new_model(final_values.shape[1], config)
    final_optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    final_positive_weight = _positive_weight(labels[refit_mask])
    for _ in range(best_epoch):
        _training_step(
            final_model,
            final_tensors,
            labels,
            refit_mask,
            final_optimizer,
            final_positive_weight,
        )

    final_model.eval()
    with torch.no_grad():
        hidden = final_model.encode(*final_tensors)
        scores = torch.sigmoid(final_model.classifier(hidden).squeeze(-1))
    return TemporalGNNTrainingRun(
        model=final_model,
        scores=scores.numpy().astype(np.float64),
        embeddings=hidden.numpy(),
        best_epoch=best_epoch,
        best_validation_pr_auc=float(best_pr_auc),
        search_train_losses=tuple(losses),
        search_validation_pr_aucs=tuple(validation_scores),
        parameter_count=sum(
            parameter.numel() for parameter in final_model.parameters()
        ),
        instrument_feature_names=final_feature_names,
        seed=config.seed,
        max_recent_events=config.max_recent_events,
    )


def evaluate_temporal_gnn_run(
    run: TemporalGNNTrainingRun,
    availability: LabelAvailability,
    cohorts: Mapping[str, np.ndarray],
    *,
    review_fraction: float = 0.05,
) -> pd.DataFrame:
    """Evaluate temporal GNN scores on named point-in-time cohorts."""

    if not 0 < review_fraction <= 1:
        raise ValueError("review_fraction must be in (0, 1]")
    labels = availability.labels.astype(np.int64)
    rows = []
    for cohort, mask in cohorts.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != labels.shape or not mask.any():
            raise ValueError(f"Cohort {cohort} must select aligned rows")
        metrics = compute_binary_metrics(
            labels[mask],
            run.scores[mask],
            top_k=max(1, ceil(review_fraction * int(mask.sum()))),
        )
        rows.append(
            {
                "model": (
                    "root_only_neural"
                    if not run.model.use_relation_context
                    else (
                        f"temporal_role_gnn_recent_k{run.max_recent_events}"
                        if run.max_recent_events is not None
                        else (
                            "temporal_role_gnn"
                            if run.model.relation_mode == "separate"
                            else "temporal_role_gnn_shared"
                        )
                    )
                ),
                "cohort": cohort,
                "review_fraction": review_fraction,
                **metrics.as_dict(),
            }
        )
    return pd.DataFrame(rows)


def _temporal_tensors(
    instruments: pd.DataFrame,
    values: np.ndarray,
    half_life_days: float,
    max_recent_events: int | None,
) -> tuple[Tensor, Tensor, Tensor]:
    context = (
        build_temporal_relation_context(
            instruments, values, half_life_days=half_life_days
        )
        if max_recent_events is None
        else build_bounded_temporal_relation_context(
            instruments,
            values,
            max_events=max_recent_events,
            half_life_days=half_life_days,
        )
    )
    return (
        torch.from_numpy(values.astype(np.float32)),
        torch.from_numpy(context.values),
        torch.from_numpy(context.metadata),
    )


def _new_model(
    instrument_channels: int, config: TemporalGNNTrainingConfig
) -> TemporalRoleGNN:
    _seed_everything(config.seed)
    return TemporalRoleGNN(
        instrument_channels,
        hidden_channels=config.hidden_channels,
        dropout=config.dropout,
        use_relation_context=config.use_relation_context,
        relation_mode=config.relation_mode,
    ).cpu()


def _training_step(
    model: TemporalRoleGNN,
    tensors: tuple[Tensor, Tensor, Tensor],
    labels: Tensor,
    mask: Tensor,
    optimizer: torch.optim.Optimizer,
    positive_weight: Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(*tensors)
    loss = F.binary_cross_entropy_with_logits(
        logits[mask], labels[mask], pos_weight=positive_weight
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _average_precision(
    model: TemporalRoleGNN,
    tensors: tuple[Tensor, Tensor, Tensor],
    labels: Tensor,
    mask: Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(*tensors)[mask])
    return float(average_precision_score(labels[mask].numpy(), scores.numpy()))


def _positive_weight(labels: Tensor) -> Tensor:
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("Class-weighted loss requires both classes")
    return (negatives / positives).reshape(())


def _require_both_classes(labels: Tensor, mask: Tensor, name: str) -> None:
    if mask.shape != labels.shape or torch.unique(labels[mask]).numel() != 2:
        raise ValueError(
            f"Temporal {name} partition must align and contain both classes"
        )


def _validate_config(config: TemporalGNNTrainingConfig) -> None:
    if (
        config.hidden_channels < 1
        or np.isnan(config.half_life_days)
        or config.half_life_days <= 0
    ):
        raise ValueError("Hidden channels and half-life must be positive or infinity")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    if config.max_epochs < 1 or config.patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    if config.minimum_improvement < 0:
        raise ValueError("minimum_improvement must be non-negative")
    if config.relation_mode not in {"separate", "shared"}:
        raise ValueError("relation_mode must be separate or shared")
    if config.max_recent_events is not None and config.max_recent_events < 1:
        raise ValueError("max_recent_events must be positive when provided")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

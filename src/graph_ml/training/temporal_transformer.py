"""Rolling-origin training for the causal temporal graph Transformer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
import random

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
from torch.nn import functional as F

from graph_ml.baselines import (
    PointInTimeFeatureFrame,
    fit_point_in_time_encoder,
    transform_point_in_time_instruments,
)
from graph_ml.data import build_temporal_event_sequences
from graph_ml.evaluation import (
    LabelAvailability,
    PointInTimeFold,
    compute_binary_metrics,
)
from graph_ml.models import TemporalGraphTransformer


@dataclass(frozen=True)
class TemporalTransformerTrainingConfig:
    """Frozen optimization and bounded-attention settings."""

    hidden_channels: int = 64
    attention_heads: int = 4
    max_events: int = 8
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 2_048
    max_epochs: int = 100
    patience: int = 15
    minimum_improvement: float = 1e-4
    seed: int = 42


@dataclass(frozen=True)
class TemporalTransformerTrainingRun:
    """Scores and selection trace from one causal attention run."""

    model: TemporalGraphTransformer
    scores: np.ndarray
    best_epoch: int
    best_validation_pr_auc: float
    search_train_losses: tuple[float, ...]
    search_validation_pr_aucs: tuple[float, ...]
    parameter_count: int
    seed: int


def fit_temporal_graph_transformer(
    instruments: pd.DataFrame,
    features: PointInTimeFeatureFrame,
    availability: LabelAvailability,
    fold: PointInTimeFold,
    config: TemporalTransformerTrainingConfig | None = None,
) -> TemporalTransformerTrainingRun:
    """Select epochs on rolling validation and refit on all legal labels."""

    config = config or TemporalTransformerTrainingConfig()
    _validate_config(config)
    labels = torch.from_numpy(availability.labels.astype(np.float32))
    masks = {
        "train": torch.from_numpy(fold.train_mask),
        "validation": torch.from_numpy(fold.validation_mask),
        "refit": torch.from_numpy(fold.refit_mask),
    }
    for name, mask in masks.items():
        if mask.shape != labels.shape or torch.unique(labels[mask]).numel() != 2:
            raise ValueError(f"Transformer {name} partition must contain both classes")

    search_tensors = _fold_tensors(instruments, features, fold.train_mask, config)
    search_model = _new_model(search_tensors[0].shape[1], config)
    optimizer = torch.optim.Adam(
        search_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.seed)
    best_pr_auc = -np.inf
    best_epoch = 1
    stale = 0
    losses: list[float] = []
    validation_scores: list[float] = []
    for epoch in range(1, config.max_epochs + 1):
        loss = _train_epoch(
            search_model,
            search_tensors,
            labels,
            masks["train"],
            optimizer,
            config.batch_size,
            generator,
        )
        score = _average_precision(
            search_model,
            search_tensors,
            labels,
            masks["validation"],
            config.batch_size,
        )
        losses.append(loss)
        validation_scores.append(score)
        if score > best_pr_auc + config.minimum_improvement:
            best_pr_auc = score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    final_tensors = _fold_tensors(instruments, features, fold.refit_mask, config)
    final_model = _new_model(final_tensors[0].shape[1], config)
    final_optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    final_generator = torch.Generator().manual_seed(config.seed)
    for _ in range(best_epoch):
        _train_epoch(
            final_model,
            final_tensors,
            labels,
            masks["refit"],
            final_optimizer,
            config.batch_size,
            final_generator,
        )
    scores = _score_batches(final_model, final_tensors, config.batch_size)
    return TemporalTransformerTrainingRun(
        model=final_model,
        scores=scores,
        best_epoch=best_epoch,
        best_validation_pr_auc=float(best_pr_auc),
        search_train_losses=tuple(losses),
        search_validation_pr_aucs=tuple(validation_scores),
        parameter_count=sum(
            parameter.numel() for parameter in final_model.parameters()
        ),
        seed=config.seed,
    )


def evaluate_temporal_transformer_run(
    run: TemporalTransformerTrainingRun,
    availability: LabelAvailability,
    cohorts: Mapping[str, np.ndarray],
    *,
    review_fraction: float = 0.05,
) -> pd.DataFrame:
    """Evaluate attention scores on named point-in-time cohorts."""

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
                "model": "temporal_graph_transformer",
                "cohort": cohort,
                "review_fraction": review_fraction,
                **metrics.as_dict(),
            }
        )
    return pd.DataFrame(rows)


def _fold_tensors(instruments, features, fit_mask, config):
    encoder = fit_point_in_time_encoder(features, fit_mask)
    current, _ = transform_point_in_time_instruments(features, encoder)
    sequences = build_temporal_event_sequences(
        instruments, current, max_events=config.max_events
    )
    return (
        torch.from_numpy(current.astype(np.float32)),
        torch.from_numpy(sequences.values),
        torch.from_numpy(sequences.age_days),
        torch.from_numpy(sequences.valid_mask),
    )


def _new_model(channels, config):
    _seed_everything(config.seed)
    return TemporalGraphTransformer(
        channels,
        hidden_channels=config.hidden_channels,
        attention_heads=config.attention_heads,
        dropout=config.dropout,
    ).cpu()


def _train_epoch(model, tensors, labels, mask, optimizer, batch_size, generator):
    model.train()
    indices = torch.nonzero(mask, as_tuple=True)[0]
    indices = indices[torch.randperm(indices.numel(), generator=generator)]
    selected_labels = labels[mask]
    positive_weight = (
        (selected_labels.numel() - selected_labels.sum()) / selected_labels.sum()
    ).reshape(())
    total = 0.0
    for batch in indices.split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        logits = model(*(tensor[batch] for tensor in tensors))
        loss = F.binary_cross_entropy_with_logits(
            logits, labels[batch], pos_weight=positive_weight
        )
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * batch.numel()
    return total / indices.numel()


def _average_precision(model, tensors, labels, mask, batch_size):
    scores = _score_batches(
        model, tensors, batch_size, torch.nonzero(mask, as_tuple=True)[0]
    )
    return float(average_precision_score(labels[mask].numpy(), scores))


def _score_batches(model, tensors, batch_size, indices=None):
    model.eval()
    if indices is None:
        indices = torch.arange(tensors[0].shape[0])
    batches = []
    with torch.no_grad():
        for batch in indices.split(batch_size):
            batches.append(torch.sigmoid(model(*(tensor[batch] for tensor in tensors))))
    return torch.cat(batches).numpy().astype(np.float64)


def _validate_config(config):
    if (
        config.hidden_channels < 1
        or config.attention_heads < 1
        or config.max_events < 1
    ):
        raise ValueError("Channels, heads, and max_events must be positive")
    if config.hidden_channels % config.attention_heads:
        raise ValueError("hidden_channels must be divisible by attention_heads")
    if not 0 <= config.dropout < 1 or config.batch_size < 1:
        raise ValueError("dropout or batch_size is invalid")
    if config.max_epochs < 1 or config.patience < 1:
        raise ValueError("max_epochs and patience must be positive")


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

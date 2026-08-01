"""Build the leakage-aware heterogeneous trade-finance graph.

The graph has two node types (``instrument`` and ``company``) and role-typed
instrument/company edges.  See ``wiki/this-project/graph-design.md`` for the
modelling rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

_UNKNOWN_CATEGORY = "__unknown__"
_MISSING_CATEGORY = "__missing__"

_NUMERIC_FEATURE_NAMES = (
    "log_invoice_amount",
    "purchase_to_invoice_ratio",
    "payment_term_days",
    "input_lag_days",
)
_CATEGORICAL_COLUMNS = ("currency", "factoring_type")
_REQUIRED_COLUMNS = frozenset(
    {
        "invoice_date",
        "due_date",
        "input_date",
        "invoice_amount",
        "purchase_amount",
        "currency",
        "factoring_type",
    }
)


@dataclass(frozen=True)
class GraphBuildConfig:
    """Configuration for a cutoff-specific graph build.

    Args:
        cutoff: Information cutoff. Company history and feature fitting use
            instruments with ``invoice_date < cutoff`` only.
        label_column: Binary instrument-level target stored as
            ``graph["instrument"].y``.
        uid_column: Stable instrument identifier.
        seller_name_column: Company name in the seller/customer role.
        buyer_name_column: Company name in the buyer/debtor role.
    """

    cutoff: str | pd.Timestamp = "2018-04-30"
    label_column: str = "has_impairment1"
    uid_column: str = "uid"
    seller_name_column: str = "customer_name_1"
    buyer_name_column: str = "debtor_name_1"


@dataclass(frozen=True)
class GraphMetadata:
    """Human-readable mappings and feature names for a constructed graph."""

    cutoff: pd.Timestamp
    instrument_uids: tuple[str, ...]
    company_keys: tuple[str, ...]
    company_display_names: tuple[str, ...]
    instrument_feature_names: tuple[str, ...]
    company_feature_names: tuple[str, ...]

    @property
    def instrument_index(self) -> dict[str, int]:
        """Map each stable instrument UID to its tensor row index."""

        return {uid: index for index, uid in enumerate(self.instrument_uids)}

    @property
    def company_index(self) -> dict[str, int]:
        """Map each canonical company key to its tensor row index."""

        return {key: index for index, key in enumerate(self.company_keys)}


@dataclass(frozen=True)
class GraphBuildResult:
    """A PyG graph together with the mappings needed to interpret its tensors."""

    graph: HeteroData
    metadata: GraphMetadata


@dataclass(frozen=True)
class _FeatureEncoder:
    numeric_medians: np.ndarray
    numeric_means: np.ndarray
    numeric_scales: np.ndarray
    categories: tuple[tuple[str, ...], ...]

    @classmethod
    def fit(cls, instruments: pd.DataFrame) -> _FeatureEncoder:
        numeric = _raw_numeric_features(instruments)
        medians = np.nanmedian(numeric, axis=0)
        if np.isnan(medians).any():
            raise ValueError(
                "Every numeric feature needs at least one pre-cutoff value"
            )
        filled = np.where(np.isnan(numeric), medians, numeric)
        means = filled.mean(axis=0)
        scales = filled.std(axis=0)
        scales[scales == 0] = 1.0

        categories = []
        for column in _CATEGORICAL_COLUMNS:
            observed = set(_categorical_values(instruments[column]))
            observed.discard(_UNKNOWN_CATEGORY)
            categories.append(tuple(sorted(observed)) + (_UNKNOWN_CATEGORY,))

        return cls(
            numeric_medians=medians,
            numeric_means=means,
            numeric_scales=scales,
            categories=tuple(categories),
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = list(_NUMERIC_FEATURE_NAMES)
        for column, categories in zip(
            _CATEGORICAL_COLUMNS, self.categories, strict=True
        ):
            names.extend(f"{column}={category}" for category in categories)
        return tuple(names)

    def transform(self, instruments: pd.DataFrame) -> np.ndarray:
        numeric = _raw_numeric_features(instruments)
        numeric = np.where(np.isnan(numeric), self.numeric_medians, numeric)
        numeric = (numeric - self.numeric_means) / self.numeric_scales

        encoded_parts = [numeric]
        for column, categories in zip(
            _CATEGORICAL_COLUMNS, self.categories, strict=True
        ):
            values = _categorical_values(instruments[column])
            known = set(categories[:-1])
            values = np.array(
                [value if value in known else _UNKNOWN_CATEGORY for value in values]
            )
            encoded_parts.append(
                np.column_stack([values == category for category in categories])
            )
        return np.column_stack(encoded_parts).astype(np.float32)


def canonicalize_company_name(name: object) -> str:
    """Return a stable company identity key from a display name.

    Unicode compatibility normalization, case folding, trimming, and whitespace
    collapsing are intentionally conservative: punctuation is retained so that
    distinct legal names are not silently merged.
    """

    if pd.isna(name):
        raise ValueError("Company names cannot be missing")
    normalized = unicodedata.normalize("NFKC", str(name)).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise ValueError("Company names cannot be blank")
    return normalized


def build_trade_finance_graph(
    instruments: pd.DataFrame,
    config: GraphBuildConfig | None = None,
) -> GraphBuildResult:
    """Build a cutoff-specific company/instrument ``HeteroData`` graph.

    Instrument feature rows have shape ``[num_instruments, F_instrument]``.
    Company feature rows have shape ``[num_companies, F_company]``. Company
    history is derived only from pre-cutoff instruments; a company first seen
    after the cutoff receives an all-zero history vector.

    Both forward and reverse role-typed edge indices have shape ``[2, E]``.
    Labels are stored only on instrument nodes.
    """

    config = config or GraphBuildConfig()
    cutoff = _normalize_cutoff(config.cutoff)
    frame = _prepare_frame(instruments, config)
    pre_cutoff_mask = frame["invoice_date"] < cutoff
    if not pre_cutoff_mask.any():
        raise ValueError("At least one instrument must precede the cutoff")

    encoder = _FeatureEncoder.fit(frame.loc[pre_cutoff_mask])
    instrument_features = encoder.transform(frame)

    seller_keys = frame[config.seller_name_column].map(canonicalize_company_name)
    buyer_keys = frame[config.buyer_name_column].map(canonicalize_company_name)
    company_keys = tuple(sorted(set(seller_keys) | set(buyer_keys)))
    company_index = {key: index for index, key in enumerate(company_keys)}

    display_names = _company_display_names(frame, config, company_keys)
    seller_indices = seller_keys.map(company_index).to_numpy(dtype=np.int64)
    buyer_indices = buyer_keys.map(company_index).to_numpy(dtype=np.int64)
    instrument_indices = np.arange(len(frame), dtype=np.int64)

    company_features, company_feature_names = _company_history_features(
        instrument_features=instrument_features,
        pre_cutoff_mask=pre_cutoff_mask.to_numpy(),
        seller_indices=seller_indices,
        buyer_indices=buyer_indices,
        num_companies=len(company_keys),
    )

    graph = HeteroData()
    graph["instrument"].x = torch.from_numpy(instrument_features)
    graph["instrument"].y = torch.from_numpy(
        frame[config.label_column].to_numpy(dtype=np.int64)
    )
    graph["instrument"].node_id = torch.arange(len(frame), dtype=torch.long)
    graph["instrument"].invoice_date = torch.from_numpy(
        frame["invoice_date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    )
    graph["instrument"].pre_cutoff_mask = torch.from_numpy(
        pre_cutoff_mask.to_numpy(copy=True)
    )

    graph["company"].x = torch.from_numpy(company_features)
    graph["company"].node_id = torch.arange(len(company_keys), dtype=torch.long)

    sold_by = torch.from_numpy(np.vstack((instrument_indices, seller_indices)))
    owed_by = torch.from_numpy(np.vstack((instrument_indices, buyer_indices)))
    graph["instrument", "sold_by", "company"].edge_index = sold_by
    graph["company", "sells", "instrument"].edge_index = sold_by.flip(0)
    graph["instrument", "owed_by", "company"].edge_index = owed_by
    graph["company", "owes", "instrument"].edge_index = owed_by.flip(0)
    graph.validate(raise_on_error=True)

    metadata = GraphMetadata(
        cutoff=cutoff,
        instrument_uids=tuple(frame[config.uid_column].astype(str)),
        company_keys=company_keys,
        company_display_names=display_names,
        instrument_feature_names=encoder.feature_names,
        company_feature_names=company_feature_names,
    )
    return GraphBuildResult(graph=graph, metadata=metadata)


def build_trade_finance_graph_from_parquet(
    path: str | Path,
    config: GraphBuildConfig | None = None,
) -> GraphBuildResult:
    """Load an instrument-level Parquet table and build its heterogeneous graph."""

    return build_trade_finance_graph(pd.read_parquet(path), config=config)


def _normalize_cutoff(value: str | pd.Timestamp) -> pd.Timestamp:
    cutoff = pd.Timestamp(value)
    if cutoff.tz is not None:
        raise ValueError("The cutoff must be timezone-naive, like invoice_date")
    return cutoff.normalize()


def _prepare_frame(instruments: pd.DataFrame, config: GraphBuildConfig) -> pd.DataFrame:
    required = _REQUIRED_COLUMNS | {
        config.uid_column,
        config.seller_name_column,
        config.buyer_name_column,
        config.label_column,
    }
    missing = sorted(required - set(instruments.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # The legacy table stores ``uid`` as both index name and column label.
    # Drop the external index so column operations remain unambiguous.
    frame = instruments.copy().reset_index(drop=True)
    for column in ("invoice_date", "due_date", "input_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"{column} contains missing or invalid dates")

    if frame[config.uid_column].isna().any():
        raise ValueError("Instrument UIDs cannot be missing")
    if frame[config.uid_column].astype(str).duplicated().any():
        raise ValueError("Instrument UIDs must be unique")
    for column in (config.seller_name_column, config.buyer_name_column):
        frame[column].map(canonicalize_company_name)

    labels = frame[config.label_column]
    if labels.isna().any() or not labels.isin([False, True, 0, 1]).all():
        raise ValueError(f"{config.label_column} must contain binary labels")

    return frame.sort_values(["invoice_date", config.uid_column]).reset_index(drop=True)


def _raw_numeric_features(instruments: pd.DataFrame) -> np.ndarray:
    invoice_amount = pd.to_numeric(
        instruments["invoice_amount"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    purchase_amount = pd.to_numeric(
        instruments["purchase_amount"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if np.nanmin(invoice_amount) < 0 or np.nanmin(purchase_amount) < 0:
        raise ValueError("Invoice and purchase amounts must be non-negative")

    purchase_ratio = np.divide(
        purchase_amount,
        invoice_amount,
        out=np.full_like(purchase_amount, np.nan),
        where=invoice_amount != 0,
    )
    payment_term = (instruments["due_date"] - instruments["invoice_date"]).dt.days
    input_lag = (instruments["input_date"] - instruments["invoice_date"]).dt.days
    return np.column_stack(
        (
            np.log1p(invoice_amount),
            purchase_ratio,
            payment_term.to_numpy(dtype=np.float64),
            input_lag.to_numpy(dtype=np.float64),
        )
    )


def _categorical_values(series: pd.Series) -> np.ndarray:
    return (
        series.astype("string")
        .fillna(_MISSING_CATEGORY)
        .str.strip()
        .replace("", _MISSING_CATEGORY)
        .to_numpy(dtype=str)
    )


def _company_display_names(
    frame: pd.DataFrame,
    config: GraphBuildConfig,
    company_keys: tuple[str, ...],
) -> tuple[str, ...]:
    display_by_key: dict[str, str] = {}
    for column in (config.seller_name_column, config.buyer_name_column):
        for display_name in frame[column]:
            key = canonicalize_company_name(display_name)
            display_by_key.setdefault(key, str(display_name).strip())
    return tuple(display_by_key[key] for key in company_keys)


def _company_history_features(
    *,
    instrument_features: np.ndarray,
    pre_cutoff_mask: np.ndarray,
    seller_indices: np.ndarray,
    buyer_indices: np.ndarray,
    num_companies: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    numeric = instrument_features[:, : len(_NUMERIC_FEATURE_NAMES)]
    role_parts = []
    names = []
    for role, indices in (("seller", seller_indices), ("buyer", buyer_indices)):
        history_indices = indices[pre_cutoff_mask]
        counts = np.bincount(history_indices, minlength=num_companies)
        sums = np.zeros((num_companies, numeric.shape[1]), dtype=np.float64)
        np.add.at(sums, history_indices, numeric[pre_cutoff_mask])
        means = np.divide(
            sums,
            counts[:, None],
            out=np.zeros_like(sums),
            where=counts[:, None] != 0,
        )
        role_parts.extend((np.log1p(counts[:, None]), means))
        names.append(f"{role}_history_log_count")
        names.extend(f"{role}_history_mean_{name}" for name in _NUMERIC_FEATURE_NAMES)

    return np.column_stack(role_parts).astype(np.float32), tuple(names)

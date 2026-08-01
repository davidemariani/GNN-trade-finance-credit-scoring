"""Aggregate EDA tables and static figures safe for portfolio publication."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from torch import Tensor


def class_balance_frame(labels: Tensor | np.ndarray) -> pd.DataFrame:
    """Return negative/positive counts and shares for binary labels."""

    values = np.asarray(labels.detach().cpu() if isinstance(labels, Tensor) else labels)
    values = values.reshape(-1)
    if values.size == 0 or not np.isin(values, [0, 1, False, True]).all():
        raise ValueError("labels must be a non-empty binary array")
    counts = np.bincount(values.astype(np.int64), minlength=2)
    return pd.DataFrame(
        {
            "outcome": ["not impaired", "impaired"],
            "count": counts,
            "share": counts / counts.sum(),
        }
    )


def temporal_volume_frame(
    invoice_dates: pd.Series,
    labels: pd.Series | np.ndarray,
    *,
    frequency: str = "MS",
) -> pd.DataFrame:
    """Aggregate instrument volume, impairments, and prevalence over time."""

    dates = pd.to_datetime(
        pd.Series(invoice_dates).reset_index(drop=True), errors="coerce"
    )
    outcomes = pd.Series(labels).reset_index(drop=True)
    if len(dates) == 0 or len(dates) != len(outcomes):
        raise ValueError("invoice_dates and labels must be non-empty and aligned")
    if dates.isna().any():
        raise ValueError("invoice_dates must contain valid dates")
    if outcomes.isna().any() or not outcomes.isin([0, 1, False, True]).all():
        raise ValueError("labels must be binary")

    frame = pd.DataFrame({"invoice_date": dates, "label": outcomes.astype(int)})
    grouped = frame.groupby(pd.Grouper(key="invoice_date", freq=frequency))["label"]
    result = grouped.agg(instrument_count="size", impairment_count="sum").reset_index()
    result["impairment_rate"] = result["impairment_count"] / result["instrument_count"]
    return result


def plot_class_balance(balance: pd.DataFrame) -> Figure:
    """Plot binary class shares and return the matplotlib figure."""

    _require_columns(balance, {"outcome", "count", "share"})
    figure, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(balance["outcome"], balance["share"], color=["#4472C4", "#C44E52"])
    axis.set(
        title="Impairment class balance", ylabel="Share of instruments", ylim=(0, 1)
    )
    axis.bar_label(bars, labels=[f"{value:.2%}" for value in balance["share"]])
    figure.tight_layout()
    return figure


def plot_temporal_volume(volume: pd.DataFrame) -> Figure:
    """Plot monthly instrument volume and impairment rate on aligned axes."""

    _require_columns(volume, {"invoice_date", "instrument_count", "impairment_rate"})
    figure, volume_axis = plt.subplots(figsize=(10, 4.5))
    rate_axis = volume_axis.twinx()
    volume_axis.plot(
        volume["invoice_date"],
        volume["instrument_count"],
        color="#4472C4",
        label="Volume",
    )
    rate_axis.plot(
        volume["invoice_date"],
        volume["impairment_rate"],
        color="#C44E52",
        label="Impairment rate",
    )
    volume_axis.set(
        title="Monthly instruments and impairment rate", ylabel="Instruments"
    )
    rate_axis.set(ylabel="Impairment rate")
    rate_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    lines = volume_axis.lines + rate_axis.lines
    volume_axis.legend(lines, [line.get_label() for line in lines], loc="upper left")
    figure.tight_layout()
    return figure


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

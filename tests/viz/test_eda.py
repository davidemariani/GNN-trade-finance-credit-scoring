import matplotlib.pyplot as plt
import pandas as pd
import pytest

from graph_ml.viz import (
    class_balance_frame,
    plot_class_balance,
    plot_temporal_volume,
    temporal_volume_frame,
)


def test_class_balance_returns_counts_and_shares():
    balance = class_balance_frame([0, 0, 0, 1])

    assert balance["count"].tolist() == [3, 1]
    assert balance["share"].tolist() == [0.75, 0.25]


def test_temporal_volume_aggregates_counts_and_rates():
    volume = temporal_volume_frame(
        pd.to_datetime(["2020-01-01", "2020-01-20", "2020-02-01"]),
        [0, 1, 1],
    )

    assert volume["instrument_count"].tolist() == [2, 1]
    assert volume["impairment_count"].tolist() == [1, 1]
    assert volume["impairment_rate"].tolist() == [0.5, 1.0]


def test_static_eda_plots_return_figures():
    balance = class_balance_frame([0, 0, 1])
    volume = temporal_volume_frame(
        pd.to_datetime(["2020-01-01", "2020-02-01", "2020-02-02"]),
        [0, 0, 1],
    )

    class_figure = plot_class_balance(balance)
    temporal_figure = plot_temporal_volume(volume)

    assert class_figure.axes[0].get_title() == "Impairment class balance"
    assert temporal_figure.axes[0].get_title() == (
        "Monthly instruments and impairment rate"
    )
    plt.close(class_figure)
    plt.close(temporal_figure)


def test_eda_helpers_reject_misaligned_or_invalid_inputs():
    with pytest.raises(ValueError, match="binary"):
        class_balance_frame([0, 2])
    with pytest.raises(ValueError, match="aligned"):
        temporal_volume_frame(pd.to_datetime(["2020-01-01"]), [0, 1])

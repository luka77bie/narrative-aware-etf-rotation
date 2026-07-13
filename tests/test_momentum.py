import numpy as np
import pandas as pd
import pytest

from src.signals.momentum import (
    calculate_momentum,
    calculate_momentum_scores,
    cross_sectional_zscore,
    latest_momentum_ranking,
)


def make_price_data() -> pd.DataFrame:
    dates = pd.date_range(
        "2024-01-01",
        periods=80,
        freq="D",
    )

    rows = []

    for index, ticker in enumerate(
        ["ETF_A", "ETF_B", "ETF_C"]
    ):
        growth_rate = 0.001 * (index + 1)

        for day, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adjusted_close": (
                        100 * (1 + growth_rate) ** day
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_calculate_momentum() -> None:
    result = calculate_momentum(
        make_price_data()
    )

    assert "mom_20" in result.columns
    assert "mom_60" in result.columns
    assert not result.dropna(
        subset=["mom_60"]
    ).empty


def test_faster_asset_has_higher_momentum() -> None:
    result = calculate_momentum(
        make_price_data()
    )

    latest = (
        result.groupby("ticker")
        .tail(1)
        .set_index("ticker")
    )

    assert (
        latest.loc["ETF_C", "mom_60"]
        > latest.loc["ETF_A", "mom_60"]
    )


def test_cross_sectional_zscore_mean_is_zero() -> None:
    values = pd.Series([1.0, 2.0, 3.0])

    result = cross_sectional_zscore(values)

    assert np.isclose(result.mean(), 0.0)


def test_zero_variance_zscore_returns_zero() -> None:
    values = pd.Series([2.0, 2.0, 2.0])

    result = cross_sectional_zscore(values)

    assert result.eq(0.0).all()


def test_momentum_scores_rank_strongest_first() -> None:
    momentum = calculate_momentum(
        make_price_data()
    )

    scored = calculate_momentum_scores(
        momentum
    )

    latest = latest_momentum_ranking(
        scored
    )

    assert latest.iloc[0]["ticker"] == "ETF_C"
    assert latest.iloc[0]["momentum_rank"] == 1


def test_invalid_weights_raise_error() -> None:
    momentum = calculate_momentum(
        make_price_data()
    )

    with pytest.raises(
        ValueError,
        match="sum to 1",
    ):
        calculate_momentum_scores(
            momentum,
            weights=(0.7, 0.7),
        )

import numpy as np
import pandas as pd
import pytest

from src.narrative.proxy import (
    calculate_narrative_proxy_scores,
    engineer_market_attention_features,
)


def make_price_data() -> pd.DataFrame:
    dates = pd.bdate_range(
        "2024-01-01",
        periods=100,
    )

    rows = []

    for index, ticker in enumerate(
        ["ETF_A", "ETF_B", "ETF_C"]
    ):
        for day, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adjusted_close": (
                        100
                        * (1.001 + index * 0.0002) ** day
                    ),
                    "volume": (
                        1_000_000
                        + index * 200_000
                        + day * 1_000
                    ),
                    "turnover": (
                        10_000_000
                        + index * 3_000_000
                        + day * 20_000
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_attention_features_created() -> None:
    result = engineer_market_attention_features(
        make_price_data(),
        short_window=20,
        long_window=60,
    )

    assert "turnover_growth" in result.columns
    assert "volume_growth" in result.columns
    assert "attention_momentum" in result.columns
    assert "volatility_expansion" in result.columns


def test_features_have_complete_late_observations() -> None:
    result = engineer_market_attention_features(
        make_price_data(),
        short_window=20,
        long_window=60,
    )

    latest = (
        result.groupby("ticker")
        .tail(1)
    )

    assert latest[
        [
            "turnover_growth",
            "volume_growth",
            "attention_momentum",
            "volatility_expansion",
        ]
    ].notna().all().all()


def test_proxy_scores_created() -> None:
    features = engineer_market_attention_features(
        make_price_data(),
        short_window=20,
        long_window=60,
    )

    result = calculate_narrative_proxy_scores(
        features
    )

    assert "narrative_proxy_score" in result.columns
    assert "narrative_proxy_rank" in result.columns


def test_invalid_proxy_weights_raise_error() -> None:
    features = engineer_market_attention_features(
        make_price_data(),
        short_window=20,
        long_window=60,
    )

    with pytest.raises(
        ValueError,
        match="sum to 1",
    ):
        calculate_narrative_proxy_scores(
            features,
            weights={
                "turnover_growth": 0.5,
                "volume_growth": 0.5,
                "attention_momentum": 0.5,
                "volatility_expansion": 0.5,
            },
        )


def test_no_duplicate_ticker_dates() -> None:
    data = make_price_data()

    duplicated = pd.concat(
        [
            data,
            data.iloc[[0]],
        ],
        ignore_index=True,
    )

    result = engineer_market_attention_features(
        duplicated,
        short_window=20,
        long_window=60,
    )

    assert not result.duplicated(
        subset=["date", "ticker"]
    ).any()

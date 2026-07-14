import numpy as np
import pandas as pd

from src.risk.metrics import (
    calculate_risk_adjusted_score,
    calculate_rolling_risk_metrics,
)


def make_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=100)

    rows = []

    for ticker, volatility in [
        ("LOW_RISK", 0.001),
        ("HIGH_RISK", 0.01),
    ]:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adjusted_close": (
                        100
                        * (1.001 ** index)
                        * (1 + volatility * np.sin(index))
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_rolling_risk_metrics_created() -> None:
    result = calculate_rolling_risk_metrics(
        make_prices(),
        window=60,
    )

    assert "volatility_60" in result.columns
    assert "downside_volatility_60" in result.columns
    assert "drawdown_60" in result.columns


def test_high_risk_asset_has_higher_volatility() -> None:
    result = calculate_rolling_risk_metrics(
        make_prices(),
        window=60,
    )

    latest = (
        result.groupby("ticker")
        .tail(1)
        .set_index("ticker")
    )

    assert (
        latest.loc["HIGH_RISK", "volatility_60"]
        > latest.loc["LOW_RISK", "volatility_60"]
    )


def test_risk_penalty_prefers_lower_volatility() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-31", "2024-01-31"]
            ),
            "ticker": ["LOW_RISK", "HIGH_RISK"],
            "z_mom_60": [1.0, 1.0],
            "volatility_60": [0.10, 0.30],
            "downside_volatility_60": [0.08, 0.25],
        }
    )

    result = calculate_risk_adjusted_score(data)

    best = result.sort_values(
        "risk_adjusted_rank"
    ).iloc[0]

    assert best["ticker"] == "LOW_RISK"

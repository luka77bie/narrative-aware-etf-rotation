from typing import Tuple

import numpy as np
import pandas as pd


def calculate_rolling_risk_metrics(
    prices: pd.DataFrame,
    window: int = 60,
    price_column: str = "adjusted_close",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Calculate rolling volatility, downside volatility and drawdown."""
    required = {"date", "ticker", price_column}
    missing = required - set(prices.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    if window <= 1:
        raise ValueError("window must be greater than 1.")

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"])

    frame["daily_return"] = (
        frame.groupby("ticker")[price_column]
        .pct_change(fill_method=None)
    )

    def add_group_metrics(group: pd.DataFrame) -> pd.DataFrame:
        returns = group["daily_return"]

        group[f"volatility_{window}"] = (
            returns.rolling(window)
            .std(ddof=0)
            * np.sqrt(periods_per_year)
        )

        downside = returns.where(returns < 0, 0.0)

        group[f"downside_volatility_{window}"] = (
            downside.rolling(window)
            .std(ddof=0)
            * np.sqrt(periods_per_year)
        )

        rolling_peak = (
            group[price_column]
            .rolling(window, min_periods=1)
            .max()
        )

        group[f"drawdown_{window}"] = (
            group[price_column] / rolling_peak - 1.0
        )

        return group

    return (
        frame.groupby("ticker", group_keys=False)
        .apply(add_group_metrics)
        .reset_index(drop=True)
    )


def cross_sectional_zscore(
    values: pd.Series,
) -> pd.Series:
    standard_deviation = values.std(ddof=0)

    if (
        pd.isna(standard_deviation)
        or np.isclose(standard_deviation, 0.0)
    ):
        return pd.Series(0.0, index=values.index)

    return (
        values - values.mean()
    ) / standard_deviation


def calculate_risk_adjusted_score(
    data: pd.DataFrame,
    momentum_column: str = "z_mom_60",
    volatility_column: str = "volatility_60",
    downside_column: str = "downside_volatility_60",
    weights: Tuple[float, float, float] = (
        1.0,
        0.25,
        0.15,
    ),
) -> pd.DataFrame:
    """Combine MOM60 with volatility penalties."""
    required = {
        "date",
        "ticker",
        momentum_column,
        volatility_column,
        downside_column,
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            "Missing score columns: "
            + ", ".join(sorted(missing))
        )

    frame = data.copy()

    frame["z_volatility"] = (
        frame.groupby("date")[volatility_column]
        .transform(cross_sectional_zscore)
    )

    frame["z_downside_volatility"] = (
        frame.groupby("date")[downside_column]
        .transform(cross_sectional_zscore)
    )

    momentum_weight, volatility_penalty, downside_penalty = weights

    frame["risk_adjusted_score"] = (
        momentum_weight * frame[momentum_column]
        - volatility_penalty * frame["z_volatility"]
        - downside_penalty * frame["z_downside_volatility"]
    )

    frame["risk_adjusted_rank"] = (
        frame.groupby("date")["risk_adjusted_score"]
        .rank(ascending=False, method="first")
    )

    return frame

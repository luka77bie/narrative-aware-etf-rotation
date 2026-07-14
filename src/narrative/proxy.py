from typing import Dict, Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "ticker",
    "adjusted_close",
    "volume",
    "turnover",
}


def cross_sectional_zscore(
    values: pd.Series,
) -> pd.Series:
    """Calculate cross-sectional z-score for one date."""
    standard_deviation = values.std(ddof=0)

    if (
        pd.isna(standard_deviation)
        or np.isclose(standard_deviation, 0.0)
    ):
        return pd.Series(
            0.0,
            index=values.index,
        )

    return (
        values - values.mean()
    ) / standard_deviation


def engineer_market_attention_features(
    price_data: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 60,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Engineer market-observable Narrative Proxy features.

    All features are computed independently by ticker and use
    trailing historical information only.
    """
    missing = REQUIRED_COLUMNS - set(
        price_data.columns
    )

    if missing:
        raise ValueError(
            "Price data is missing columns: "
            + ", ".join(sorted(missing))
        )

    if short_window <= 1:
        raise ValueError(
            "short_window must be greater than 1."
        )

    if long_window <= short_window:
        raise ValueError(
            "long_window must be greater than short_window."
        )

    frame = price_data.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    numeric_columns = [
        "adjusted_close",
        "volume",
        "turnover",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if frame["date"].isna().any():
        raise ValueError(
            "Price data contains invalid dates."
        )

    if frame[numeric_columns].isna().any().any():
        raise ValueError(
            "Price data contains invalid numeric values."
        )

    if (frame["adjusted_close"] <= 0).any():
        raise ValueError(
            "Adjusted close must be positive."
        )

    if (frame[["volume", "turnover"]] < 0).any().any():
        raise ValueError(
            "Volume and turnover cannot be negative."
        )

    frame = (
        frame.sort_values(
            ["ticker", "date"]
        )
        .drop_duplicates(
            subset=["ticker", "date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    groups = []

    for _, group in frame.groupby(
        "ticker",
        sort=False,
    ):
        group = group.copy()

        group["daily_return"] = (
            group["adjusted_close"]
            .pct_change(fill_method=None)
        )

        turnover_short = (
            group["turnover"]
            .rolling(
                window=short_window,
                min_periods=short_window,
            )
            .mean()
        )

        turnover_long = (
            group["turnover"]
            .rolling(
                window=long_window,
                min_periods=long_window,
            )
            .mean()
        )

        volume_short = (
            group["volume"]
            .rolling(
                window=short_window,
                min_periods=short_window,
            )
            .mean()
        )

        volume_long = (
            group["volume"]
            .rolling(
                window=long_window,
                min_periods=long_window,
            )
            .mean()
        )

        volatility_short = (
            group["daily_return"]
            .rolling(
                window=short_window,
                min_periods=short_window,
            )
            .std(ddof=0)
            * np.sqrt(periods_per_year)
        )

        volatility_long = (
            group["daily_return"]
            .rolling(
                window=long_window,
                min_periods=long_window,
            )
            .std(ddof=0)
            * np.sqrt(periods_per_year)
        )

        turnover_long_std = (
            group["turnover"]
            .rolling(
                window=long_window,
                min_periods=long_window,
            )
            .std(ddof=0)
        )

        group["turnover_growth"] = (
            np.log1p(turnover_short)
            - np.log1p(turnover_long)
        )

        group["volume_growth"] = (
            np.log1p(volume_short)
            - np.log1p(volume_long)
        )

        group["attention_momentum"] = (
            group["turnover"] - turnover_long
        ) / turnover_long_std.replace(0, np.nan)

        group["volatility_expansion"] = (
            volatility_short
            / volatility_long.replace(0, np.nan)
            - 1.0
        )

        groups.append(group)

    return pd.concat(
        groups,
        ignore_index=True,
    )


def calculate_narrative_proxy_scores(
    feature_data: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Calculate ETF-level Narrative Proxy Score.
    """
    if weights is None:
        weights = {
            "turnover_growth": 0.40,
            "volume_growth": 0.25,
            "attention_momentum": 0.25,
            "volatility_expansion": 0.10,
        }

    if not np.isclose(
        sum(weights.values()),
        1.0,
    ):
        raise ValueError(
            "Narrative Proxy weights must sum to 1."
        )

    missing = set(weights) - set(
        feature_data.columns
    )

    if missing:
        raise ValueError(
            "Narrative Proxy data is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = feature_data.copy()

    for feature in weights:
        zscore_column = f"z_{feature}"

        frame[zscore_column] = (
            frame.groupby("date")[feature]
            .transform(cross_sectional_zscore)
        )

    frame["narrative_proxy_score"] = 0.0

    for feature, weight in weights.items():
        frame["narrative_proxy_score"] += (
            weight
            * frame[f"z_{feature}"]
        )

    frame["narrative_proxy_rank"] = (
        frame.groupby("date")[
            "narrative_proxy_score"
        ]
        .rank(
            ascending=False,
            method="first",
        )
    )

    return frame

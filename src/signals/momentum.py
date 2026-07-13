from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def calculate_momentum(
    prices: pd.DataFrame,
    lookbacks: Iterable[int] = (20, 60),
    price_column: str = "adjusted_close",
) -> pd.DataFrame:
    """Calculate trailing momentum for each ETF."""
    required_columns = {"date", "ticker", price_column}
    missing_columns = required_columns - set(prices.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    frame = prices.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame[price_column] = pd.to_numeric(
        frame[price_column],
        errors="coerce",
    )

    if frame["date"].isna().any():
        raise ValueError("Price data contains invalid dates.")

    if frame[price_column].isna().any():
        raise ValueError("Price data contains missing prices.")

    if (frame[price_column] <= 0).any():
        raise ValueError("Price data contains non-positive prices.")

    frame = (
        frame.sort_values(["ticker", "date"])
        .drop_duplicates(
            subset=["ticker", "date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    grouped_prices = frame.groupby(
        "ticker",
        sort=False,
    )[price_column]

    for lookback in lookbacks:
        if lookback <= 0:
            raise ValueError(
                "Momentum lookbacks must be positive."
            )

        frame[f"mom_{lookback}"] = grouped_prices.pct_change(
            periods=lookback,
            fill_method=None,
        )

    return frame


def cross_sectional_zscore(
    values: pd.Series,
) -> pd.Series:
    """Calculate z-score across ETFs for one date."""
    mean = values.mean()
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
        values - mean
    ) / standard_deviation


def calculate_momentum_scores(
    momentum_data: pd.DataFrame,
    lookbacks: Tuple[int, int] = (20, 60),
    weights: Tuple[float, float] = (0.5, 0.5),
) -> pd.DataFrame:
    """Calculate composite momentum score and rank."""
    if len(lookbacks) != len(weights):
        raise ValueError(
            "lookbacks and weights must have equal length."
        )

    if not np.isclose(sum(weights), 1.0):
        raise ValueError(
            "Momentum weights must sum to 1."
        )

    frame = momentum_data.copy()

    momentum_columns = [
        f"mom_{lookback}"
        for lookback in lookbacks
    ]

    missing_columns = (
        set(momentum_columns)
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing momentum columns: "
            + ", ".join(sorted(missing_columns))
        )

    for lookback in lookbacks:
        momentum_column = f"mom_{lookback}"
        zscore_column = f"z_mom_{lookback}"

        frame[zscore_column] = (
            frame.groupby("date")[momentum_column]
            .transform(cross_sectional_zscore)
        )

    frame["momentum_score"] = 0.0

    for lookback, weight in zip(
        lookbacks,
        weights,
    ):
        frame["momentum_score"] += (
            weight
            * frame[f"z_mom_{lookback}"]
        )

    frame["momentum_rank"] = (
        frame.groupby("date")["momentum_score"]
        .rank(
            ascending=False,
            method="first",
        )
    )

    return frame


def latest_momentum_ranking(
    scored_data: pd.DataFrame,
) -> pd.DataFrame:
    """Return latest complete cross-sectional ranking."""
    complete = scored_data.dropna(
        subset=[
            "mom_20",
            "mom_60",
            "momentum_score",
        ]
    )

    if complete.empty:
        raise ValueError(
            "No complete momentum observations available."
        )

    latest_date = complete["date"].max()

    return (
        complete.loc[
            complete["date"] == latest_date
        ]
        .sort_values("momentum_rank")
        .reset_index(drop=True)
    )

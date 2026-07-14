import numpy as np
import pandas as pd


def apply_volatility_target_overlay(
    strategy_returns: pd.DataFrame,
    cash_returns: pd.DataFrame,
    target_volatility: float = 0.15,
    lookback: int = 60,
    periods_per_year: int = 252,
    minimum_exposure: float = 0.0,
    maximum_exposure: float = 1.0,
) -> pd.DataFrame:
    """
    Apply a lagged volatility-targeting overlay to daily strategy returns.

    Exposure for date t uses realised volatility calculated through t-1.
    Residual portfolio weight is allocated to the cash asset.
    """
    required_strategy = {"date", "net_return"}
    required_cash = {"date", "cash_return"}

    missing_strategy = required_strategy - set(
        strategy_returns.columns
    )
    missing_cash = required_cash - set(
        cash_returns.columns
    )

    if missing_strategy:
        raise ValueError(
            "Strategy returns missing columns: "
            + ", ".join(sorted(missing_strategy))
        )

    if missing_cash:
        raise ValueError(
            "Cash returns missing columns: "
            + ", ".join(sorted(missing_cash))
        )

    if target_volatility <= 0:
        raise ValueError(
            "target_volatility must be positive."
        )

    if lookback <= 1:
        raise ValueError(
            "lookback must be greater than 1."
        )

    frame = strategy_returns[
        ["date", "net_return"]
    ].copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    cash = cash_returns[
        ["date", "cash_return"]
    ].copy()

    cash["date"] = pd.to_datetime(
        cash["date"],
        errors="coerce",
    )

    frame = frame.merge(
        cash,
        on="date",
        how="left",
        validate="one_to_one",
    )

    frame["cash_return"] = (
        frame["cash_return"]
        .fillna(0.0)
    )

    frame["realised_volatility"] = (
        frame["net_return"]
        .rolling(
            window=lookback,
            min_periods=lookback,
        )
        .std(ddof=0)
        * np.sqrt(periods_per_year)
    )

    # Shift ensures today's exposure only uses information
    # available through the previous trading day.
    frame["lagged_realised_volatility"] = (
        frame["realised_volatility"]
        .shift(1)
    )

    raw_exposure = (
        target_volatility
        / frame["lagged_realised_volatility"]
    )

    frame["risky_exposure"] = raw_exposure.clip(
        lower=minimum_exposure,
        upper=maximum_exposure,
    )

    # Remain fully invested until sufficient history exists.
    frame["risky_exposure"] = (
        frame["risky_exposure"]
        .fillna(maximum_exposure)
    )

    frame["cash_exposure"] = (
        1.0 - frame["risky_exposure"]
    )

    frame["overlay_return"] = (
        frame["risky_exposure"]
        * frame["net_return"]
        + frame["cash_exposure"]
        * frame["cash_return"]
    )

    frame["overlay_nav"] = (
        1.0 + frame["overlay_return"]
    ).cumprod()

    return frame

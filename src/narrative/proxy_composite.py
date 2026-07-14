import numpy as np
import pandas as pd


def cross_sectional_zscore(
    values: pd.Series,
) -> pd.Series:
    """Calculate a cross-sectional z-score for one date."""
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


def combine_momentum_and_proxy(
    momentum: pd.DataFrame,
    proxy: pd.DataFrame,
    proxy_weight: float,
) -> pd.DataFrame:
    """
    Combine MOM60 and market-attention proxy.

    Policy data is deliberately excluded because the current
    policy archive remains validation-only.
    """
    if not 0.0 <= proxy_weight <= 1.0:
        raise ValueError(
            "proxy_weight must be between 0 and 1."
        )

    required_momentum = {
        "date",
        "ticker",
        "mom_60",
        "z_mom_60",
    }

    required_proxy = {
        "date",
        "ticker",
        "narrative_proxy_score",
    }

    missing_momentum = (
        required_momentum
        - set(momentum.columns)
    )

    missing_proxy = (
        required_proxy
        - set(proxy.columns)
    )

    if missing_momentum:
        raise ValueError(
            "Momentum data is missing columns: "
            + ", ".join(sorted(missing_momentum))
        )

    if missing_proxy:
        raise ValueError(
            "Proxy data is missing columns: "
            + ", ".join(sorted(missing_proxy))
        )

    momentum_frame = momentum.copy()
    proxy_frame = proxy.copy()

    momentum_frame["date"] = pd.to_datetime(
        momentum_frame["date"],
        errors="coerce",
    )

    proxy_frame["date"] = pd.to_datetime(
        proxy_frame["date"],
        errors="coerce",
    )

    if (
        momentum_frame["date"].isna().any()
        or proxy_frame["date"].isna().any()
    ):
        raise ValueError(
            "Momentum or proxy data contains invalid dates."
        )

    combined = momentum_frame.merge(
        proxy_frame[
            [
                "date",
                "ticker",
                "narrative_proxy_score",
            ]
        ],
        on=[
            "date",
            "ticker",
        ],
        how="inner",
        validate="one_to_one",
    )

    if combined.empty:
        raise ValueError(
            "Momentum and proxy data have no overlapping rows."
        )

    combined["z_narrative_proxy"] = (
        combined.groupby("date")[
            "narrative_proxy_score"
        ]
        .transform(cross_sectional_zscore)
    )

    momentum_weight = 1.0 - proxy_weight

    combined["momentum_score"] = (
        momentum_weight
        * combined["z_mom_60"]
        + proxy_weight
        * combined["z_narrative_proxy"]
    )

    combined["momentum_rank"] = (
        combined.groupby("date")[
            "momentum_score"
        ]
        .rank(
            ascending=False,
            method="first",
        )
    )

    combined["proxy_weight"] = (
        proxy_weight
    )

    combined["signal_type"] = (
        "historical_market_data_proxy"
    )

    return combined

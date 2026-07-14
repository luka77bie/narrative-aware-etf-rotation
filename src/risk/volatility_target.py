import numpy as np
import pandas as pd


def calculate_target_exposure(
    realised_volatility: float,
    target_volatility: float = 0.15,
    maximum_exposure: float = 1.0,
    minimum_exposure: float = 0.0,
) -> float:
    """
    Calculate portfolio risky-asset exposure.

    exposure = target volatility / realised volatility
    """
    if target_volatility <= 0:
        raise ValueError(
            "target_volatility must be positive."
        )

    if maximum_exposure <= 0:
        raise ValueError(
            "maximum_exposure must be positive."
        )

    if realised_volatility is None:
        return maximum_exposure

    if pd.isna(realised_volatility):
        return maximum_exposure

    if realised_volatility <= 0:
        return maximum_exposure

    exposure = (
        target_volatility
        / realised_volatility
    )

    return float(
        np.clip(
            exposure,
            minimum_exposure,
            maximum_exposure,
        )
    )


def apply_volatility_target_to_weights(
    risky_weights: dict,
    realised_volatility: float,
    target_volatility: float = 0.15,
    cash_ticker: str = "159001",
) -> dict:
    """
    Scale risky weights and allocate residual weight to cash.
    """
    exposure = calculate_target_exposure(
        realised_volatility=realised_volatility,
        target_volatility=target_volatility,
    )

    scaled_weights = {
        ticker: weight * exposure
        for ticker, weight in risky_weights.items()
    }

    cash_weight = 1.0 - sum(
        scaled_weights.values()
    )

    if cash_weight > 0:
        scaled_weights[cash_ticker] = cash_weight

    return scaled_weights

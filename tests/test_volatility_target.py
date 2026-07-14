import numpy as np

from src.risk.volatility_target import (
    apply_volatility_target_to_weights,
    calculate_target_exposure,
)


def test_exposure_reduced_when_volatility_high() -> None:
    exposure = calculate_target_exposure(
        realised_volatility=0.30,
        target_volatility=0.15,
    )

    assert np.isclose(exposure, 0.5)


def test_exposure_capped_at_one() -> None:
    exposure = calculate_target_exposure(
        realised_volatility=0.10,
        target_volatility=0.15,
    )

    assert np.isclose(exposure, 1.0)


def test_residual_weight_allocated_to_cash() -> None:
    result = apply_volatility_target_to_weights(
        risky_weights={
            "A": 0.5,
            "B": 0.5,
        },
        realised_volatility=0.30,
        target_volatility=0.15,
        cash_ticker="CASH",
    )

    assert np.isclose(result["A"], 0.25)
    assert np.isclose(result["B"], 0.25)
    assert np.isclose(result["CASH"], 0.50)
    assert np.isclose(sum(result.values()), 1.0)

import numpy as np
import pandas as pd

from src.risk.volatility_overlay import (
    apply_volatility_target_overlay,
)


def make_returns() -> tuple:
    dates = pd.bdate_range(
        "2024-01-01",
        periods=100,
    )

    strategy = pd.DataFrame(
        {
            "date": dates,
            "net_return": [
                0.02 if index % 2 == 0 else -0.02
                for index in range(100)
            ],
        }
    )

    cash = pd.DataFrame(
        {
            "date": dates,
            "cash_return": [0.0001] * 100,
        }
    )

    return strategy, cash


def test_overlay_creates_required_columns() -> None:
    strategy, cash = make_returns()

    result = apply_volatility_target_overlay(
        strategy_returns=strategy,
        cash_returns=cash,
        target_volatility=0.15,
        lookback=20,
    )

    assert "risky_exposure" in result.columns
    assert "cash_exposure" in result.columns
    assert "overlay_return" in result.columns
    assert "overlay_nav" in result.columns


def test_high_volatility_reduces_exposure() -> None:
    strategy, cash = make_returns()

    result = apply_volatility_target_overlay(
        strategy_returns=strategy,
        cash_returns=cash,
        target_volatility=0.10,
        lookback=20,
    )

    assert (
        result["risky_exposure"].iloc[-1]
        < 1.0
    )


def test_exposures_sum_to_one() -> None:
    strategy, cash = make_returns()

    result = apply_volatility_target_overlay(
        strategy_returns=strategy,
        cash_returns=cash,
        target_volatility=0.15,
        lookback=20,
    )

    total = (
        result["risky_exposure"]
        + result["cash_exposure"]
    )

    assert np.allclose(total, 1.0)


def test_overlay_uses_lagged_volatility() -> None:
    strategy, cash = make_returns()

    result = apply_volatility_target_overlay(
        strategy_returns=strategy,
        cash_returns=cash,
        target_volatility=0.15,
        lookback=20,
    )

    # First 20-day estimate is not used until the next day.
    assert result.loc[19, "risky_exposure"] == 1.0

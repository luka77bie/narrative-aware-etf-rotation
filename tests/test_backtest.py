import numpy as np
import pandas as pd

from src.backtest.engine import (
    build_price_matrix,
    calculate_performance_metrics,
    get_month_end_signal_dates,
)
from src.portfolio.construction import (
    calculate_turnover,
    select_top_n_equal_weight,
)


def test_select_top_n_equal_weight() -> None:
    ranking = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "momentum_score": [3.0, 2.0, 1.0],
        }
    )

    result = select_top_n_equal_weight(
        ranking,
        top_n=2,
    )

    assert result["ticker"].tolist() == [
        "A",
        "B",
    ]

    assert np.isclose(
        result["weight"].sum(),
        1.0,
    )


def test_calculate_turnover() -> None:
    old_weights = {
        "A": 0.5,
        "B": 0.5,
    }

    new_weights = {
        "B": 0.5,
        "C": 0.5,
    }

    turnover = calculate_turnover(
        old_weights,
        new_weights,
    )

    assert np.isclose(
        turnover,
        0.5,
    )


def test_build_price_matrix() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "ticker": [
                "A",
                "B",
                "A",
                "B",
            ],
            "adjusted_close": [
                100,
                200,
                101,
                202,
            ],
        }
    )

    matrix = build_price_matrix(prices)

    assert matrix.shape == (2, 2)


def test_month_end_signal_dates() -> None:
    dates = pd.bdate_range(
        "2024-01-01",
        "2024-03-15",
    )

    result = get_month_end_signal_dates(
        dates
    )

    assert result[0] == pd.Timestamp(
        "2024-01-31"
    )

    assert result[1] == pd.Timestamp(
        "2024-02-29"
    )


def test_performance_metrics() -> None:
    returns = pd.Series(
        [0.01, -0.005, 0.002, 0.003]
    )

    metrics = calculate_performance_metrics(
        returns
    )

    assert "sharpe" in metrics
    assert "maximum_drawdown" in metrics
    assert metrics["total_return"] > 0

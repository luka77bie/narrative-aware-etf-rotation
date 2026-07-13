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


def test_new_weights_do_not_capture_execution_day_return() -> None:
    from src.backtest.engine import (
        run_monthly_top_n_backtest,
    )

    dates = pd.bdate_range(
        "2024-01-01",
        "2024-03-15",
    )

    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A"] * len(dates),
            "adjusted_close": [
                100.0 + index
                for index in range(len(dates))
            ],
        }
    )

    signals = prices.copy()
    signals["mom_20"] = 0.10
    signals["mom_60"] = 0.20
    signals["momentum_score"] = 1.0

    result = run_monthly_top_n_backtest(
        prices=prices,
        scored_signals=signals,
        top_n=1,
        transaction_cost_rate=0.0,
        minimum_signal_assets=1,
    )

    rebalances = result["rebalances"]
    returns = result["returns"]

    first_execution_date = pd.Timestamp(
        rebalances.iloc[0]["execution_date"]
    )

    assert (
        returns.loc[
            first_execution_date,
            "gross_return",
        ]
        == 0.0
    )


def test_backtest_skips_dates_with_insufficient_assets() -> None:
    from src.backtest.engine import (
        run_monthly_top_n_backtest,
    )

    dates = pd.bdate_range(
        "2024-01-01",
        "2024-03-15",
    )

    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A"] * len(dates),
            "adjusted_close": [
                100.0 + index
                for index in range(len(dates))
            ],
        }
    )

    signals = prices.copy()
    signals["mom_20"] = 0.10
    signals["mom_60"] = 0.20
    signals["momentum_score"] = 1.0

    result = run_monthly_top_n_backtest(
        prices=prices,
        scored_signals=signals,
        top_n=1,
        transaction_cost_rate=0.0,
        minimum_signal_assets=2,
    )

    assert result["rebalances"].empty

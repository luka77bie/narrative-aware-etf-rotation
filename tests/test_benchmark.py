import pandas as pd

from src.evaluation.benchmark import (
    build_buy_and_hold_benchmark,
    build_comparison_table,
)


def make_benchmark_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2024-01-01",
                periods=10,
                freq="D",
            ),
            "ticker": ["510300"] * 10,
            "adjusted_close": [
                100,
                101,
                102,
                101,
                103,
                104,
                105,
                106,
                105,
                107,
            ],
        }
    )


def test_build_buy_and_hold_benchmark() -> None:
    result = build_buy_and_hold_benchmark(
        price_data=make_benchmark_prices(),
        ticker="510300",
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-01-10"),
    )

    assert not result["returns"].empty
    assert not result["metrics"].empty
    assert (
        result["returns"]["benchmark_nav"].iloc[-1]
        > 1.0
    )


def test_build_comparison_table() -> None:
    strategy = pd.DataFrame(
        [
            {
                "total_return": 0.20,
                "cagr": 0.10,
                "annual_volatility": 0.15,
                "sharpe": 0.60,
                "sortino": 0.80,
                "maximum_drawdown": -0.12,
                "calmar": 0.83,
            }
        ]
    )

    benchmark = pd.DataFrame(
        [
            {
                "model": "Buy and Hold 510300",
                "total_return": 0.10,
                "cagr": 0.05,
                "annual_volatility": 0.18,
                "sharpe": 0.30,
                "sortino": 0.40,
                "maximum_drawdown": -0.20,
                "calmar": 0.25,
            }
        ]
    )

    comparison = build_comparison_table(
        strategy,
        benchmark,
    )

    assert len(comparison) == 2
    assert comparison.iloc[0]["model"] == "Momentum Top-3"

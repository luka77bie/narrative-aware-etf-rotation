from typing import Dict

import pandas as pd

from src.backtest.engine import calculate_performance_metrics


def build_buy_and_hold_benchmark(
    price_data: pd.DataFrame,
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """
    Build a buy-and-hold benchmark from adjusted close prices.
    """
    required_columns = {
        "date",
        "ticker",
        "adjusted_close",
    }

    missing = required_columns - set(price_data.columns)

    if missing:
        raise ValueError(
            "Benchmark price data is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = price_data.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame = frame.loc[
        (frame["ticker"].astype(str) == str(ticker))
        & (frame["date"] >= pd.Timestamp(start_date))
        & (frame["date"] <= pd.Timestamp(end_date))
    ].copy()

    frame = (
        frame.sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if frame.empty:
        raise ValueError(
            f"No benchmark observations found for {ticker}."
        )

    frame["benchmark_return"] = (
        frame["adjusted_close"]
        .pct_change(fill_method=None)
        .fillna(0.0)
    )

    frame["benchmark_nav"] = (
        1.0 + frame["benchmark_return"]
    ).cumprod()

    metrics = calculate_performance_metrics(
        frame["benchmark_return"]
    )

    metrics_frame = pd.DataFrame(
        [
            {
                "model": f"Buy and Hold {ticker}",
                **metrics,
            }
        ]
    )

    return {
        "returns": frame[
            [
                "date",
                "benchmark_return",
                "benchmark_nav",
            ]
        ],
        "metrics": metrics_frame,
    }


def build_comparison_table(
    strategy_metrics: pd.DataFrame,
    benchmark_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Combine strategy and benchmark metrics."""
    strategy = strategy_metrics.copy()
    benchmark = benchmark_metrics.copy()

    strategy.insert(
        0,
        "model",
        "Momentum Top-3",
    )

    common_columns = [
        "model",
        "total_return",
        "cagr",
        "annual_volatility",
        "sharpe",
        "sortino",
        "maximum_drawdown",
        "calmar",
    ]

    return pd.concat(
        [
            strategy[common_columns],
            benchmark[common_columns],
        ],
        ignore_index=True,
    )

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.market_data import load_cached_price_data
from src.evaluation.benchmark import (
    build_buy_and_hold_benchmark,
    build_comparison_table,
)


STRATEGY_RETURNS_PATH = Path(
    "outputs/momentum_backtest_returns.csv"
)

STRATEGY_METRICS_PATH = Path(
    "outputs/momentum_backtest_metrics.csv"
)

OUTPUT_DIRECTORY = Path("outputs")


def main() -> int:
    if not STRATEGY_RETURNS_PATH.exists():
        raise FileNotFoundError(
            "Momentum backtest results not found. Run:\n"
            "python3 scripts/run_momentum_backtest.py"
        )

    strategy_returns = pd.read_csv(
        STRATEGY_RETURNS_PATH,
        parse_dates=["date"],
    )

    strategy_metrics = pd.read_csv(
        STRATEGY_METRICS_PATH,
    )

    start_date = strategy_returns["date"].min()
    end_date = strategy_returns["date"].max()

    benchmark_prices = load_cached_price_data(
        ticker="510300",
        cache_directory="data/raw",
    )

    benchmark = build_buy_and_hold_benchmark(
        price_data=benchmark_prices,
        ticker="510300",
        start_date=start_date,
        end_date=end_date,
    )

    comparison = build_comparison_table(
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark["metrics"],
    )

    benchmark_returns = benchmark["returns"]

    merged_nav = strategy_returns[
        [
            "date",
            "net_nav",
        ]
    ].merge(
        benchmark_returns,
        on="date",
        how="inner",
    )

    comparison.to_csv(
        OUTPUT_DIRECTORY
        / "momentum_vs_benchmark_metrics.csv",
        index=False,
    )

    merged_nav.to_csv(
        OUTPUT_DIRECTORY
        / "momentum_vs_benchmark_nav.csv",
        index=False,
    )

    display = comparison.copy()

    percentage_columns = [
        "total_return",
        "cagr",
        "annual_volatility",
        "maximum_drawdown",
    ]

    for column in percentage_columns:
        display[column] *= 100

    print("")
    print("Momentum vs CSI 300 Benchmark")
    print("=" * 100)
    print(display.round(3).to_string(index=False))

    print("")
    print(
        "Metrics: "
        "outputs/momentum_vs_benchmark_metrics.csv"
    )
    print(
        "NAV: "
        "outputs/momentum_vs_benchmark_nav.csv"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

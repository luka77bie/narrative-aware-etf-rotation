import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.backtest.engine import (
    calculate_performance_metrics,
    run_monthly_top_n_backtest,
)
from src.data.market_data import (
    load_cached_price_data,
)


UNIVERSE_PATH = Path(
    "config/etf_universe.csv"
)

SIGNAL_PATH = Path(
    "outputs/momentum_signal_history.csv"
)

OUTPUT_DIRECTORY = Path("outputs")


def load_universe():
    """
    Return the signal universe and the broader price universe.

    signal_universe:
        Included non-cash ETFs used for momentum ranking.

    price_universe:
        Signal ETFs plus the cash reserve ETF 159001.
    """
    universe = pd.read_csv(
        UNIVERSE_PATH,
        dtype={"ticker": "string"},
    )

    included = (
        universe["include"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    universe = universe.loc[included].copy()

    signal_universe = universe.loc[
        universe["style"]
        .astype(str)
        .str.lower()
        .ne("cash")
    ].copy()

    cash_universe = universe.loc[
        universe["ticker"] == "159001"
    ].copy()

    if signal_universe.empty:
        raise ValueError(
            "No non-cash ETFs available."
        )

    if cash_universe.empty:
        raise ValueError(
            "Cash reserve ETF 159001 is missing "
            "from the official universe."
        )

    price_universe = pd.concat(
        [
            signal_universe,
            cash_universe,
        ],
        ignore_index=True,
    ).drop_duplicates(
        subset=["ticker"]
    )

    return signal_universe, price_universe


def load_prices(
    universe: pd.DataFrame,
) -> pd.DataFrame:
    frames = []

    for row in universe.itertuples(index=False):
        ticker = str(row.ticker).zfill(6)

        data = load_cached_price_data(
            ticker=ticker,
            cache_directory="data/raw",
        )

        frames.append(
            data[
                [
                    "date",
                    "ticker",
                    "adjusted_close",
                ]
            ]
        )

        print(
            f"[LOADED] {ticker} {row.name}: "
            f"{len(data)} rows"
        )

    if not frames:
        raise RuntimeError(
            "No cached ETF prices were loaded."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def main() -> int:
    if not SIGNAL_PATH.exists():
        raise FileNotFoundError(
            "Momentum signal history not found. Run:\n"
            "python3 scripts/run_momentum_signal.py"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    signal_universe, price_universe = load_universe()
    prices = load_prices(price_universe)

    signals = pd.read_csv(
        SIGNAL_PATH,
        dtype={"ticker": "string"},
        parse_dates=["date"],
    )

    signals = signals.loc[
        signals["ticker"].isin(
            signal_universe["ticker"]
        )
    ].copy()

    result = run_monthly_top_n_backtest(
        prices=prices,
        scored_signals=signals,
        top_n=3,
        transaction_cost_rate=0.001,
        minimum_signal_assets=10,
        use_cash_filter=True,
        cash_ticker="159001",
        absolute_momentum_column="mom_60",
    )

    returns = result["returns"].copy()
    holdings = result["holdings"].copy()
    rebalances = result["rebalances"].copy()
    metrics = result["metrics"].copy()

    if not rebalances.empty:
        first_execution_date = pd.Timestamp(
            rebalances["execution_date"].min()
        )

        returns = returns.loc[
            returns.index >= first_execution_date
        ].copy()

        returns["gross_nav"] = (
            1.0 + returns["gross_return"]
        ).cumprod()

        returns["net_nav"] = (
            1.0 + returns["net_return"]
        ).cumprod()

        recomputed_metrics = calculate_performance_metrics(
            returns["net_return"]
        )

        metrics = pd.DataFrame(
            [
                {
                    **recomputed_metrics,
                    "top_n": 3,
                    "transaction_cost_rate": 0.001,
                    "rebalance_count": len(rebalances),
                    "average_turnover": (
                        rebalances["turnover"].mean()
                        if not rebalances.empty
                        else 0.0
                    ),
                }
            ]
        )

    returns.to_csv(
        OUTPUT_DIRECTORY
        / "cash_filter_backtest_returns.csv"
    )

    holdings.to_csv(
        OUTPUT_DIRECTORY
        / "cash_filter_backtest_holdings.csv",
        index=False,
    )

    rebalances.to_csv(
        OUTPUT_DIRECTORY
        / "cash_filter_backtest_rebalances.csv",
        index=False,
    )

    metrics.to_csv(
        OUTPUT_DIRECTORY
        / "cash_filter_backtest_metrics.csv",
        index=False,
    )

    print("")
    print("MOM60 + Cash Filter Backtest Configuration")
    print("=" * 60)
    print(f"Signal assets: {len(signal_universe)}")
    print(f"Price assets including cash: {len(price_universe)}")
    print("Selection: Top 3 Equal Weight")
    print("Rebalance: Monthly")
    print("Signal: MOM60 only with absolute momentum cash filter")
    print("Execution: Next trading-day close")
    print("Transaction cost: 10 bps × turnover")

    print("")
    print("Performance Metrics")
    print("=" * 60)

    display_metrics = metrics.copy()

    percentage_columns = [
        "total_return",
        "cagr",
        "annual_volatility",
        "maximum_drawdown",
        "average_turnover",
    ]

    for column in percentage_columns:
        display_metrics[column] = (
            display_metrics[column] * 100
        )

    print(
        display_metrics.round(3)
        .to_string(index=False)
    )

    print("")
    print("Latest 10 Rebalances")
    print("=" * 100)

    if not rebalances.empty:
        print(
            rebalances.tail(10)
            .to_string(index=False)
        )

    print("")
    print(
        "Returns: "
        "outputs/cash_filter_backtest_returns.csv"
    )
    print(
        "Holdings: "
        "outputs/cash_filter_backtest_holdings.csv"
    )
    print(
        "Rebalances: "
        "outputs/cash_filter_backtest_rebalances.csv"
    )
    print(
        "Metrics: "
        "outputs/cash_filter_backtest_metrics.csv"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

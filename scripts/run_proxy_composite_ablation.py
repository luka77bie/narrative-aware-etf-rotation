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
from src.data.market_data import load_cached_price_data
from src.narrative.proxy_composite import (
    combine_momentum_and_proxy,
)


MOMENTUM_PATH = Path(
    "outputs/momentum_signal_history.csv"
)

PROXY_PATH = Path(
    "outputs/narrative_proxy_signal_history.csv"
)

UNIVERSE_PATH = Path(
    "config/etf_universe.csv"
)

OUTPUT_DIRECTORY = Path("outputs")

PROXY_WEIGHTS = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.50,
]


def load_universe() -> pd.DataFrame:
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

    universe = universe.loc[
        universe["style"]
        .astype(str)
        .str.lower()
        .ne("cash")
    ].copy()

    universe["ticker"] = (
        universe["ticker"]
        .astype(str)
        .str.zfill(6)
    )

    if universe.empty:
        raise ValueError(
            "No included non-cash ETFs were found."
        )

    return universe


def load_prices(
    universe: pd.DataFrame,
) -> pd.DataFrame:
    frames = []

    for row in universe.itertuples(
        index=False
    ):
        ticker = str(row.ticker).zfill(6)

        data = load_cached_price_data(
            ticker=ticker,
            cache_directory="data/raw",
        )

        required_columns = {
            "date",
            "ticker",
            "adjusted_close",
        }

        missing = (
            required_columns
            - set(data.columns)
        )

        if missing:
            raise ValueError(
                f"{ticker} price data is missing: "
                + ", ".join(sorted(missing))
            )

        frames.append(
            data[
                [
                    "date",
                    "ticker",
                    "adjusted_close",
                ]
            ].copy()
        )

        print(
            f"[LOADED] {ticker} {row.name}: "
            f"{len(data)} rows"
        )

    if not frames:
        raise ValueError(
            "No price frames were loaded."
        )

    prices = pd.concat(
        frames,
        ignore_index=True,
    )

    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="coerce",
    )

    prices["ticker"] = (
        prices["ticker"]
        .astype(str)
        .str.zfill(6)
    )

    if prices["date"].isna().any():
        raise ValueError(
            "Price panel contains invalid dates."
        )

    return prices


def load_signals():
    if not MOMENTUM_PATH.exists():
        raise FileNotFoundError(
            "Momentum signal history is missing. Run:\n"
            "python3 scripts/run_momentum_signal.py"
        )

    if not PROXY_PATH.exists():
        raise FileNotFoundError(
            "Narrative Proxy history is missing. Run:\n"
            "python3 scripts/run_narrative_proxy_signal.py"
        )

    momentum = pd.read_csv(
        MOMENTUM_PATH,
        dtype={"ticker": "string"},
        parse_dates=["date"],
    )

    proxy = pd.read_csv(
        PROXY_PATH,
        dtype={"ticker": "string"},
        parse_dates=["date"],
    )

    momentum["ticker"] = (
        momentum["ticker"]
        .astype(str)
        .str.zfill(6)
    )

    proxy["ticker"] = (
        proxy["ticker"]
        .astype(str)
        .str.zfill(6)
    )

    return momentum, proxy


def main() -> int:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    universe = load_universe()
    prices = load_prices(universe)
    momentum, proxy = load_signals()

    valid_tickers = set(
        universe["ticker"].astype(str)
    )

    momentum = momentum.loc[
        momentum["ticker"].isin(
            valid_tickers
        )
    ].copy()

    proxy = proxy.loc[
        proxy["ticker"].isin(
            valid_tickers
        )
    ].copy()

    overlap = momentum.merge(
        proxy[
            [
                "date",
                "ticker",
            ]
        ],
        on=[
            "date",
            "ticker",
        ],
        how="inner",
    )

    if overlap.empty:
        raise ValueError(
            "Momentum and proxy signals have no overlap."
        )

    common_start_date = overlap["date"].min()
    common_end_date = overlap["date"].max()

    print("")
    print("Proxy Composite Ablation Configuration")
    print("=" * 90)
    print(
        f"Common date range: "
        f"{common_start_date.date()} to "
        f"{common_end_date.date()}"
    )
    print(
        f"Common signal rows: {len(overlap)}"
    )
    print(
        f"Universe ETFs: {len(universe)}"
    )
    print("Top N: 3")
    print("Rebalance: monthly")
    print("Transaction cost: 10 bps × turnover")
    print("")

    metric_rows = []

    for proxy_weight in PROXY_WEIGHTS:
        print(
            f"[RUN] Proxy weight "
            f"{proxy_weight:.0%}"
        )

        combined = combine_momentum_and_proxy(
            momentum=momentum,
            proxy=proxy,
            proxy_weight=proxy_weight,
        )

        result = run_monthly_top_n_backtest(
            prices=prices,
            scored_signals=combined,
            top_n=3,
            transaction_cost_rate=0.001,
            minimum_signal_assets=10,
        )

        returns = result["returns"].copy()
        rebalances = result["rebalances"].copy()

        if rebalances.empty:
            raise ValueError(
                f"No rebalances generated for "
                f"proxy weight {proxy_weight:.0%}."
            )

        if "date" not in returns.columns:
            index_name = (
                returns.index.name
                if returns.index.name
                else "date"
            )

            returns = (
                returns.reset_index()
                .rename(
                    columns={
                        index_name: "date",
                        "index": "date",
                    }
                )
            )

        returns["date"] = pd.to_datetime(
            returns["date"],
            errors="coerce",
        )

        if returns["date"].isna().any():
            raise ValueError(
                "Backtest returns contain invalid dates."
            )

        if returns.empty:
            raise ValueError(
                f"No returns generated for "
                f"proxy weight {proxy_weight:.0%}."
            )

        execution_date_column = (
            "execution_date"
            if "execution_date" in rebalances.columns
            else "date"
        )

        first_execution_date = pd.to_datetime(
            rebalances[execution_date_column],
            errors="coerce",
        ).min()

        if pd.isna(first_execution_date):
            raise ValueError(
                "Unable to determine first execution date."
            )

        evaluation_returns = returns.loc[
            returns["date"] >= first_execution_date
        ].copy()

        if evaluation_returns.empty:
            raise ValueError(
                "No returns remain after first execution date."
            )

        metrics = calculate_performance_metrics(
            evaluation_returns["net_return"]
        )

        average_turnover = (
            rebalances["turnover"].mean()
            if not rebalances.empty
            else 0.0
        )

        metric_rows.append(
            {
                "model": (
                    f"MOM60 + "
                    f"{proxy_weight:.0%} Proxy"
                ),
                "proxy_weight": proxy_weight,
                **metrics,
                "average_turnover": (
                    average_turnover
                ),
                "backtest_start_date": (
                    evaluation_returns["date"].min()
                ),
                "backtest_end_date": (
                    evaluation_returns["date"].max()
                ),
                "return_observations": len(
                    evaluation_returns
                ),
                "rebalance_count": len(
                    rebalances
                ),
            }
        )

        prefix = (
            "proxy_composite_"
            f"{int(proxy_weight * 100)}pct"
        )

        combined.to_csv(
            OUTPUT_DIRECTORY
            / f"{prefix}_signals.csv",
            index=False,
        )

        evaluation_returns.to_csv(
            OUTPUT_DIRECTORY
            / f"{prefix}_returns.csv",
            index=False,
        )

        result["holdings"].to_csv(
            OUTPUT_DIRECTORY
            / f"{prefix}_holdings.csv",
            index=False,
        )

        rebalances.to_csv(
            OUTPUT_DIRECTORY
            / f"{prefix}_rebalances.csv",
            index=False,
        )

    metrics_frame = pd.DataFrame(
        metric_rows
    )

    output_path = (
        OUTPUT_DIRECTORY
        / "proxy_composite_ablation_metrics.csv"
    )

    metrics_frame.to_csv(
        output_path,
        index=False,
    )

    display = metrics_frame.copy()

    percentage_columns = [
        "cagr",
        "annual_volatility",
        "maximum_drawdown",
        "average_turnover",
    ]

    for column in percentage_columns:
        display[column] *= 100

    display_columns = [
        "model",
        "cagr",
        "annual_volatility",
        "sharpe",
        "sortino",
        "maximum_drawdown",
        "calmar",
        "average_turnover",
        "rebalance_count",
    ]

    print("")
    print("Proxy Composite Ablation")
    print("=" * 130)

    print(
        display[
            display_columns
        ]
        .round(3)
        .to_string(index=False)
    )

    print("")
    print(f"Output: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

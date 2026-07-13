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
from src.evaluation.ablation import (
    build_ablation_signal,
    summarise_ablation_results,
)


UNIVERSE_PATH = Path(
    "config/etf_universe.csv"
)

SIGNAL_PATH = Path(
    "outputs/momentum_signal_history.csv"
)

OUTPUT_DIRECTORY = Path("outputs")


VARIANTS = {
    "MOM20 only": "mom20_only",
    "MOM60 only": "mom60_only",
    "MOM20 + MOM60": "mom20_mom60",
}


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

    return universe


def load_prices(
    universe: pd.DataFrame,
) -> pd.DataFrame:
    frames = []

    for ticker in universe["ticker"]:
        ticker = str(ticker).zfill(6)

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

    return pd.concat(
        frames,
        ignore_index=True,
    )


def recompute_metrics(
    result: dict,
) -> pd.DataFrame:
    returns = result["returns"].copy()
    rebalances = result["rebalances"].copy()

    if rebalances.empty:
        raise ValueError(
            "Ablation backtest produced no rebalances."
        )

    first_execution_date = pd.Timestamp(
        rebalances["execution_date"].min()
    )

    returns = returns.loc[
        returns.index >= first_execution_date
    ].copy()

    metrics = calculate_performance_metrics(
        returns["net_return"]
    )

    return pd.DataFrame(
        [
            {
                **metrics,
                "top_n": 3,
                "transaction_cost_rate": 0.001,
                "rebalance_count": len(rebalances),
                "average_turnover": (
                    rebalances["turnover"].mean()
                ),
            }
        ]
    )


def main() -> int:
    if not SIGNAL_PATH.exists():
        raise FileNotFoundError(
            "Momentum signal history is missing. Run:\n"
            "python3 scripts/run_momentum_signal.py"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    universe = load_universe()
    prices = load_prices(universe)

    base_signals = pd.read_csv(
        SIGNAL_PATH,
        dtype={"ticker": "string"},
        parse_dates=["date"],
    )

    base_signals = base_signals.loc[
        base_signals["ticker"].isin(
            universe["ticker"]
        )
    ].copy()

    metrics_results = []

    for display_name, variant in VARIANTS.items():
        print(f"[RUN] {display_name}")

        variant_signals = build_ablation_signal(
            scored_signals=base_signals,
            variant=variant,
        )

        result = run_monthly_top_n_backtest(
            prices=prices,
            scored_signals=variant_signals,
            top_n=3,
            transaction_cost_rate=0.001,
        )

        metrics = recompute_metrics(result)
        metrics.insert(0, "model", display_name)

        metrics_results.append(metrics)

        safe_name = (
            variant
            .replace("+", "_")
            .replace(" ", "_")
        )

        result["returns"].to_csv(
            OUTPUT_DIRECTORY
            / f"ablation_{safe_name}_returns.csv"
        )

        result["rebalances"].to_csv(
            OUTPUT_DIRECTORY
            / f"ablation_{safe_name}_rebalances.csv",
            index=False,
        )

    summary = pd.concat(
        metrics_results,
        ignore_index=True,
    )

    summary.to_csv(
        OUTPUT_DIRECTORY
        / "momentum_ablation_metrics.csv",
        index=False,
    )

    display = summary.copy()

    percentage_columns = [
        "total_return",
        "cagr",
        "annual_volatility",
        "maximum_drawdown",
        "average_turnover",
    ]

    for column in percentage_columns:
        display[column] *= 100

    print("")
    print("Momentum Ablation Results")
    print("=" * 120)

    print(
        display[
            [
                "model",
                "total_return",
                "cagr",
                "annual_volatility",
                "sharpe",
                "sortino",
                "maximum_drawdown",
                "calmar",
                "average_turnover",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("")
    print(
        "Output: "
        "outputs/momentum_ablation_metrics.csv"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

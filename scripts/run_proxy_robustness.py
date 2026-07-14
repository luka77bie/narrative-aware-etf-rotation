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


MODELS = {
    "MOM60": 0.00,
    "MOM60 + 50% Proxy": 0.50,
}


TRANSACTION_COSTS = [
    0.0005,
    0.0010,
    0.0020,
    0.0030,
]


SUBPERIODS = {
    "Full Sample": (
        "2019-10-08",
        "2026-07-13",
    ),
    "Pre-2022": (
        "2019-10-08",
        "2021-12-31",
    ),
    "2022-2023": (
        "2022-01-01",
        "2023-12-31",
    ),
    "2024+": (
        "2024-01-01",
        "2026-07-13",
    ),
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

    universe["ticker"] = (
        universe["ticker"]
        .astype(str)
        .str.zfill(6)
    )

    return universe


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
            ].copy()
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

    return prices


def normalise_returns(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    frame = returns.copy()

    if "date" not in frame.columns:
        index_name = (
            frame.index.name
            if frame.index.name
            else "index"
        )

        frame = (
            frame.reset_index()
            .rename(
                columns={
                    index_name: "date",
                    "index": "date",
                }
            )
        )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    if frame["date"].isna().any():
        raise ValueError(
            "Backtest returns contain invalid dates."
        )

    return frame


def evaluate_subperiods(
    model_name: str,
    transaction_cost: float,
    returns: pd.DataFrame,
    average_turnover: float,
) -> list:
    rows = []

    for period_name, (
        start_date,
        end_date,
    ) in SUBPERIODS.items():
        period_returns = returns.loc[
            (
                returns["date"]
                >= pd.Timestamp(start_date)
            )
            & (
                returns["date"]
                <= pd.Timestamp(end_date)
            )
        ].copy()

        if len(period_returns) < 20:
            continue

        metrics = calculate_performance_metrics(
            period_returns["net_return"]
        )

        rows.append(
            {
                "model": model_name,
                "transaction_cost_bps": (
                    transaction_cost * 10_000
                ),
                "period": period_name,
                "start_date": (
                    period_returns["date"].min()
                ),
                "end_date": (
                    period_returns["date"].max()
                ),
                "observations": len(period_returns),
                **metrics,
                "average_turnover": (
                    average_turnover
                ),
            }
        )

    return rows


def main() -> int:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    universe = load_universe()
    prices = load_prices(universe)

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

    for frame in [momentum, proxy]:
        frame["ticker"] = (
            frame["ticker"]
            .astype(str)
            .str.zfill(6)
        )

    valid_tickers = set(
        universe["ticker"]
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

    rows = []

    for model_name, proxy_weight in MODELS.items():
        signals = combine_momentum_and_proxy(
            momentum=momentum,
            proxy=proxy,
            proxy_weight=proxy_weight,
        )

        for transaction_cost in TRANSACTION_COSTS:
            print(
                f"[RUN] {model_name}, "
                f"cost={transaction_cost * 10_000:.0f} bps"
            )

            result = run_monthly_top_n_backtest(
                prices=prices,
                scored_signals=signals,
                top_n=3,
                transaction_cost_rate=(
                    transaction_cost
                ),
                minimum_signal_assets=10,
            )

            returns = normalise_returns(
                result["returns"]
            )

            rebalances = (
                result["rebalances"].copy()
            )

            if rebalances.empty:
                raise ValueError(
                    f"No rebalances for {model_name}."
                )

            execution_column = (
                "execution_date"
                if "execution_date"
                in rebalances.columns
                else "date"
            )

            first_execution_date = (
                pd.to_datetime(
                    rebalances[
                        execution_column
                    ],
                    errors="coerce",
                ).min()
            )

            returns = returns.loc[
                returns["date"]
                >= first_execution_date
            ].copy()

            average_turnover = (
                rebalances["turnover"].mean()
            )

            rows.extend(
                evaluate_subperiods(
                    model_name=model_name,
                    transaction_cost=(
                        transaction_cost
                    ),
                    returns=returns,
                    average_turnover=(
                        average_turnover
                    ),
                )
            )

    results = pd.DataFrame(rows)

    output_path = (
        OUTPUT_DIRECTORY
        / "proxy_robustness_metrics.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    display = results.copy()

    for column in [
        "cagr",
        "annual_volatility",
        "maximum_drawdown",
        "average_turnover",
    ]:
        display[column] *= 100

    columns = [
        "model",
        "transaction_cost_bps",
        "period",
        "cagr",
        "sharpe",
        "sortino",
        "maximum_drawdown",
        "calmar",
        "average_turnover",
    ]

    print("")
    print("Proxy Robustness Analysis")
    print("=" * 140)

    print(
        display[columns]
        .round(3)
        .to_string(index=False)
    )

    print("")
    print(f"Output: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

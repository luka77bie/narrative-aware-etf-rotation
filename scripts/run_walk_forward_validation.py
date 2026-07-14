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
from src.evaluation.walk_forward import (
    filter_window,
    generate_walk_forward_windows,
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


TRAIN_MONTHS = 36
TEST_MONTHS = 12
STEP_MONTHS = 12

TOP_N = 3
MINIMUM_SIGNAL_ASSETS = 10
TRANSACTION_COST_RATE = 0.001


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

    universe = universe.loc[
        included
    ].copy()

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


def load_price_panel(
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

        required = {
            "date",
            "ticker",
            "adjusted_close",
        }

        missing = required - set(data.columns)

        if missing:
            raise ValueError(
                f"{ticker} price data is missing columns: "
                + ", ".join(sorted(missing))
            )

        frame = data[
            [
                "date",
                "ticker",
                "adjusted_close",
            ]
        ].copy()

        frames.append(frame)

        print(
            f"[LOADED] {ticker} {row.name}: "
            f"{len(frame)} rows"
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


def load_signal_inputs():
    if not MOMENTUM_PATH.exists():
        raise FileNotFoundError(
            "Momentum history is missing. Run:\n"
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


def get_first_execution_date(
    rebalances: pd.DataFrame,
) -> pd.Timestamp:
    if rebalances.empty:
        raise ValueError(
            "Backtest generated no rebalances."
        )

    if "execution_date" in rebalances.columns:
        column = "execution_date"

    elif "date" in rebalances.columns:
        column = "date"

    elif "signal_date" in rebalances.columns:
        column = "signal_date"

    else:
        raise ValueError(
            "Rebalances contain no execution-date column."
        )

    first_execution_date = pd.to_datetime(
        rebalances[column],
        errors="coerce",
    ).min()

    if pd.isna(first_execution_date):
        raise ValueError(
            "Unable to determine first execution date."
        )

    return first_execution_date


def evaluate_fold(
    model_name: str,
    proxy_weight: float,
    fold,
    prices: pd.DataFrame,
    momentum: pd.DataFrame,
    proxy: pd.DataFrame,
):
    """
    Run one strictly out-of-sample fold.

    Signals before test_start may be retained as historical context,
    but only test-period returns are evaluated.
    """
    signals = combine_momentum_and_proxy(
        momentum=momentum,
        proxy=proxy,
        proxy_weight=proxy_weight,
    )

    # Keep all historical signal information available up to test_end.
    # This preserves trailing calculations and pre-test portfolio state.
    fold_signals = signals.loc[
        signals["date"] <= fold.test_end
    ].copy()

    fold_prices = prices.loc[
        prices["date"] <= fold.test_end
    ].copy()

    result = run_monthly_top_n_backtest(
        prices=fold_prices,
        scored_signals=fold_signals,
        top_n=TOP_N,
        transaction_cost_rate=TRANSACTION_COST_RATE,
        minimum_signal_assets=MINIMUM_SIGNAL_ASSETS,
    )

    returns = normalise_returns(
        result["returns"]
    )

    rebalances = result[
        "rebalances"
    ].copy()

    first_execution_date = (
        get_first_execution_date(
            rebalances
        )
    )

    effective_test_start = max(
        fold.test_start,
        first_execution_date,
    )

    oos_returns = returns.loc[
        (
            returns["date"]
            >= effective_test_start
        )
        & (
            returns["date"]
            <= fold.test_end
        )
    ].copy()

    if oos_returns.empty:
        return None

    oos_rebalances = rebalances.copy()

    if "execution_date" in oos_rebalances.columns:
        rebalance_date_column = (
            "execution_date"
        )

    elif "date" in oos_rebalances.columns:
        rebalance_date_column = "date"

    else:
        rebalance_date_column = (
            "signal_date"
        )

    oos_rebalances[
        rebalance_date_column
    ] = pd.to_datetime(
        oos_rebalances[
            rebalance_date_column
        ],
        errors="coerce",
    )

    oos_rebalances = oos_rebalances.loc[
        (
            oos_rebalances[
                rebalance_date_column
            ]
            >= effective_test_start
        )
        & (
            oos_rebalances[
                rebalance_date_column
            ]
            <= fold.test_end
        )
    ].copy()

    metrics = calculate_performance_metrics(
        oos_returns["net_return"]
    )

    average_turnover = (
        oos_rebalances["turnover"].mean()
        if (
            not oos_rebalances.empty
            and "turnover"
            in oos_rebalances.columns
        )
        else 0.0
    )

    metric_row = {
        "model": model_name,
        "proxy_weight": proxy_weight,
        "fold": fold.fold,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "test_start": effective_test_start,
        "test_end": fold.test_end,
        "return_observations": len(
            oos_returns
        ),
        "rebalance_count": len(
            oos_rebalances
        ),
        **metrics,
        "average_turnover": (
            average_turnover
        ),
    }

    oos_returns["model"] = model_name
    oos_returns["fold"] = fold.fold
    oos_returns["test_start"] = (
        effective_test_start
    )
    oos_returns["test_end"] = (
        fold.test_end
    )

    return {
        "metrics": metric_row,
        "returns": oos_returns,
    }


def main() -> int:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    universe = load_universe()
    prices = load_price_panel(
        universe
    )

    momentum, proxy = (
        load_signal_inputs()
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

    overlap = overlap.dropna(
        subset=[
            "mom_60",
            "z_mom_60",
        ]
    )

    if overlap.empty:
        raise ValueError(
            "Momentum and proxy signals have no "
            "valid overlapping observations."
        )

    daily_coverage = (
        overlap.groupby("date")["ticker"]
        .nunique()
    )

    eligible_dates = daily_coverage.loc[
        daily_coverage
        >= MINIMUM_SIGNAL_ASSETS
    ]

    if eligible_dates.empty:
        raise ValueError(
            "No date has sufficient overlapping assets."
        )

    evaluation_start = (
        eligible_dates.index.min()
    )

    evaluation_end = (
        overlap["date"].max()
    )

    windows = generate_walk_forward_windows(
        start_date=str(
            pd.Timestamp(
                evaluation_start
            ).date()
        ),
        end_date=str(
            pd.Timestamp(
                evaluation_end
            ).date()
        ),
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
    )

    if not windows:
        raise ValueError(
            "No walk-forward windows were generated."
        )

    print("")
    print("Walk-Forward Validation Configuration")
    print("=" * 100)
    print(
        f"Evaluation range: "
        f"{pd.Timestamp(evaluation_start).date()} "
        f"to "
        f"{pd.Timestamp(evaluation_end).date()}"
    )
    print(f"Training months: {TRAIN_MONTHS}")
    print(f"Test months: {TEST_MONTHS}")
    print(f"Step months: {STEP_MONTHS}")
    print(f"Folds: {len(windows)}")
    print(f"Top N: {TOP_N}")
    print(
        "Transaction cost: "
        f"{TRANSACTION_COST_RATE * 10_000:.0f} bps"
    )
    print("")

    metric_rows = []
    return_frames = []

    for model_name, proxy_weight in (
        MODELS.items()
    ):
        for fold in windows:
            print(
                f"[RUN] {model_name} | "
                f"Fold {fold.fold} | "
                f"Test "
                f"{fold.test_start.date()} "
                f"to {fold.test_end.date()}"
            )

            result = evaluate_fold(
                model_name=model_name,
                proxy_weight=proxy_weight,
                fold=fold,
                prices=prices,
                momentum=momentum,
                proxy=proxy,
            )

            if result is None:
                print(
                    "  [SKIP] No OOS returns."
                )
                continue

            metric_rows.append(
                result["metrics"]
            )

            return_frames.append(
                result["returns"]
            )

    if not metric_rows:
        raise ValueError(
            "No walk-forward fold produced returns."
        )

    fold_metrics = pd.DataFrame(
        metric_rows
    )

    oos_returns = pd.concat(
        return_frames,
        ignore_index=True,
    )

    duplicate_keys = oos_returns.duplicated(
        subset=[
            "model",
            "date",
        ],
        keep=False,
    )

    if duplicate_keys.any():
        duplicates = oos_returns.loc[
            duplicate_keys,
            [
                "model",
                "date",
                "fold",
            ],
        ]

        raise ValueError(
            "Overlapping OOS test returns detected:\n"
            + duplicates.head(20).to_string(
                index=False
            )
        )

    aggregate_rows = []

    for model_name, group in (
        oos_returns.groupby("model")
    ):
        group = group.sort_values(
            "date"
        )

        metrics = calculate_performance_metrics(
            group["net_return"]
        )

        model_fold_metrics = (
            fold_metrics.loc[
                fold_metrics["model"]
                == model_name
            ]
        )

        aggregate_rows.append(
            {
                "model": model_name,
                "oos_start": (
                    group["date"].min()
                ),
                "oos_end": (
                    group["date"].max()
                ),
                "oos_observations": len(
                    group
                ),
                "fold_count": (
                    model_fold_metrics[
                        "fold"
                    ].nunique()
                ),
                "positive_cagr_folds": int(
                    (
                        model_fold_metrics[
                            "cagr"
                        ] > 0
                    ).sum()
                ),
                "positive_sharpe_folds": int(
                    (
                        model_fold_metrics[
                            "sharpe"
                        ] > 0
                    ).sum()
                ),
                "mean_fold_sharpe": (
                    model_fold_metrics[
                        "sharpe"
                    ].mean()
                ),
                "median_fold_sharpe": (
                    model_fold_metrics[
                        "sharpe"
                    ].median()
                ),
                **metrics,
                "average_turnover": (
                    model_fold_metrics[
                        "average_turnover"
                    ].mean()
                ),
            }
        )

    aggregate_metrics = pd.DataFrame(
        aggregate_rows
    )

    folds_path = (
        OUTPUT_DIRECTORY
        / "walk_forward_fold_metrics.csv"
    )

    returns_path = (
        OUTPUT_DIRECTORY
        / "walk_forward_oos_returns.csv"
    )

    aggregate_path = (
        OUTPUT_DIRECTORY
        / "walk_forward_aggregate_metrics.csv"
    )

    fold_metrics.to_csv(
        folds_path,
        index=False,
    )

    oos_returns.to_csv(
        returns_path,
        index=False,
    )

    aggregate_metrics.to_csv(
        aggregate_path,
        index=False,
    )

    fold_display = fold_metrics.copy()

    for column in [
        "cagr",
        "annual_volatility",
        "maximum_drawdown",
        "average_turnover",
    ]:
        fold_display[column] *= 100

    print("")
    print("Walk-Forward Fold Metrics")
    print("=" * 150)

    print(
        fold_display[
            [
                "model",
                "fold",
                "test_start",
                "test_end",
                "cagr",
                "sharpe",
                "maximum_drawdown",
                "calmar",
                "rebalance_count",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    aggregate_display = (
        aggregate_metrics.copy()
    )

    for column in [
        "cagr",
        "annual_volatility",
        "maximum_drawdown",
        "average_turnover",
    ]:
        aggregate_display[column] *= 100

    print("")
    print("Aggregate OOS Metrics")
    print("=" * 130)

    print(
        aggregate_display[
            [
                "model",
                "fold_count",
                "cagr",
                "sharpe",
                "sortino",
                "maximum_drawdown",
                "calmar",
                "mean_fold_sharpe",
                "median_fold_sharpe",
                "positive_sharpe_folds",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("")
    print(f"Fold metrics: {folds_path}")
    print(f"OOS returns: {returns_path}")
    print(
        f"Aggregate metrics: "
        f"{aggregate_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

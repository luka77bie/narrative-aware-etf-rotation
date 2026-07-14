import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.market_data import (
    load_cached_price_data,
)
from src.risk.metrics import (
    calculate_risk_adjusted_score,
    calculate_rolling_risk_metrics,
)


UNIVERSE_PATH = Path(
    "config/etf_universe.csv"
)

MOMENTUM_SIGNAL_PATH = Path(
    "outputs/momentum_signal_history.csv"
)

OUTPUT_DIRECTORY = Path("outputs")


def load_non_cash_universe() -> pd.DataFrame:
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


def load_price_panel(
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

    return pd.concat(
        frames,
        ignore_index=True,
    )


def main() -> int:
    if not MOMENTUM_SIGNAL_PATH.exists():
        raise FileNotFoundError(
            "Momentum signal history is missing. Run:\n"
            "python3 scripts/run_momentum_signal.py"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    universe = load_non_cash_universe()
    prices = load_price_panel(universe)

    momentum = pd.read_csv(
        MOMENTUM_SIGNAL_PATH,
        dtype={"ticker": "string"},
        parse_dates=["date"],
    )

    momentum = momentum.loc[
        momentum["ticker"].isin(
            universe["ticker"]
        )
    ].copy()

    risk_metrics = calculate_rolling_risk_metrics(
        prices=prices,
        window=60,
        price_column="adjusted_close",
    )

    risk_columns = [
        "date",
        "ticker",
        "volatility_60",
        "downside_volatility_60",
        "drawdown_60",
    ]

    combined = momentum.merge(
        risk_metrics[risk_columns],
        on=[
            "date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    combined = calculate_risk_adjusted_score(
        data=combined,
        momentum_column="z_mom_60",
        volatility_column="volatility_60",
        downside_column="downside_volatility_60",
        weights=(
            1.0,
            0.25,
            0.15,
        ),
    )

    # Make the signal compatible with the existing backtest engine.
    combined["momentum_score"] = (
        combined["risk_adjusted_score"]
    )

    combined["momentum_rank"] = (
        combined["risk_adjusted_rank"]
    )

    output_path = (
        OUTPUT_DIRECTORY
        / "risk_adjusted_signal_history.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    complete = combined.dropna(
        subset=[
            "mom_60",
            "volatility_60",
            "downside_volatility_60",
            "risk_adjusted_score",
        ]
    )

    latest_date = complete["date"].max()

    latest = (
        complete.loc[
            complete["date"] == latest_date
        ]
        .sort_values("risk_adjusted_rank")
        .reset_index(drop=True)
    )

    latest_path = (
        OUTPUT_DIRECTORY
        / "latest_risk_adjusted_ranking.csv"
    )

    latest.to_csv(
        latest_path,
        index=False,
    )

    print("")
    print(
        f"Latest signal date: "
        f"{pd.Timestamp(latest_date).date()}"
    )

    print("")
    print("Top 10 Risk-Adjusted Ranking")
    print("=" * 110)

    columns = [
        "risk_adjusted_rank",
        "ticker",
        "name",
        "secondary_theme",
        "mom_60",
        "volatility_60",
        "downside_volatility_60",
        "risk_adjusted_score",
    ]

    print(
        latest[columns]
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    print("")
    print(f"History: {output_path}")
    print(f"Latest ranking: {latest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import sys
from math import ceil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.market_data import load_cached_price_data
from src.signals.momentum import (
    calculate_momentum,
    calculate_momentum_scores,
)


UNIVERSE_PATH = Path("config/etf_universe.csv")
OUTPUT_DIRECTORY = Path("outputs")


def load_research_universe() -> pd.DataFrame:
    """Load included non-cash ETFs for momentum ranking."""
    universe = pd.read_csv(
        UNIVERSE_PATH,
        dtype={"ticker": "string"},
    )

    included_mask = (
        universe["include"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    universe = universe.loc[included_mask].copy()

    # Money-market ETF is retained as a cash reserve asset,
    # but excluded from the equity/sector momentum ranking.
    research_universe = universe.loc[
        universe["style"].str.lower() != "cash"
    ].copy()

    if research_universe.empty:
        raise ValueError(
            "No non-cash ETFs available for momentum ranking."
        )

    return research_universe


def load_price_panel(
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Load all ETF caches into one long-format price panel."""
    frames = []
    failures = []

    metadata_columns = [
        "ticker",
        "name",
        "primary_theme",
        "secondary_theme",
        "style",
    ]

    metadata = universe[
        metadata_columns
    ].copy()

    for row in universe.itertuples(index=False):
        ticker = str(row.ticker).zfill(6)

        try:
            data = load_cached_price_data(
                ticker=ticker,
                cache_directory="data/raw",
            )

            frames.append(data)

            print(
                f"[LOADED] {ticker} {row.name}: "
                f"{len(data)} rows"
            )

        except Exception as exc:
            failures.append(
                {
                    "ticker": ticker,
                    "name": row.name,
                    "error": str(exc),
                }
            )

            print(
                f"[FAILED] {ticker} {row.name}: {exc}"
            )

    if failures:
        failure_path = (
            OUTPUT_DIRECTORY
            / "momentum_load_failures.csv"
        )

        pd.DataFrame(failures).to_csv(
            failure_path,
            index=False,
        )

    if not frames:
        raise RuntimeError(
            "No ETF price caches could be loaded."
        )

    prices = pd.concat(
        frames,
        ignore_index=True,
    )

    prices = prices.merge(
        metadata,
        on="ticker",
        how="left",
        validate="many_to_one",
    )

    return prices


def select_latest_valid_cross_section(
    scored_data: pd.DataFrame,
    expected_assets: int,
    minimum_coverage: float = 0.80,
) -> pd.DataFrame:
    """
    Select the latest date containing sufficient complete ETF signals.

    This avoids ranking only one or two ETFs when some files have
    a later date than the rest of the universe.
    """
    complete = scored_data.dropna(
        subset=[
            "mom_20",
            "mom_60",
            "momentum_score",
        ]
    ).copy()

    if complete.empty:
        raise ValueError(
            "No complete momentum signals available."
        )

    minimum_assets = ceil(
        expected_assets * minimum_coverage
    )

    coverage = (
        complete.groupby("date")["ticker"]
        .nunique()
        .sort_index()
    )

    valid_dates = coverage.loc[
        coverage >= minimum_assets
    ]

    if valid_dates.empty:
        raise ValueError(
            "No date meets the minimum cross-sectional "
            f"coverage requirement of {minimum_assets} ETFs."
        )

    latest_date = valid_dates.index.max()

    ranking = (
        complete.loc[
            complete["date"] == latest_date
        ]
        .sort_values(
            [
                "momentum_score",
                "ticker",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    ranking["momentum_rank"] = (
        range(1, len(ranking) + 1)
    )

    print("")
    print(
        f"Latest valid signal date: "
        f"{pd.Timestamp(latest_date).date()}"
    )
    print(
        f"Cross-sectional coverage: "
        f"{len(ranking)}/{expected_assets}"
    )

    return ranking


def main() -> int:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    universe = load_research_universe()

    print(
        f"Momentum research universe: "
        f"{len(universe)} ETFs"
    )

    prices = load_price_panel(universe)

    momentum = calculate_momentum(
        prices=prices,
        lookbacks=(20, 60),
        price_column="adjusted_close",
    )

    scored = calculate_momentum_scores(
        momentum_data=momentum,
        lookbacks=(20, 60),
        weights=(0.5, 0.5),
    )

    ranking = select_latest_valid_cross_section(
        scored_data=scored,
        expected_assets=len(universe),
        minimum_coverage=0.80,
    )

    output_columns = [
        "date",
        "momentum_rank",
        "ticker",
        "name",
        "primary_theme",
        "secondary_theme",
        "adjusted_close",
        "mom_20",
        "mom_60",
        "z_mom_20",
        "z_mom_60",
        "momentum_score",
    ]

    ranking_output = ranking[
        output_columns
    ].copy()

    percentage_columns = [
        "mom_20",
        "mom_60",
    ]

    for column in percentage_columns:
        ranking_output[column] = (
            ranking_output[column] * 100
        )

    ranking_output = ranking_output.rename(
        columns={
            "mom_20": "mom_20_pct",
            "mom_60": "mom_60_pct",
        }
    )

    ranking_path = (
        OUTPUT_DIRECTORY
        / "latest_momentum_ranking.csv"
    )

    ranking_output.to_csv(
        ranking_path,
        index=False,
    )

    history_path = (
        OUTPUT_DIRECTORY
        / "momentum_signal_history.csv"
    )

    scored.to_csv(
        history_path,
        index=False,
    )

    display_columns = [
        "momentum_rank",
        "ticker",
        "name",
        "secondary_theme",
        "mom_20_pct",
        "mom_60_pct",
        "momentum_score",
    ]

    print("")
    print("Top 10 Momentum Ranking")
    print("=" * 100)

    print(
        ranking_output[
            display_columns
        ]
        .head(10)
        .round(
            {
                "mom_20_pct": 2,
                "mom_60_pct": 2,
                "momentum_score": 3,
            }
        )
        .to_string(index=False)
    )

    print("")
    print(f"Ranking output: {ranking_path}")
    print(f"Signal history: {history_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

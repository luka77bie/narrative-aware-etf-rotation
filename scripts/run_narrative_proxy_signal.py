import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.market_data import (
    load_cached_price_data,
)
from src.narrative.proxy import (
    calculate_narrative_proxy_scores,
    engineer_market_attention_features,
)


UNIVERSE_PATH = Path(
    "config/etf_universe.csv"
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
                    "volume",
                    "turnover",
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
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    universe = load_non_cash_universe()
    prices = load_price_panel(universe)

    features = engineer_market_attention_features(
        price_data=prices,
        short_window=20,
        long_window=60,
    )

    scored = calculate_narrative_proxy_scores(
        feature_data=features,
    )

    metadata = universe[
        [
            "ticker",
            "name",
            "primary_theme",
            "secondary_theme",
        ]
    ]

    scored = scored.merge(
        metadata,
        on="ticker",
        how="left",
        validate="many_to_one",
    )

    complete = scored.dropna(
        subset=[
            "turnover_growth",
            "volume_growth",
            "attention_momentum",
            "volatility_expansion",
            "narrative_proxy_score",
        ]
    ).copy()

    if complete.empty:
        raise ValueError(
            "No complete Narrative Proxy observations."
        )

    latest_date = complete["date"].max()

    latest = (
        complete.loc[
            complete["date"] == latest_date
        ]
        .sort_values(
            "narrative_proxy_rank"
        )
        .reset_index(drop=True)
    )

    history_path = (
        OUTPUT_DIRECTORY
        / "narrative_proxy_signal_history.csv"
    )

    latest_path = (
        OUTPUT_DIRECTORY
        / "latest_narrative_proxy_ranking.csv"
    )

    scored.to_csv(
        history_path,
        index=False,
    )

    latest.to_csv(
        latest_path,
        index=False,
    )

    print("")
    print(
        f"Latest Narrative Proxy Date: "
        f"{pd.Timestamp(latest_date).date()}"
    )

    print("")
    print("Top 10 Narrative Proxy Ranking")
    print("=" * 120)

    columns = [
        "narrative_proxy_rank",
        "ticker",
        "name",
        "secondary_theme",
        "turnover_growth",
        "volume_growth",
        "attention_momentum",
        "volatility_expansion",
        "narrative_proxy_score",
    ]

    print(
        latest[columns]
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    print("")
    print(f"History: {history_path}")
    print(f"Latest ranking: {latest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

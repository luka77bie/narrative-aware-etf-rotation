import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.narrative.mapping import (
    expand_theme_ticker_mapping,
    map_narrative_scores_to_etfs,
)


MOMENTUM_PATH = Path(
    "outputs/momentum_signal_history.csv"
)

NARRATIVE_PATH = Path(
    "outputs/narrative_signal_history.csv"
)

THEME_PATH = Path(
    "config/narrative_themes.csv"
)

OUTPUT_DIRECTORY = Path("outputs")


def main() -> int:
    if not MOMENTUM_PATH.exists():
        raise FileNotFoundError(
            "Momentum signal history is missing."
        )

    if not NARRATIVE_PATH.exists():
        raise FileNotFoundError(
            "Narrative signal history is missing."
        )

    momentum = pd.read_csv(
        MOMENTUM_PATH,
        dtype={"ticker": "string"},
        parse_dates=["date"],
    )

    narrative = pd.read_csv(
        NARRATIVE_PATH,
        dtype={"theme_id": "string"},
        parse_dates=["date"],
    )

    themes = pd.read_csv(
        THEME_PATH,
        dtype={
            "theme_id": "string",
            "mapped_tickers": "string",
        },
    )

    include_mask = (
        themes["include"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    themes = themes.loc[include_mask].copy()

    mapping = expand_theme_ticker_mapping(
        themes
    )

    etf_narrative = map_narrative_scores_to_etfs(
        narrative_scores=narrative,
        theme_mapping=mapping,
    )

    combined = momentum.merge(
        etf_narrative,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )

    combined["narrative_score"] = (
        combined["narrative_score"]
        .fillna(0.0)
    )

    # MOM60 is the validated Quant Baseline.
    combined["composite_score"] = (
        0.70 * combined["z_mom_60"]
        + 0.30 * combined["narrative_score"]
    )

    combined["composite_rank"] = (
        combined.groupby("date")[
            "composite_score"
        ]
        .rank(
            ascending=False,
            method="first",
        )
    )

    complete = combined.dropna(
        subset=[
            "mom_60",
            "z_mom_60",
            "composite_score",
        ]
    ).copy()

    if complete.empty:
        raise ValueError(
            "No complete composite signals are available."
        )

    latest_date = complete["date"].max()

    latest = (
        complete.loc[
            complete["date"] == latest_date
        ]
        .sort_values("composite_rank")
        .reset_index(drop=True)
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        OUTPUT_DIRECTORY
        / "composite_signal_history.csv"
    )

    latest_path = (
        OUTPUT_DIRECTORY
        / "latest_composite_ranking.csv"
    )

    combined.to_csv(
        history_path,
        index=False,
    )

    latest.to_csv(
        latest_path,
        index=False,
    )

    print("")
    print(
        f"Latest Composite Signal Date: "
        f"{pd.Timestamp(latest_date).date()}"
    )

    print("")
    print("Top 10 Composite Ranking")
    print("=" * 110)

    display_columns = [
        "composite_rank",
        "ticker",
        "name",
        "secondary_theme",
        "mom_60",
        "z_mom_60",
        "narrative_score",
        "composite_score",
    ]

    print(
        latest[
            display_columns
        ]
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

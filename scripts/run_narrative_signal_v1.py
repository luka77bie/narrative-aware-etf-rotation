import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.narrative.scoring import (
    calculate_narrative_scores,
    engineer_narrative_features,
)


DEFAULT_DATA_PATH = Path(
    "data/sample/narrative/narrative_features.csv"
)

THEME_PATH = Path(
    "config/narrative_themes.csv"
)

OUTPUT_DIRECTORY = Path("outputs")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Narrative Signal V1 from structured "
            "theme-level observations."
        )
    )

    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Path to raw narrative feature CSV.",
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=7,
        help="Rolling narrative feature lookback.",
    )

    return parser.parse_args()


def load_theme_metadata() -> pd.DataFrame:
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

    return themes.loc[
        include_mask
    ].copy()


def main() -> int:
    args = parse_arguments()

    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Narrative data not found: {data_path}"
        )

    raw_data = pd.read_csv(
        data_path,
        dtype={"theme_id": "string"},
    )

    themes = load_theme_metadata()

    raw_data = raw_data.loc[
        raw_data["theme_id"].isin(
            themes["theme_id"]
        )
    ].copy()

    if raw_data.empty:
        raise ValueError(
            "No narrative observations match "
            "the configured themes."
        )

    features = engineer_narrative_features(
        data=raw_data,
        lookback=args.lookback,
    )

    scored = calculate_narrative_scores(
        feature_data=features,
        weights={
            "news_growth": 0.40,
            "policy_intensity": 0.30,
            "attention_change": 0.30,
        },
    )

    scored = scored.merge(
        themes[
            [
                "theme_id",
                "theme_name",
                "primary_theme",
                "mapped_tickers",
            ]
        ],
        on="theme_id",
        how="left",
        validate="many_to_one",
    )

    complete = scored.dropna(
        subset=[
            "news_growth",
            "policy_intensity",
            "attention_change",
            "narrative_score",
        ]
    ).copy()

    if complete.empty:
        raise ValueError(
            "No complete Narrative Scores are available."
        )

    latest_date = complete["date"].max()

    latest = (
        complete.loc[
            complete["date"] == latest_date
        ]
        .sort_values(
            [
                "narrative_rank",
                "theme_id",
            ]
        )
        .reset_index(drop=True)
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        OUTPUT_DIRECTORY
        / "narrative_signal_history.csv"
    )

    latest_path = (
        OUTPUT_DIRECTORY
        / "latest_narrative_ranking.csv"
    )

    scored.to_csv(
        history_path,
        index=False,
    )

    latest.to_csv(
        latest_path,
        index=False,
    )

    display_columns = [
        "narrative_rank",
        "theme_id",
        "theme_name",
        "news_growth",
        "policy_intensity",
        "attention_change",
        "narrative_score",
        "mapped_tickers",
    ]

    print("")
    print(
        f"Latest Narrative Signal Date: "
        f"{pd.Timestamp(latest_date).date()}"
    )

    print("")
    print("Narrative Theme Ranking")
    print("=" * 120)

    print(
        latest[
            display_columns
        ]
        .round(
            {
                "news_growth": 4,
                "policy_intensity": 2,
                "attention_change": 4,
                "narrative_score": 4,
            }
        )
        .to_string(index=False)
    )

    print("")
    print(f"History: {history_path}")
    print(f"Latest ranking: {latest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

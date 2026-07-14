import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.narrative.aggregation import (
    aggregate_daily_news_features,
    build_complete_daily_panel,
)
from src.narrative.provenance import (
    load_news_dataset,
    validate_news_provenance,
)


THEME_PATH = Path(
    "config/narrative_themes.csv"
)

OUTPUT_DIRECTORY = Path(
    "data/processed/news"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and aggregate real historical news "
            "into daily theme features."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to real historical news CSV.",
    )

    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional panel start date YYYY-MM-DD.",
    )

    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional panel end date YYYY-MM-DD.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    themes = pd.read_csv(
        THEME_PATH,
        dtype={"theme_id": "string"},
    )

    include_mask = (
        themes["include"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    theme_ids = (
        themes.loc[
            include_mask,
            "theme_id",
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    raw_news = load_news_dataset(
        args.data
    )

    validated_news = validate_news_provenance(
        data=raw_news,
        allowed_theme_ids=theme_ids,
        formal_research=True,
    )

    if validated_news.empty:
        raise ValueError(
            "The validated news dataset is empty."
        )

    aggregated = aggregate_daily_news_features(
        validated_news
    )

    start_date = (
        args.start_date
        or aggregated["date"].min().date().isoformat()
    )

    end_date = (
        args.end_date
        or aggregated["date"].max().date().isoformat()
    )

    panel = build_complete_daily_panel(
        aggregated_features=aggregated,
        theme_ids=theme_ids,
        start_date=start_date,
        end_date=end_date,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregated_path = (
        OUTPUT_DIRECTORY
        / "daily_news_features_observed.csv"
    )

    panel_path = (
        OUTPUT_DIRECTORY
        / "daily_news_features_panel.csv"
    )

    aggregated.to_csv(
        aggregated_path,
        index=False,
    )

    panel.to_csv(
        panel_path,
        index=False,
    )

    print("Daily News Feature Build")
    print("=" * 70)
    print(
        "Validated articles:",
        len(validated_news),
    )
    print(
        "Observed theme-days:",
        len(aggregated),
    )
    print(
        "Complete panel rows:",
        len(panel),
    )
    print(
        "Date range:",
        panel["date"].min().date(),
        "to",
        panel["date"].max().date(),
    )
    print(
        "Themes:",
        panel["theme_id"].nunique(),
    )
    print(f"Observed output: {aggregated_path}")
    print(f"Panel output: {panel_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

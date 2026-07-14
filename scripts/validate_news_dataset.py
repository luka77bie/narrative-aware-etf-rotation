import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.narrative.provenance import (
    load_news_dataset,
    validate_news_provenance,
)


THEME_PATH = Path(
    "config/narrative_themes.csv"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate historical news provenance "
            "before Narrative scoring."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to historical news CSV.",
    )

    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help=(
            "Allow synthetic data for software testing only."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    themes = pd.read_csv(
        THEME_PATH,
        dtype={"theme_id": "string"},
    )

    included = (
        themes["include"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    allowed_theme_ids = set(
        themes.loc[
            included,
            "theme_id",
        ]
        .dropna()
        .astype(str)
    )

    data = load_news_dataset(
        args.data
    )

    validated = validate_news_provenance(
        data=data,
        allowed_theme_ids=allowed_theme_ids,
        formal_research=not args.allow_synthetic,
    )

    print("News Dataset Validation")
    print("=" * 70)
    print(f"Rows: {len(validated)}")
    print(
        "Unique articles:",
        validated["article_id"].nunique(),
    )
    print(
        "Unique sources:",
        validated["source"].nunique(),
    )
    print(
        "Themes:",
        validated["theme_id"].nunique(),
    )
    print(
        "Published range:",
        validated["published_at"].min(),
        "to",
        validated["published_at"].max(),
    )
    print("")
    print("Validation status: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

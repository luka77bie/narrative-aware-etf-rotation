import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.narrative.policy_archive import (
    aggregate_daily_policy_features,
    load_policy_archive,
    validate_policy_archive,
)
from src.narrative.source_registry import (
    audit_source_readiness,
    load_and_validate_source_registry,
)


THEME_PATH = Path(
    "config/narrative_themes.csv"
)

OUTPUT_DIRECTORY = Path(
    "data/processed/policy"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an official policy archive export."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to official policy archive CSV.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    themes = pd.read_csv(
        THEME_PATH,
        dtype={"theme_id": "string"},
    )

    allowed_theme_ids = set(
        themes.loc[
            themes["include"]
            .astype(str)
            .str.lower()
            .eq("true"),
            "theme_id",
        ]
        .dropna()
        .astype(str)
    )

    registry = (
        load_and_validate_source_registry(
            "config/news_sources.csv"
        )
    )

    audited = audit_source_readiness(
        registry
    )

    allowed_source_ids = set(
        audited.loc[
            audited["ready_for_ingestion"],
            "source_id",
        ]
        .dropna()
        .astype(str)
    )

    if not allowed_source_ids:
        raise ValueError(
            "No official source is approved for policy ingestion."
        )

    raw_policy = load_policy_archive(
        args.data
    )

    validated = validate_policy_archive(
        data=raw_policy,
        allowed_theme_ids=allowed_theme_ids,
        allowed_source_ids=allowed_source_ids,
    )

    daily_features = (
        aggregate_daily_policy_features(
            validated
        )
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    validated_path = (
        OUTPUT_DIRECTORY
        / "validated_policy_archive.csv"
    )

    features_path = (
        OUTPUT_DIRECTORY
        / "daily_policy_features.csv"
    )

    validated.to_csv(
        validated_path,
        index=False,
    )

    daily_features.to_csv(
        features_path,
        index=False,
    )

    print("Official Policy Archive Build")
    print("=" * 70)
    print(f"Documents: {len(validated)}")
    print(
        "Themes:",
        validated["theme_id"].nunique(),
    )
    print(
        "Authorities:",
        validated[
            "issuing_authority"
        ].nunique(),
    )
    print(f"Validated output: {validated_path}")
    print(f"Daily features: {features_path}")
    print("Status: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from src.narrative.policy_signal import (
    build_policy_theme_panel,
    calculate_policy_narrative_score,
    engineer_policy_signal_features,
)


DEFAULT_POLICY_PATH = Path(
    "data/raw/policy/"
    "manual_state_council_gazette.csv"
)

THEME_PATH = Path(
    "config/narrative_themes.csv"
)

OUTPUT_DIRECTORY = Path("outputs")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Policy Narrative Signal V1 from "
            "manually verified official metadata."
        )
    )

    parser.add_argument(
        "--data",
        default=str(DEFAULT_POLICY_PATH),
        help="Validated manual policy metadata CSV.",
    )

    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional panel start date.",
    )

    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional panel end date.",
    )

    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Explicitly mark output as pipeline validation only."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if not args.validation_only:
        raise RuntimeError(
            "The current source is not approved for formal "
            "research. Run with --validation-only."
        )

    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Policy metadata not found: {data_path}"
        )

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

    theme_ids = (
        themes.loc[
            included,
            "theme_id",
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    raw_policy = load_policy_archive(
        data_path
    )

    validated = validate_policy_archive(
        data=raw_policy,
        allowed_theme_ids=set(theme_ids),
        allowed_source_ids={
            "state_council_gazette"
        },
    )

    daily = aggregate_daily_policy_features(
        validated
    )

    start_date = (
        args.start_date
        or daily["date"].min().date().isoformat()
    )

    end_date = (
        args.end_date
        or daily["date"].max().date().isoformat()
    )

    panel = build_policy_theme_panel(
        daily_policy_features=daily,
        theme_ids=theme_ids,
        start_date=start_date,
        end_date=end_date,
    )

    features = engineer_policy_signal_features(
        policy_panel=panel,
        short_window=30,
        long_window=90,
    )

    scored = calculate_policy_narrative_score(
        features
    )

    scored["research_status"] = (
        "pipeline_validation_only"
    )

    complete = scored.dropna(
        subset=[
            "policy_narrative_score",
            "policy_narrative_rank",
        ]
    ).copy()

    latest_date = complete["date"].max()

    latest = (
        complete.loc[
            complete["date"] == latest_date
        ]
        .sort_values(
            "policy_narrative_rank"
        )
        .reset_index(drop=True)
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        OUTPUT_DIRECTORY
        / "policy_narrative_signal_history.csv"
    )

    latest_path = (
        OUTPUT_DIRECTORY
        / "latest_policy_narrative_ranking.csv"
    )

    daily_path = (
        OUTPUT_DIRECTORY
        / "observed_daily_policy_features.csv"
    )

    daily.to_csv(
        daily_path,
        index=False,
    )

    scored.to_csv(
        history_path,
        index=False,
    )

    latest.to_csv(
        latest_path,
        index=False,
    )

    print("Policy Narrative Signal V1")
    print("=" * 100)
    print(f"Validated documents: {len(validated)}")
    print(
        f"Date range: {start_date} to {end_date}"
    )
    print(f"Themes in panel: {len(theme_ids)}")
    print(
        "Research status: "
        "PIPELINE VALIDATION ONLY"
    )

    print("")
    print("Latest Policy Theme Ranking")
    print("=" * 100)

    display_columns = [
        "policy_narrative_rank",
        "theme_id",
        "policy_intensity_30",
        "policy_breadth_30",
        "policy_acceleration",
        "policy_narrative_score",
    ]

    print(
        latest[
            display_columns
        ]
        .round(4)
        .to_string(index=False)
    )

    print("")
    print(f"Daily observed features: {daily_path}")
    print(f"Signal history: {history_path}")
    print(f"Latest ranking: {latest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

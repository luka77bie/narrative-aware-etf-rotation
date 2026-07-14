import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.narrative.source_approval import (
    apply_source_review_to_registry,
    load_source_review,
    validate_source_review,
)
from src.narrative.source_registry import (
    load_and_validate_source_registry,
)


REGISTRY_PATH = Path(
    "config/news_sources.csv"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review documentary evidence before "
            "approving an official data source."
        )
    )

    parser.add_argument(
        "--review",
        required=True,
        help="Path to source-review JSON.",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Apply an approved review to "
            "config/news_sources.csv."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    review = load_source_review(
        args.review
    )

    validated = validate_source_review(
        review
    )

    print("Official Source Review")
    print("=" * 70)
    print(
        json.dumps(
            {
                key: (
                    value.isoformat()
                    if isinstance(
                        value,
                        pd.Timestamp,
                    )
                    else value
                )
                for key, value
                in validated.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if not args.apply:
        print("")
        print(
            "Review validated but not applied."
        )
        return 0

    registry = (
        load_and_validate_source_registry(
            REGISTRY_PATH
        )
    )

    updated = apply_source_review_to_registry(
        registry=registry,
        review=validated,
    )

    updated.to_csv(
        REGISTRY_PATH,
        index=False,
    )

    print("")
    print(
        "Source registry updated:",
        REGISTRY_PATH,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

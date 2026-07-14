import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.narrative.policy_archive import (
    load_policy_archive,
    validate_policy_archive,
)


DEFAULT_SOURCE_ID = "state_council_gazette"

OUTPUT_DIRECTORY = Path(
    "data/raw/policy"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import manually collected metadata from an "
            "official policy archive. This command does not "
            "download webpages or generate policy records."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "CSV created from manually verified official "
            "policy metadata."
        ),
    )

    parser.add_argument(
        "--source-id",
        default=DEFAULT_SOURCE_ID,
        help="Official source identifier.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    input_path = Path(args.input)

    raw = load_policy_archive(
        input_path
    )

    source_ids = set(
        raw["source_id"]
        .dropna()
        .astype(str)
    )

    if source_ids != {args.source_id}:
        raise ValueError(
            "Input source_id values must exactly match "
            f"{args.source_id}; found {sorted(source_ids)}"
        )

    # This stage validates document integrity only.
    # Formal approval is still checked by the downstream
    # normalise_policy_archive.py workflow.
    validated = validate_policy_archive(
        data=raw,
        allowed_source_ids={
            args.source_id
        },
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIRECTORY
        / f"manual_{args.source_id}.csv"
    )

    validated.to_csv(
        output_path,
        index=False,
    )

    print("Manual Official Policy Metadata Import")
    print("=" * 70)
    print(f"Source ID: {args.source_id}")
    print(f"Rows: {len(validated)}")
    print(
        "Date range:",
        validated["published_at"].min(),
        "to",
        validated["published_at"].max(),
    )
    print(f"Output: {output_path}")
    print("")
    print(
        "Status: METADATA VALIDATED, "
        "NOT APPROVED FOR FORMAL RESEARCH"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.narrative.ingestion import (
    build_ingestion_manifest,
    normalise_news_export,
    save_manifest,
)
from src.narrative.source_registry import (
    get_approved_source,
    load_and_validate_source_registry,
)
from src.narrative.provenance import (
    validate_news_provenance,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalise a real historical news export "
            "into the canonical project schema."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to real source export CSV.",
    )

    parser.add_argument(
        "--source-id",
        required=True,
        help="Enabled source_id in config/news_sources.csv.",
    )

    parser.add_argument(
        "--column-map",
        required=True,
        help=(
            "Path to JSON mapping canonical fields "
            "to raw export columns."
        ),
    )

    parser.add_argument(
        "--retrieved-at",
        required=True,
        help=(
            "Actual retrieval timestamp in ISO-8601 format."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    input_path = Path(args.input)
    map_path = Path(args.column_map)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    if not map_path.exists():
        raise FileNotFoundError(
            f"Column map not found: {map_path}"
        )

    registry = load_and_validate_source_registry(
        "config/news_sources.csv"
    )

    source_record = get_approved_source(
        registry=registry,
        source_id=args.source_id,
    )

    column_map = json.loads(
        map_path.read_text(
            encoding="utf-8",
        )
    )

    raw_data = pd.read_csv(
        input_path
    )

    normalised = normalise_news_export(
        raw_data=raw_data,
        source_record=source_record,
        column_map=column_map,
        retrieved_at=args.retrieved_at,
    )

    themes = pd.read_csv(
        "config/narrative_themes.csv",
        dtype={"theme_id": "string"},
    )

    enabled_themes = set(
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

    validated = validate_news_provenance(
        data=normalised,
        allowed_theme_ids=enabled_themes,
        formal_research=True,
    )

    output_directory = Path(
        "data/raw/news"
    )

    manifest_directory = Path(
        "data/raw/news/manifests"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / f"validated_{args.source_id}.csv"
    )

    manifest_path = (
        manifest_directory
        / f"{args.source_id}_manifest.json"
    )

    validated.to_csv(
        output_path,
        index=False,
    )

    manifest = build_ingestion_manifest(
        source_record=source_record,
        input_path=input_path,
        output_path=output_path,
        normalised_data=validated,
        retrieved_at=args.retrieved_at,
    )

    save_manifest(
        manifest,
        manifest_path,
    )

    print("News Export Normalisation")
    print("=" * 70)
    print(f"Source: {args.source_id}")
    print(f"Rows: {len(validated)}")
    print(f"Output: {output_path}")
    print(f"Manifest: {manifest_path}")
    print("Status: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

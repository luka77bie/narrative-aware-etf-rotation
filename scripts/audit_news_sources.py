import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.narrative.source_registry import (
    audit_source_readiness,
    load_and_validate_source_registry,
)


def main() -> int:
    registry = load_and_validate_source_registry(
        "config/news_sources.csv"
    )

    audit = audit_source_readiness(
        registry
    )

    display_columns = [
        "source_id",
        "source_name",
        "data_access_method",
        "license_status",
        "enabled",
        "approved_for_research",
        "ready_for_ingestion",
        "readiness_reason",
    ]

    print("Historical News Source Audit")
    print("=" * 120)

    print(
        audit[display_columns]
        .to_string(index=False)
    )

    ready_count = int(
        audit["ready_for_ingestion"].sum()
    )

    print("")
    print(f"Sources registered: {len(audit)}")
    print(f"Sources ready: {ready_count}")

    if ready_count == 0:
        print(
            "No source is currently approved for ingestion."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

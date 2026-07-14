from pathlib import Path

import pandas as pd
import pytest

from src.narrative.source_registry import (
    audit_source_readiness,
    get_approved_source,
    load_and_validate_source_registry,
)


def make_registry(
    approved: bool = True,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": ["source_a"],
            "source_name": ["Source A"],
            "source_category": [
                "official_government"
            ],
            "base_domain": ["example.gov.cn"],
            "language": ["zh"],
            "data_access_method": [
                "manual_export"
            ],
            "requires_license": [False],
            "license_status": [
                "public_access"
            ],
            "robots_reviewed": [approved],
            "terms_reviewed": [approved],
            "historical_start_date": [
                "2020-01-01"
            ],
            "enabled": [approved],
            "approved_for_research": [
                approved
            ],
            "notes": ["Test source"],
        }
    )


def test_ready_source_passes_audit() -> None:
    result = audit_source_readiness(
        make_registry(approved=True)
    )

    assert result.iloc[0][
        "ready_for_ingestion"
    ]

    assert result.iloc[0][
        "readiness_reason"
    ] == "READY"


def test_unapproved_source_fails_audit() -> None:
    result = audit_source_readiness(
        make_registry(approved=False)
    )

    assert not result.iloc[0][
        "ready_for_ingestion"
    ]

    assert "not approved" in result.iloc[0][
        "readiness_reason"
    ]


def test_get_approved_source_rejects_disabled() -> None:
    registry = make_registry(
        approved=False
    )

    with pytest.raises(
        ValueError,
        match="not ready",
    ):
        get_approved_source(
            registry,
            "source_a",
        )


def test_registry_file_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sources.csv"

    make_registry(
        approved=True
    ).to_csv(
        path,
        index=False,
    )

    result = load_and_validate_source_registry(
        path
    )

    assert len(result) == 1
    assert result.iloc[0]["enabled"]


def test_duplicate_source_id_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sources.csv"

    data = pd.concat(
        [
            make_registry(True),
            make_registry(True),
        ],
        ignore_index=True,
    )

    data.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate source_id",
    ):
        load_and_validate_source_registry(
            path
        )

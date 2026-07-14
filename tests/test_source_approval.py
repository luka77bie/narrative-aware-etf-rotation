import pandas as pd
import pytest

from src.narrative.source_approval import (
    apply_source_review_to_registry,
    validate_source_review,
)


def make_review(
    permitted: bool = True,
) -> dict:
    return {
        "source_id": "gov_cn",
        "reviewed_at": (
            "2026-07-14T10:00:00Z"
        ),
        "reviewer": "Researcher",
        "access_method": "manual_export",
        "terms_url": (
            "https://example.gov.cn/terms"
        ),
        "robots_url": (
            "https://example.gov.cn/robots.txt"
        ),
        "research_use_permitted": permitted,
        "local_storage_permitted": permitted,
        "text_analysis_permitted": permitted,
        "historical_export_available": permitted,
        "review_notes": (
            "Documented source review."
        ),
    }


def make_registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": ["gov_cn"],
            "source_name": ["Government"],
            "robots_reviewed": [False],
            "terms_reviewed": [False],
            "enabled": [False],
            "approved_for_research": [False],
        }
    )


def test_complete_review_passes() -> None:
    result = validate_source_review(
        make_review(True)
    )

    assert result["ready_for_research"]
    assert len(
        result["evidence_sha256"]
    ) == 64


def test_negative_permission_blocks_approval() -> None:
    result = validate_source_review(
        make_review(False)
    )

    assert not result[
        "ready_for_research"
    ]


def test_missing_review_field_fails() -> None:
    review = make_review()
    del review["terms_url"]

    with pytest.raises(
        ValueError,
        match="missing fields",
    ):
        validate_source_review(review)


def test_approved_review_updates_registry() -> None:
    result = apply_source_review_to_registry(
        registry=make_registry(),
        review=make_review(True),
    )

    assert result.iloc[0][
        "approved_for_research"
    ]

    assert result.iloc[0]["enabled"]


def test_unapproved_review_keeps_source_disabled() -> None:
    result = apply_source_review_to_registry(
        registry=make_registry(),
        review=make_review(False),
    )

    assert not result.iloc[0][
        "approved_for_research"
    ]

    assert not result.iloc[0][
        "enabled"
    ]

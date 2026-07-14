import pandas as pd
import pytest

from src.narrative.policy_archive import (
    validate_policy_archive,
)


def make_realistic_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "document_id": [
                "gazette-example-001",
            ],
            "published_at": [
                "2024-01-01T00:00:00Z",
            ],
            "retrieved_at": [
                "2026-07-14T00:00:00Z",
            ],
            "issuing_authority": [
                "国务院办公厅",
            ],
            "source_id": [
                "state_council_gazette",
            ],
            "url": [
                "https://www.gov.cn/example-policy"
            ],
            "title": [
                "Official policy metadata example"
            ],
            "summary": [""],
            "full_text": [""],
            "theme_id": ["ai"],
            "document_type": ["policy"],
        }
    )


def test_manual_metadata_validation() -> None:
    result = validate_policy_archive(
        make_realistic_metadata(),
        allowed_source_ids={
            "state_council_gazette"
        },
        allowed_theme_ids={"ai"},
    )

    assert len(result) == 1
    assert (
        result.iloc[0]["source_id"]
        == "state_council_gazette"
    )


def test_wrong_source_is_rejected() -> None:
    data = make_realistic_metadata()
    data.loc[0, "source_id"] = "unapproved_source"

    with pytest.raises(
        ValueError,
        match="Unapproved",
    ):
        validate_policy_archive(
            data,
            allowed_source_ids={
                "state_council_gazette"
            },
        )

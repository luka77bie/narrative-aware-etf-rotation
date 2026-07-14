import pandas as pd
import pytest

from src.narrative.policy_archive import (
    aggregate_daily_policy_features,
    validate_policy_archive,
)


def make_policy_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "document_id": [
                "",
                "official-002",
            ],
            "published_at": [
                "2024-01-01T08:00:00Z",
                "2024-01-01T10:00:00Z",
            ],
            "retrieved_at": [
                "2024-01-02T08:00:00Z",
                "2024-01-02T08:00:00Z",
            ],
            "issuing_authority": [
                "Authority A",
                "Authority B",
            ],
            "source_id": [
                "gov_source",
                "gov_source",
            ],
            "url": [
                "https://gov.example.cn/policy/1",
                "https://gov.example.cn/policy/2",
            ],
            "title": [
                "AI policy",
                "AI industry guideline",
            ],
            "summary": [
                "",
                "Official summary",
            ],
            "full_text": [
                "",
                "Official full text",
            ],
            "theme_id": [
                "ai",
                "ai",
            ],
            "document_type": [
                "policy",
                "guideline",
            ],
        }
    )


def test_valid_policy_archive_passes() -> None:
    result = validate_policy_archive(
        make_policy_data(),
        allowed_theme_ids={"ai"},
        allowed_source_ids={"gov_source"},
    )

    assert len(result) == 2
    assert result["document_id"].ne("").all()


def test_missing_policy_id_is_generated() -> None:
    result = validate_policy_archive(
        make_policy_data()
    )

    generated_id = result.loc[
        result["title"] == "AI policy",
        "document_id",
    ].iloc[0]

    assert len(generated_id) == 64


def test_unapproved_source_fails() -> None:
    with pytest.raises(
        ValueError,
        match="Unapproved",
    ):
        validate_policy_archive(
            make_policy_data(),
            allowed_source_ids={"different_source"},
        )


def test_invalid_document_type_fails() -> None:
    data = make_policy_data()

    data.loc[
        0,
        "document_type",
    ] = "blog"

    with pytest.raises(
        ValueError,
        match="Unsupported",
    ):
        validate_policy_archive(data)


def test_policy_daily_aggregation() -> None:
    validated = validate_policy_archive(
        make_policy_data()
    )

    result = aggregate_daily_policy_features(
        validated
    )

    assert len(result) == 1
    assert result.iloc[0]["policy_count"] == 2
    assert (
        result.iloc[0]["issuing_authority_count"]
        == 2
    )

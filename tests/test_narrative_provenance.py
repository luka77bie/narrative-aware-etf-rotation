import pandas as pd
import pytest

from src.narrative.provenance import (
    apply_point_in_time_filter,
    validate_news_provenance,
)


def make_valid_news() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": [
                "article_001",
                "article_002",
            ],
            "published_at": [
                "2024-01-01T08:00:00Z",
                "2024-01-02T09:00:00Z",
            ],
            "retrieved_at": [
                "2024-01-01T09:00:00Z",
                "2024-01-02T10:00:00Z",
            ],
            "source": [
                "Official News Agency",
                "Exchange Announcement",
            ],
            "url": [
                "https://example.com/article-001",
                "https://example.com/article-002",
            ],
            "title": [
                "AI policy update",
                "Gold market update",
            ],
            "content": [
                "Verified historical article content.",
                "Verified historical article content.",
            ],
            "language": [
                "en",
                "en",
            ],
            "theme_id": [
                "ai",
                "gold",
            ],
            "match_method": [
                "keyword",
                "manual",
            ],
            "is_policy_source": [
                True,
                False,
            ],
        }
    )


def test_valid_news_passes() -> None:
    result = validate_news_provenance(
        make_valid_news(),
        allowed_theme_ids={
            "ai",
            "gold",
        },
    )

    assert len(result) == 2
    assert result["published_at"].notna().all()


def test_duplicate_article_id_fails() -> None:
    data = make_valid_news()

    data.loc[1, "article_id"] = (
        data.loc[0, "article_id"]
    )

    with pytest.raises(
        ValueError,
        match="Duplicate article_id",
    ):
        validate_news_provenance(data)


def test_retrieved_before_published_fails() -> None:
    data = make_valid_news()

    data.loc[
        0,
        "retrieved_at",
    ] = "2023-12-31T08:00:00Z"

    with pytest.raises(
        ValueError,
        match="retrieved_at cannot",
    ):
        validate_news_provenance(data)


def test_synthetic_source_rejected_in_formal_mode() -> None:
    data = make_valid_news()

    data.loc[
        0,
        "source",
    ] = "synthetic_sample"

    with pytest.raises(
        ValueError,
        match="Synthetic or generated",
    ):
        validate_news_provenance(
            data,
            formal_research=True,
        )


def test_synthetic_source_allowed_for_testing() -> None:
    data = make_valid_news()

    data.loc[
        0,
        "source",
    ] = "synthetic_sample"

    result = validate_news_provenance(
        data,
        formal_research=False,
    )

    assert len(result) == 2


def test_unknown_theme_fails() -> None:
    data = make_valid_news()

    data.loc[0, "theme_id"] = "unknown"

    with pytest.raises(
        ValueError,
        match="Unknown theme_id",
    ):
        validate_news_provenance(
            data,
            allowed_theme_ids={
                "ai",
                "gold",
            },
        )


def test_point_in_time_filter() -> None:
    data = validate_news_provenance(
        make_valid_news(),
        formal_research=True,
    )

    result = apply_point_in_time_filter(
        data,
        as_of="2024-01-01T12:00:00Z",
    )

    assert len(result) == 1
    assert (
        result.iloc[0]["article_id"]
        == "article_001"
    )

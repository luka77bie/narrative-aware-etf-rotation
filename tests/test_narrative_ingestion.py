from pathlib import Path

import pandas as pd
import pytest

from src.narrative.ingestion import (
    build_ingestion_manifest,
    canonicalise_url,
    generate_article_id,
    normalise_news_export,
)


def make_source_record(
    enabled: bool = True,
) -> pd.Series:
    return pd.Series(
        {
            "source_id": "official_source",
            "source_name": "Official Source",
            "source_category": "official_news_agency",
            "base_domain": "example.com",
            "language": "zh",
            "requires_license": False,
            "enabled": enabled,
        }
    )


def make_raw_export() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "publish_time": [
                "2024-01-01T08:00:00Z",
            ],
            "article_url": [
                "https://example.com/news/1?utm_source=test",
            ],
            "headline": [
                "Verified headline",
            ],
            "body": [
                "Verified article body.",
            ],
            "theme": [
                "ai",
            ],
        }
    )


def test_canonicalise_url_removes_tracking() -> None:
    result = canonicalise_url(
        "https://Example.com/news/1/"
        "?utm_source=test&id=10#section"
    )

    assert result == (
        "https://example.com/news/1?id=10"
    )


def test_article_id_is_deterministic() -> None:
    first = generate_article_id(
        source_id="source",
        canonical_url="https://example.com/a",
        published_at="2024-01-01T08:00:00Z",
    )

    second = generate_article_id(
        source_id="source",
        canonical_url="https://example.com/a",
        published_at="2024-01-01T08:00:00Z",
    )

    assert first == second


def test_disabled_source_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="not enabled",
    ):
        normalise_news_export(
            raw_data=make_raw_export(),
            source_record=make_source_record(
                enabled=False
            ),
            column_map={
                "published_at": "publish_time",
                "url": "article_url",
                "title": "headline",
                "content": "body",
                "theme_id": "theme",
            },
            retrieved_at="2024-01-02T08:00:00Z",
        )


def test_real_export_is_normalised() -> None:
    result = normalise_news_export(
        raw_data=make_raw_export(),
        source_record=make_source_record(),
        column_map={
            "published_at": "publish_time",
            "url": "article_url",
            "title": "headline",
            "content": "body",
            "theme_id": "theme",
        },
        retrieved_at="2024-01-02T08:00:00Z",
    )

    assert len(result) == 1
    assert result.iloc[0]["theme_id"] == "ai"
    assert "utm_source" not in result.iloc[0]["url"]
    assert len(result.iloc[0]["article_id"]) == 64


def test_manifest_uses_real_file_hash(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "export.csv"
    output_path = tmp_path / "normalised.csv"

    input_path.write_text(
        "real exported content",
        encoding="utf-8",
    )

    data = normalise_news_export(
        raw_data=make_raw_export(),
        source_record=make_source_record(),
        column_map={
            "published_at": "publish_time",
            "url": "article_url",
            "title": "headline",
            "content": "body",
            "theme_id": "theme",
        },
        retrieved_at="2024-01-02T08:00:00Z",
    )

    manifest = build_ingestion_manifest(
        source_record=make_source_record(),
        input_path=input_path,
        output_path=output_path,
        normalised_data=data,
        retrieved_at="2024-01-02T08:00:00Z",
    )

    assert len(manifest["input_sha256"]) == 64
    assert manifest["row_count"] == 1

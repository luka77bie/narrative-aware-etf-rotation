import pandas as pd

from src.narrative.aggregation import (
    aggregate_daily_news_features,
    build_complete_daily_panel,
)


def make_validated_news() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": [
                "a1",
                "a2",
                "a3",
            ],
            "published_at": pd.to_datetime(
                [
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T05:00:00Z",
                    "2024-01-02T03:00:00Z",
                ],
                utc=True,
            ),
            "source": [
                "Source A",
                "Source B",
                "Source A",
            ],
            "source_category": [
                "official_government",
                "financial_media",
                "financial_media",
            ],
            "theme_id": [
                "ai",
                "ai",
                "gold",
            ],
            "is_policy_source": [
                True,
                False,
                False,
            ],
        }
    )


def test_daily_news_aggregation() -> None:
    result = aggregate_daily_news_features(
        make_validated_news()
    )

    ai = result.loc[
        result["theme_id"] == "ai"
    ].iloc[0]

    assert ai["news_count"] == 2
    assert ai["policy_count"] == 1
    assert ai["unique_sources"] == 2


def test_source_weights_are_applied() -> None:
    result = aggregate_daily_news_features(
        make_validated_news()
    )

    ai = result.loc[
        result["theme_id"] == "ai"
    ].iloc[0]

    assert ai["weighted_news_count"] == 2.5


def test_complete_panel_fills_missing_days() -> None:
    aggregated = aggregate_daily_news_features(
        make_validated_news()
    )

    panel = build_complete_daily_panel(
        aggregated_features=aggregated,
        theme_ids=[
            "ai",
            "gold",
        ],
        start_date="2024-01-01",
        end_date="2024-01-03",
    )

    assert len(panel) == 6

    missing_day = panel.loc[
        (panel["date"] == pd.Timestamp("2024-01-03"))
        & (panel["theme_id"] == "ai")
    ].iloc[0]

    assert missing_day["news_count"] == 0
    assert missing_day["policy_count"] == 0


def test_duplicate_articles_are_rejected() -> None:
    data = make_validated_news()

    data.loc[1, "article_id"] = "a1"

    try:
        aggregate_daily_news_features(data)

    except ValueError as exc:
        assert "duplicate article_id" in str(exc)

    else:
        raise AssertionError(
            "Duplicate article IDs should be rejected."
        )

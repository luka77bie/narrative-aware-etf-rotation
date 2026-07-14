import pandas as pd

from src.narrative.mapping import (
    expand_theme_ticker_mapping,
    map_narrative_scores_to_etfs,
)


def make_themes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "theme_id": ["ai", "gold"],
            "theme_name": ["人工智能", "黄金"],
            "mapped_tickers": [
                "159363|515880",
                "518880",
            ],
        }
    )


def test_expand_theme_ticker_mapping() -> None:
    result = expand_theme_ticker_mapping(
        make_themes()
    )

    assert len(result) == 3
    assert "159363" in result["ticker"].tolist()
    assert "518880" in result["ticker"].tolist()


def test_map_narrative_scores_to_etfs() -> None:
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-31",
                    "2024-01-31",
                ]
            ),
            "theme_id": ["ai", "gold"],
            "narrative_score": [1.2, -0.4],
        }
    )

    mapping = expand_theme_ticker_mapping(
        make_themes()
    )

    result = map_narrative_scores_to_etfs(
        narrative_scores=scores,
        theme_mapping=mapping,
    )

    assert len(result) == 3

    ai_etf = result.loc[
        result["ticker"] == "159363"
    ].iloc[0]

    assert ai_etf["narrative_score"] == 1.2


def test_multiple_themes_are_averaged() -> None:
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-31",
                    "2024-01-31",
                ]
            ),
            "theme_id": ["theme_a", "theme_b"],
            "narrative_score": [1.0, 3.0],
        }
    )

    mapping = pd.DataFrame(
        {
            "theme_id": [
                "theme_a",
                "theme_b",
            ],
            "theme_name": [
                "Theme A",
                "Theme B",
            ],
            "ticker": [
                "510300",
                "510300",
            ],
        }
    )

    result = map_narrative_scores_to_etfs(
        scores,
        mapping,
    )

    assert result.iloc[0]["narrative_score"] == 2.0
    assert result.iloc[0]["narrative_theme_count"] == 2

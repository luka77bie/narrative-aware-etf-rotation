from typing import Dict, Optional

import pandas as pd


DEFAULT_SOURCE_WEIGHTS = {
    "official_government": 1.50,
    "exchange": 1.40,
    "official_news_agency": 1.25,
    "financial_media": 1.00,
    "company_announcement": 1.20,
    "other": 0.75,
}


def normalise_source_category(
    value: str,
) -> str:
    """Normalise source-category labels."""
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
    )


def aggregate_daily_news_features(
    validated_news: pd.DataFrame,
    source_weights: Optional[
        Dict[str, float]
    ] = None,
) -> pd.DataFrame:
    """
    Aggregate validated article-level news into daily theme features.

    This function assumes the input has already passed provenance
    validation. It does not generate or infer missing articles.
    """
    required_columns = {
        "article_id",
        "published_at",
        "source",
        "theme_id",
        "is_policy_source",
    }

    missing = required_columns - set(
        validated_news.columns
    )

    if missing:
        raise ValueError(
            "Validated news data is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = validated_news.copy()

    frame["published_at"] = pd.to_datetime(
        frame["published_at"],
        errors="coerce",
        utc=True,
    )

    if frame["published_at"].isna().any():
        raise ValueError(
            "Validated news data contains invalid "
            "published_at values."
        )

    if frame["article_id"].duplicated().any():
        raise ValueError(
            "Validated news data contains duplicate article_id values."
        )

    if source_weights is None:
        source_weights = DEFAULT_SOURCE_WEIGHTS

    if "source_category" not in frame.columns:
        frame["source_category"] = "other"

    frame["source_category"] = (
        frame["source_category"]
        .map(normalise_source_category)
    )

    frame["source_weight"] = (
        frame["source_category"]
        .map(source_weights)
        .fillna(source_weights.get("other", 0.75))
    )

    frame["date"] = (
        frame["published_at"]
        .dt.tz_convert("Asia/Shanghai")
        .dt.normalize()
        .dt.tz_localize(None)
    )

    frame["is_policy_source"] = (
        frame["is_policy_source"]
        .astype(bool)
    )

    aggregated = (
        frame.groupby(
            ["date", "theme_id"],
            as_index=False,
        )
        .agg(
            news_count=(
                "article_id",
                "nunique",
            ),
            policy_count=(
                "is_policy_source",
                "sum",
            ),
            unique_sources=(
                "source",
                "nunique",
            ),
            weighted_news_count=(
                "source_weight",
                "sum",
            ),
        )
        .sort_values(
            ["date", "theme_id"]
        )
        .reset_index(drop=True)
    )

    aggregated["policy_count"] = (
        aggregated["policy_count"]
        .astype(int)
    )

    return aggregated


def build_complete_daily_panel(
    aggregated_features: pd.DataFrame,
    theme_ids: list,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Build a complete date-theme panel.

    Missing combinations are set to zero because they represent
    no validated articles observed for that theme and date.

    This does not create synthetic articles.
    """
    required = {
        "date",
        "theme_id",
        "news_count",
        "policy_count",
        "unique_sources",
        "weighted_news_count",
    }

    missing = required - set(
        aggregated_features.columns
    )

    if missing:
        raise ValueError(
            "Aggregated news features are missing columns: "
            + ", ".join(sorted(missing))
        )

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
    )

    index = pd.MultiIndex.from_product(
        [
            dates,
            sorted(set(theme_ids)),
        ],
        names=[
            "date",
            "theme_id",
        ],
    )

    panel = (
        aggregated_features
        .set_index(
            ["date", "theme_id"]
        )
        .reindex(index)
        .fillna(
            {
                "news_count": 0,
                "policy_count": 0,
                "unique_sources": 0,
                "weighted_news_count": 0.0,
            }
        )
        .reset_index()
    )

    integer_columns = [
        "news_count",
        "policy_count",
        "unique_sources",
    ]

    for column in integer_columns:
        panel[column] = (
            panel[column]
            .astype(int)
        )

    return panel

from pathlib import Path
from typing import Iterable, Optional, Set, Union

import pandas as pd


REQUIRED_NEWS_COLUMNS = {
    "article_id",
    "published_at",
    "retrieved_at",
    "source",
    "url",
    "title",
    "content",
    "language",
    "theme_id",
    "match_method",
    "is_policy_source",
}


ALLOWED_MATCH_METHODS = {
    "keyword",
    "rule_based",
    "manual",
    "classifier",
}


ALLOWED_LANGUAGES = {
    "zh",
    "en",
}


PROHIBITED_FORMAL_SOURCES = {
    "synthetic_sample",
    "generated",
    "mock",
    "fake",
}


def load_news_dataset(
    path: Union[str, Path],
) -> pd.DataFrame:
    """Load a historical news dataset with stable string types."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"News dataset not found: {file_path}"
        )

    data = pd.read_csv(
        file_path,
        dtype={
            "article_id": "string",
            "source": "string",
            "url": "string",
            "title": "string",
            "content": "string",
            "language": "string",
            "theme_id": "string",
            "match_method": "string",
        },
    )

    return data


def validate_news_provenance(
    data: pd.DataFrame,
    allowed_theme_ids: Optional[Iterable[str]] = None,
    formal_research: bool = True,
) -> pd.DataFrame:
    """
    Validate source provenance and point-in-time integrity.

    formal_research=True rejects synthetic, generated or mock sources.
    """
    missing_columns = (
        REQUIRED_NEWS_COLUMNS
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "News dataset is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    frame = data.copy()

    string_columns = [
        "article_id",
        "source",
        "url",
        "title",
        "content",
        "language",
        "theme_id",
        "match_method",
    ]

    for column in string_columns:
        frame[column] = (
            frame[column]
            .astype("string")
            .str.strip()
        )

    frame["published_at"] = pd.to_datetime(
        frame["published_at"],
        errors="coerce",
        utc=True,
    )

    frame["retrieved_at"] = pd.to_datetime(
        frame["retrieved_at"],
        errors="coerce",
        utc=True,
    )

    frame["is_policy_source"] = (
        frame["is_policy_source"]
        .astype("string")
        .str.lower()
        .map({
            "true": True,
            "false": False,
        })
    )

    empty_required = []

    for column in string_columns:
        if (
            frame[column].isna().any()
            or frame[column].eq("").any()
        ):
            empty_required.append(column)

    if empty_required:
        raise ValueError(
            "News dataset contains empty required fields: "
            + ", ".join(sorted(empty_required))
        )

    if frame["published_at"].isna().any():
        raise ValueError(
            "News dataset contains invalid published_at values."
        )

    if frame["retrieved_at"].isna().any():
        raise ValueError(
            "News dataset contains invalid retrieved_at values."
        )

    invalid_time_order = (
        frame["retrieved_at"]
        < frame["published_at"]
    )

    if invalid_time_order.any():
        raise ValueError(
            "retrieved_at cannot be earlier than published_at."
        )

    if frame["article_id"].duplicated().any():
        duplicate_ids = (
            frame.loc[
                frame["article_id"].duplicated(
                    keep=False
                ),
                "article_id",
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Duplicate article_id values found: "
            f"{duplicate_ids}"
        )

    if frame["url"].duplicated().any():
        duplicate_urls = (
            frame.loc[
                frame["url"].duplicated(
                    keep=False
                ),
                "url",
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Duplicate URLs found: {duplicate_urls}"
        )

    invalid_languages = (
        set(frame["language"].dropna())
        - ALLOWED_LANGUAGES
    )

    if invalid_languages:
        raise ValueError(
            "Unsupported language values: "
            f"{sorted(invalid_languages)}"
        )

    invalid_methods = (
        set(frame["match_method"].dropna())
        - ALLOWED_MATCH_METHODS
    )

    if invalid_methods:
        raise ValueError(
            "Unsupported match_method values: "
            f"{sorted(invalid_methods)}"
        )

    if frame["is_policy_source"].isna().any():
        raise ValueError(
            "is_policy_source must contain true or false."
        )

    invalid_urls = ~frame["url"].str.match(
        r"^https?://",
        na=False,
    )

    if invalid_urls.any():
        raise ValueError(
            "Every news record must contain a valid HTTP(S) URL."
        )

    if allowed_theme_ids is not None:
        allowed_theme_set: Set[str] = {
            str(theme_id)
            for theme_id in allowed_theme_ids
        }

        invalid_themes = (
            set(frame["theme_id"].dropna())
            - allowed_theme_set
        )

        if invalid_themes:
            raise ValueError(
                "Unknown theme_id values: "
                f"{sorted(invalid_themes)}"
            )

    if formal_research:
        normalised_sources = (
            frame["source"]
            .str.lower()
            .str.replace(" ", "_")
        )

        prohibited_mask = normalised_sources.isin(
            PROHIBITED_FORMAL_SOURCES
        )

        if prohibited_mask.any():
            prohibited = (
                frame.loc[
                    prohibited_mask,
                    "source",
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                "Synthetic or generated sources are prohibited "
                f"in formal research datasets: {prohibited}"
            )

    return (
        frame.sort_values(
            [
                "published_at",
                "article_id",
            ]
        )
        .reset_index(drop=True)
    )


def apply_point_in_time_filter(
    data: pd.DataFrame,
    as_of: Union[str, pd.Timestamp],
) -> pd.DataFrame:
    """
    Keep only records known by the specified point in time.

    A record is available only when both:
    published_at <= as_of
    retrieved_at <= as_of
    """
    as_of_timestamp = pd.Timestamp(
        as_of,
        tz="UTC",
    )

    required = {
        "published_at",
        "retrieved_at",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            "Point-in-time filter is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = data.copy()

    frame["published_at"] = pd.to_datetime(
        frame["published_at"],
        errors="coerce",
        utc=True,
    )

    frame["retrieved_at"] = pd.to_datetime(
        frame["retrieved_at"],
        errors="coerce",
        utc=True,
    )

    available = (
        (frame["published_at"] <= as_of_timestamp)
        & (frame["retrieved_at"] <= as_of_timestamp)
    )

    return (
        frame.loc[available]
        .sort_values(
            [
                "published_at",
                "article_id",
            ]
        )
        .reset_index(drop=True)
    )

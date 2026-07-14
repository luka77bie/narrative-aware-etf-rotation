import hashlib
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

from src.narrative.ingestion import canonicalise_url


REQUIRED_POLICY_COLUMNS = {
    "document_id",
    "published_at",
    "retrieved_at",
    "issuing_authority",
    "source_id",
    "url",
    "title",
    "summary",
    "full_text",
    "theme_id",
    "document_type",
}


ALLOWED_DOCUMENT_TYPES = {
    "policy",
    "regulation",
    "notice",
    "plan",
    "guideline",
}


def generate_policy_id(
    source_id: str,
    canonical_url: str,
    published_at: Union[str, pd.Timestamp],
) -> str:
    """Generate a stable ID from genuine policy metadata."""
    timestamp = pd.to_datetime(
        published_at,
        errors="raise",
        utc=True,
    ).isoformat()

    payload = (
        f"{source_id}|{canonical_url}|{timestamp}"
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def load_policy_archive(
    path: Union[str, Path],
) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Policy archive not found: {file_path}"
        )

    return pd.read_csv(
        file_path,
        dtype={
            "document_id": "string",
            "issuing_authority": "string",
            "source_id": "string",
            "url": "string",
            "title": "string",
            "summary": "string",
            "full_text": "string",
            "theme_id": "string",
            "document_type": "string",
        },
        keep_default_na=False,
    )


def validate_policy_archive(
    data: pd.DataFrame,
    allowed_theme_ids: Optional[
        Iterable[str]
    ] = None,
    allowed_source_ids: Optional[
        Iterable[str]
    ] = None,
) -> pd.DataFrame:
    """
    Validate official policy records.

    Missing summary or full_text is allowed.
    Missing title, authority, URL or timestamps is not allowed.
    """
    missing_columns = (
        REQUIRED_POLICY_COLUMNS
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Policy archive is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    frame = data.copy()

    string_columns = [
        "document_id",
        "issuing_authority",
        "source_id",
        "url",
        "title",
        "summary",
        "full_text",
        "theme_id",
        "document_type",
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

    required_text_columns = [
        "issuing_authority",
        "source_id",
        "url",
        "title",
        "theme_id",
        "document_type",
    ]

    invalid_required = []

    for column in required_text_columns:
        if (
            frame[column].isna().any()
            or frame[column].eq("").any()
        ):
            invalid_required.append(column)

    if invalid_required:
        raise ValueError(
            "Policy archive contains empty required fields: "
            + ", ".join(sorted(invalid_required))
        )

    if frame["published_at"].isna().any():
        raise ValueError(
            "Policy archive contains invalid published_at values."
        )

    if frame["retrieved_at"].isna().any():
        raise ValueError(
            "Policy archive contains invalid retrieved_at values."
        )

    if (
        frame["retrieved_at"]
        < frame["published_at"]
    ).any():
        raise ValueError(
            "retrieved_at cannot be earlier than published_at."
        )

    frame["url"] = frame["url"].map(
        canonicalise_url
    )

    invalid_types = (
        set(frame["document_type"])
        - ALLOWED_DOCUMENT_TYPES
    )

    if invalid_types:
        raise ValueError(
            "Unsupported document_type values: "
            f"{sorted(invalid_types)}"
        )

    if allowed_theme_ids is not None:
        allowed_themes = {
            str(value)
            for value in allowed_theme_ids
        }

        invalid_themes = (
            set(frame["theme_id"])
            - allowed_themes
        )

        if invalid_themes:
            raise ValueError(
                "Unknown theme_id values: "
                f"{sorted(invalid_themes)}"
            )

    if allowed_source_ids is not None:
        allowed_sources = {
            str(value)
            for value in allowed_source_ids
        }

        invalid_sources = (
            set(frame["source_id"])
            - allowed_sources
        )

        if invalid_sources:
            raise ValueError(
                "Unapproved policy source_id values: "
                f"{sorted(invalid_sources)}"
            )

    missing_ids = (
        frame["document_id"].isna()
        | frame["document_id"].eq("")
    )

    frame.loc[
        missing_ids,
        "document_id",
    ] = [
        generate_policy_id(
            source_id=row.source_id,
            canonical_url=row.url,
            published_at=row.published_at,
        )
        for row in frame.loc[
            missing_ids
        ].itertuples(index=False)
    ]

    if frame["document_id"].duplicated().any():
        raise ValueError(
            "Policy archive contains duplicate document_id values."
        )

    if frame["url"].duplicated().any():
        raise ValueError(
            "Policy archive contains duplicate URLs."
        )

    return (
        frame.sort_values(
            [
                "published_at",
                "document_id",
            ]
        )
        .reset_index(drop=True)
    )


def aggregate_daily_policy_features(
    validated_policy: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate real policy documents into daily theme features."""
    required = {
        "document_id",
        "published_at",
        "issuing_authority",
        "theme_id",
        "document_type",
    }

    missing = required - set(
        validated_policy.columns
    )

    if missing:
        raise ValueError(
            "Validated policy data is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = validated_policy.copy()

    frame["date"] = (
        pd.to_datetime(
            frame["published_at"],
            utc=True,
        )
        .dt.tz_convert("Asia/Shanghai")
        .dt.normalize()
        .dt.tz_localize(None)
    )

    result = (
        frame.groupby(
            [
                "date",
                "theme_id",
            ],
            as_index=False,
        )
        .agg(
            policy_count=(
                "document_id",
                "nunique",
            ),
            issuing_authority_count=(
                "issuing_authority",
                "nunique",
            ),
            policy_type_count=(
                "document_type",
                "nunique",
            ),
        )
        .sort_values(
            [
                "date",
                "theme_id",
            ]
        )
        .reset_index(drop=True)
    )

    return result

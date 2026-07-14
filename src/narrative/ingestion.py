import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd


CANONICAL_COLUMNS = [
    "article_id",
    "published_at",
    "retrieved_at",
    "source",
    "source_category",
    "url",
    "title",
    "content",
    "language",
    "theme_id",
    "match_method",
    "is_policy_source",
]


TRACKING_QUERY_PREFIXES = (
    "utm_",
    "spm",
    "from",
    "source",
    "ref",
)


def canonicalise_url(url: str) -> str:
    """
    Remove fragments and common tracking query parameters.

    The article URL remains traceable to the original publisher.
    """
    value = str(url).strip()

    if not value:
        raise ValueError("URL cannot be empty.")

    parts = urlsplit(value)

    if parts.scheme not in {"http", "https"}:
        raise ValueError(
            f"Unsupported URL scheme: {parts.scheme}"
        )

    filtered_query = []

    for key, query_value in parse_qsl(
        parts.query,
        keep_blank_values=True,
    ):
        normalised_key = key.lower()

        if normalised_key.startswith(
            TRACKING_QUERY_PREFIXES
        ):
            continue

        filtered_query.append(
            (key, query_value)
        )

    path = parts.path.rstrip("/") or "/"

    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            path,
            urlencode(filtered_query),
            "",
        )
    )


def generate_article_id(
    source_id: str,
    canonical_url: str,
    published_at: Union[str, pd.Timestamp],
) -> str:
    """
    Generate a deterministic ID from real source metadata.

    No random or synthetic identifier is used.
    """
    timestamp = pd.to_datetime(
        published_at,
        errors="raise",
        utc=True,
    ).isoformat()

    payload = (
        f"{source_id}|{canonical_url}|{timestamp}"
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()

    return digest


def load_source_registry(
    path: Union[str, Path] = "config/news_sources.csv",
) -> pd.DataFrame:
    registry_path = Path(path)

    if not registry_path.exists():
        raise FileNotFoundError(
            f"News source registry not found: "
            f"{registry_path}"
        )

    registry = pd.read_csv(
        registry_path,
        dtype={
            "source_id": "string",
            "source_name": "string",
            "source_category": "string",
            "base_domain": "string",
            "language": "string",
        },
    )

    required = {
        "source_id",
        "source_name",
        "source_category",
        "base_domain",
        "language",
        "requires_license",
        "enabled",
    }

    missing = required - set(registry.columns)

    if missing:
        raise ValueError(
            "Source registry is missing columns: "
            + ", ".join(sorted(missing))
        )

    registry["enabled"] = (
        registry["enabled"]
        .astype("string")
        .str.lower()
        .map({
            "true": True,
            "false": False,
        })
    )

    registry["requires_license"] = (
        registry["requires_license"]
        .astype("string")
        .str.lower()
        .map({
            "true": True,
            "false": False,
        })
    )

    if registry["source_id"].duplicated().any():
        raise ValueError(
            "Source registry contains duplicate source_id values."
        )

    if registry["enabled"].isna().any():
        raise ValueError(
            "Source registry enabled must contain true or false."
        )

    return registry


def normalise_news_export(
    raw_data: pd.DataFrame,
    source_record: pd.Series,
    column_map: Dict[str, str],
    retrieved_at: Optional[
        Union[str, pd.Timestamp]
    ] = None,
) -> pd.DataFrame:
    """
    Convert a real source export into the canonical news schema.

    column_map maps canonical field names to raw export field names.
    Example:
        {
            "published_at": "publish_time",
            "url": "article_url",
            "title": "headline",
            "content": "body",
        }
    """
    if not bool(source_record["enabled"]):
        raise ValueError(
            f"News source is not enabled: "
            f"{source_record['source_id']}"
        )

    required_mapping = {
        "published_at",
        "url",
        "title",
        "content",
    }

    missing_mapping = (
        required_mapping - set(column_map)
    )

    if missing_mapping:
        raise ValueError(
            "Column map is missing canonical fields: "
            + ", ".join(sorted(missing_mapping))
        )

    missing_raw_columns = {
        raw_column
        for raw_column in column_map.values()
        if raw_column not in raw_data.columns
    }

    if missing_raw_columns:
        raise ValueError(
            "Raw news export is missing columns: "
            + ", ".join(sorted(missing_raw_columns))
        )

    frame = pd.DataFrame()

    for canonical_name, raw_name in column_map.items():
        frame[canonical_name] = raw_data[raw_name]

    frame["published_at"] = pd.to_datetime(
        frame["published_at"],
        errors="coerce",
        utc=True,
    )

    if frame["published_at"].isna().any():
        raise ValueError(
            "Raw news export contains invalid publication times."
        )

    if retrieved_at is None:
        raise ValueError(
            "retrieved_at must be supplied explicitly "
            "for auditability."
        )

    retrieval_timestamp = pd.to_datetime(
        retrieved_at,
        errors="raise",
        utc=True,
    )

    frame["retrieved_at"] = retrieval_timestamp

    frame["url"] = frame["url"].map(
        canonicalise_url
    )

    source_id = str(
        source_record["source_id"]
    )

    frame["article_id"] = [
        generate_article_id(
            source_id=source_id,
            canonical_url=url,
            published_at=published_at,
        )
        for url, published_at in zip(
            frame["url"],
            frame["published_at"],
        )
    ]

    frame["source"] = source_record[
        "source_name"
    ]

    frame["source_category"] = source_record[
        "source_category"
    ]

    frame["language"] = source_record[
        "language"
    ]

    if "theme_id" not in frame.columns:
        frame["theme_id"] = pd.NA

    if "match_method" not in frame.columns:
        frame["match_method"] = "unclassified"

    if "is_policy_source" not in frame.columns:
        frame["is_policy_source"] = (
            source_record["source_category"]
            == "official_government"
        )

    for column in ["title", "content"]:
        frame[column] = (
            frame[column]
            .astype("string")
            .str.strip()
        )

    if frame["title"].isna().any():
        raise ValueError(
            "Raw news export contains empty titles."
        )

    if frame["content"].isna().any():
        raise ValueError(
            "Raw news export contains empty content."
        )

    return frame[CANONICAL_COLUMNS]


def build_ingestion_manifest(
    source_record: pd.Series,
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    normalised_data: pd.DataFrame,
    retrieved_at: Union[str, pd.Timestamp],
) -> Dict[str, object]:
    """Build an auditable ingestion manifest."""
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input export not found: {input_file}"
        )

    file_digest = hashlib.sha256(
        input_file.read_bytes()
    ).hexdigest()

    manifest = {
        "source_id": str(
            source_record["source_id"]
        ),
        "source_name": str(
            source_record["source_name"]
        ),
        "input_file": str(input_file),
        "input_sha256": file_digest,
        "output_file": str(output_file),
        "retrieved_at": pd.to_datetime(
            retrieved_at,
            utc=True,
        ).isoformat(),
        "row_count": int(
            len(normalised_data)
        ),
        "unique_article_count": int(
            normalised_data[
                "article_id"
            ].nunique()
        ),
        "minimum_published_at": (
            normalised_data[
                "published_at"
            ].min().isoformat()
            if not normalised_data.empty
            else None
        ),
        "maximum_published_at": (
            normalised_data[
                "published_at"
            ].max().isoformat()
            if not normalised_data.empty
            else None
        ),
    }

    return manifest


def save_manifest(
    manifest: Dict[str, object],
    path: Union[str, Path],
) -> None:
    manifest_path = Path(path)

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

from pathlib import Path
from typing import Iterable, Union

import pandas as pd


REQUIRED_SOURCE_COLUMNS = {
    "source_id",
    "source_name",
    "source_category",
    "base_domain",
    "language",
    "data_access_method",
    "requires_license",
    "license_status",
    "robots_reviewed",
    "terms_reviewed",
    "historical_start_date",
    "enabled",
    "approved_for_research",
    "notes",
}


ALLOWED_SOURCE_CATEGORIES = {
    "official_government",
    "exchange",
    "official_news_agency",
    "financial_media",
    "company_announcement",
    "other",
}


ALLOWED_ACCESS_METHODS = {
    "manual_export",
    "licensed_export",
    "official_api",
    "rss",
    "archive_export",
}


ALLOWED_LICENSE_STATUSES = {
    "public_access",
    "licensed",
    "not_acquired",
    "unknown",
}


BOOLEAN_COLUMNS = [
    "requires_license",
    "robots_reviewed",
    "terms_reviewed",
    "enabled",
    "approved_for_research",
]


def parse_boolean_column(
    values: pd.Series,
    column_name: str,
) -> pd.Series:
    """Parse strict true/false values."""
    parsed = (
        values.astype("string")
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )

    if parsed.isna().any():
        raise ValueError(
            f"{column_name} must contain only true or false."
        )

    return parsed


def load_and_validate_source_registry(
    path: Union[str, Path] = "config/news_sources.csv",
) -> pd.DataFrame:
    """Load and structurally validate the source registry."""
    registry_path = Path(path)

    if not registry_path.exists():
        raise FileNotFoundError(
            f"News source registry not found: {registry_path}"
        )

    registry = pd.read_csv(
        registry_path,
        dtype={
            "source_id": "string",
            "source_name": "string",
            "source_category": "string",
            "base_domain": "string",
            "language": "string",
            "data_access_method": "string",
            "license_status": "string",
            "notes": "string",
        },
    )

    missing = (
        REQUIRED_SOURCE_COLUMNS
        - set(registry.columns)
    )

    if missing:
        raise ValueError(
            "Source registry is missing columns: "
            + ", ".join(sorted(missing))
        )

    string_columns = [
        "source_id",
        "source_name",
        "source_category",
        "language",
        "data_access_method",
        "license_status",
    ]

    for column in string_columns:
        registry[column] = (
            registry[column]
            .astype("string")
            .str.strip()
        )

        if (
            registry[column].isna().any()
            or registry[column].eq("").any()
        ):
            raise ValueError(
                f"Source registry contains empty {column} values."
            )

    for column in BOOLEAN_COLUMNS:
        registry[column] = parse_boolean_column(
            registry[column],
            column,
        )

    if registry["source_id"].duplicated().any():
        duplicates = (
            registry.loc[
                registry["source_id"].duplicated(
                    keep=False
                ),
                "source_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Duplicate source_id values found: {duplicates}"
        )

    invalid_categories = (
        set(registry["source_category"])
        - ALLOWED_SOURCE_CATEGORIES
    )

    if invalid_categories:
        raise ValueError(
            "Unsupported source_category values: "
            f"{sorted(invalid_categories)}"
        )

    invalid_methods = (
        set(registry["data_access_method"])
        - ALLOWED_ACCESS_METHODS
    )

    if invalid_methods:
        raise ValueError(
            "Unsupported data_access_method values: "
            f"{sorted(invalid_methods)}"
        )

    invalid_license_statuses = (
        set(registry["license_status"])
        - ALLOWED_LICENSE_STATUSES
    )

    if invalid_license_statuses:
        raise ValueError(
            "Unsupported license_status values: "
            f"{sorted(invalid_license_statuses)}"
        )

    registry["historical_start_date"] = (
        pd.to_datetime(
            registry["historical_start_date"],
            errors="coerce",
        )
    )

    return registry


def audit_source_readiness(
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate whether each source is ready for formal ingestion.

    This does not grant permission. It only checks the documented
    approval fields in the registry.
    """
    frame = registry.copy()

    reasons = []

    for row in frame.itertuples(index=False):
        row_reasons = []

        if not row.robots_reviewed:
            row_reasons.append(
                "robots policy not reviewed"
            )

        if not row.terms_reviewed:
            row_reasons.append(
                "terms not reviewed"
            )

        if (
            row.requires_license
            and row.license_status != "licensed"
        ):
            row_reasons.append(
                "required licence not acquired"
            )

        if not row.approved_for_research:
            row_reasons.append(
                "not approved for research"
            )

        if not row.enabled:
            row_reasons.append(
                "source disabled"
            )

        reasons.append(
            " | ".join(row_reasons)
            if row_reasons
            else "READY"
        )

    frame["readiness_reason"] = reasons

    frame["ready_for_ingestion"] = (
        frame["readiness_reason"]
        == "READY"
    )

    return frame


def get_approved_source(
    registry: pd.DataFrame,
    source_id: str,
) -> pd.Series:
    """Return one source only when fully approved and enabled."""
    audited = audit_source_readiness(registry)

    matched = audited.loc[
        audited["source_id"] == source_id
    ]

    if matched.empty:
        raise ValueError(
            f"Unknown source_id: {source_id}"
        )

    source = matched.iloc[0]

    if not bool(source["ready_for_ingestion"]):
        raise ValueError(
            f"Source is not ready for ingestion: "
            f"{source_id}. "
            f"{source['readiness_reason']}"
        )

    return source

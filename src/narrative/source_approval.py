import hashlib
import json
from pathlib import Path
from typing import Dict, Union

import pandas as pd


REQUIRED_REVIEW_FIELDS = {
    "source_id",
    "reviewed_at",
    "reviewer",
    "access_method",
    "terms_url",
    "robots_url",
    "research_use_permitted",
    "local_storage_permitted",
    "text_analysis_permitted",
    "historical_export_available",
    "review_notes",
}


BOOLEAN_REVIEW_FIELDS = [
    "research_use_permitted",
    "local_storage_permitted",
    "text_analysis_permitted",
    "historical_export_available",
]


def parse_strict_boolean(
    value: object,
    field_name: str,
) -> bool:
    """Parse a strict boolean value."""
    if isinstance(value, bool):
        return value

    normalised = str(value).strip().lower()

    if normalised == "true":
        return True

    if normalised == "false":
        return False

    raise ValueError(
        f"{field_name} must be true or false."
    )


def load_source_review(
    path: Union[str, Path],
) -> Dict[str, object]:
    """Load a source review JSON document."""
    review_path = Path(path)

    if not review_path.exists():
        raise FileNotFoundError(
            f"Source review not found: {review_path}"
        )

    return json.loads(
        review_path.read_text(
            encoding="utf-8",
        )
    )


def validate_source_review(
    review: Dict[str, object],
) -> Dict[str, object]:
    """
    Validate documentary evidence for one source.

    This function does not independently grant legal permission.
    It verifies that the required review evidence is recorded.
    """
    missing = REQUIRED_REVIEW_FIELDS - set(
        review
    )

    if missing:
        raise ValueError(
            "Source review is missing fields: "
            + ", ".join(sorted(missing))
        )

    validated = dict(review)

    text_fields = [
        "source_id",
        "reviewer",
        "access_method",
        "terms_url",
        "robots_url",
        "review_notes",
    ]

    for field in text_fields:
        value = str(
            validated[field]
        ).strip()

        if not value:
            raise ValueError(
                f"{field} cannot be empty."
            )

        validated[field] = value

    validated["reviewed_at"] = pd.to_datetime(
        validated["reviewed_at"],
        errors="coerce",
        utc=True,
    )

    if pd.isna(validated["reviewed_at"]):
        raise ValueError(
            "reviewed_at must be a valid timestamp."
        )

    for field in BOOLEAN_REVIEW_FIELDS:
        validated[field] = parse_strict_boolean(
            validated[field],
            field,
        )

    for field in [
        "terms_url",
        "robots_url",
    ]:
        if not validated[field].startswith(
            ("http://", "https://")
        ):
            raise ValueError(
                f"{field} must be a valid HTTP(S) URL."
            )

    validated["ready_for_research"] = all(
        [
            validated[
                "research_use_permitted"
            ],
            validated[
                "local_storage_permitted"
            ],
            validated[
                "text_analysis_permitted"
            ],
            validated[
                "historical_export_available"
            ],
        ]
    )

    evidence_payload = {
        key: (
            value.isoformat()
            if isinstance(value, pd.Timestamp)
            else value
        )
        for key, value in validated.items()
        if key != "evidence_sha256"
    }

    validated["evidence_sha256"] = (
        hashlib.sha256(
            json.dumps(
                evidence_payload,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )

    return validated


def apply_source_review_to_registry(
    registry: pd.DataFrame,
    review: Dict[str, object],
) -> pd.DataFrame:
    """
    Apply a validated review result to the source registry.

    A source is enabled only when every required permission is true.
    """
    validated_review = validate_source_review(
        review
    )

    source_id = validated_review[
        "source_id"
    ]

    matched = (
        registry["source_id"]
        .astype(str)
        .eq(str(source_id))
    )

    if not matched.any():
        raise ValueError(
            f"Unknown source_id: {source_id}"
        )

    frame = registry.copy()

    ready = bool(
        validated_review[
            "ready_for_research"
        ]
    )

    frame.loc[
        matched,
        "terms_reviewed",
    ] = True

    frame.loc[
        matched,
        "robots_reviewed",
    ] = True

    frame.loc[
        matched,
        "approved_for_research",
    ] = ready

    frame.loc[
        matched,
        "enabled",
    ] = ready

    return frame

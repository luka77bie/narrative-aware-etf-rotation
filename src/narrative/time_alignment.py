from typing import Dict, Optional

import pandas as pd


def normalise_signal_dates(
    data: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """Normalise a signal table to timezone-naive calendar dates."""
    if date_column not in data.columns:
        raise ValueError(
            f"Signal data is missing column: {date_column}"
        )

    frame = data.copy()

    frame[date_column] = pd.to_datetime(
        frame[date_column],
        errors="coerce",
    )

    if frame[date_column].isna().any():
        raise ValueError(
            f"Signal data contains invalid {date_column} values."
        )

    frame[date_column] = (
        frame[date_column]
        .dt.normalize()
    )

    return frame


def audit_duplicate_signal_rows(
    data: pd.DataFrame,
    key_columns: list,
) -> Dict[str, object]:
    """Check whether a signal table has duplicate logical keys."""
    missing = set(key_columns) - set(data.columns)

    if missing:
        raise ValueError(
            "Duplicate audit is missing columns: "
            + ", ".join(sorted(missing))
        )

    duplicate_mask = data.duplicated(
        subset=key_columns,
        keep=False,
    )

    return {
        "duplicate_rows": int(
            duplicate_mask.sum()
        ),
        "has_duplicates": bool(
            duplicate_mask.any()
        ),
    }


def audit_policy_point_in_time(
    policy_data: pd.DataFrame,
    signal_date_column: str = "date",
    publication_column: str = "published_at",
    retrieval_column: str = "retrieved_at",
    market_close_hour: int = 15,
) -> pd.DataFrame:
    """
    Audit policy records against signal dates.

    A policy is usable on signal date t only when both publication
    and retrieval timestamps are no later than market close on t.
    """
    required = {
        signal_date_column,
        publication_column,
        retrieval_column,
    }

    missing = required - set(policy_data.columns)

    if missing:
        raise ValueError(
            "Policy point-in-time audit is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = policy_data.copy()

    frame[signal_date_column] = pd.to_datetime(
        frame[signal_date_column],
        errors="coerce",
    )

    frame[publication_column] = pd.to_datetime(
        frame[publication_column],
        errors="coerce",
        utc=True,
    )

    frame[retrieval_column] = pd.to_datetime(
        frame[retrieval_column],
        errors="coerce",
        utc=True,
    )

    if frame[
        [
            signal_date_column,
            publication_column,
            retrieval_column,
        ]
    ].isna().any().any():
        raise ValueError(
            "Policy point-in-time audit contains invalid timestamps."
        )

    signal_cutoff = (
        frame[signal_date_column]
        .dt.normalize()
        + pd.Timedelta(
            hours=market_close_hour
        )
    )

    signal_cutoff = signal_cutoff.dt.tz_localize(
        "Asia/Shanghai"
    ).dt.tz_convert("UTC")

    frame["signal_cutoff_at"] = signal_cutoff

    frame["published_in_time"] = (
        frame[publication_column]
        <= frame["signal_cutoff_at"]
    )

    frame["retrieved_in_time"] = (
        frame[retrieval_column]
        <= frame["signal_cutoff_at"]
    )

    frame["available_at"] = pd.concat(
        [
            frame[publication_column].rename(
                "published_at"
            ),
            frame[retrieval_column].rename(
                "retrieved_at"
            ),
        ],
        axis=1,
    ).max(axis=1)

    frame["available_in_time"] = (
        frame["available_at"]
        <= frame["signal_cutoff_at"]
    )

    frame["point_in_time_valid"] = (
        frame["published_in_time"]
        & frame["retrieved_in_time"]
        & frame["available_in_time"]
    )

    return frame


def audit_signal_execution_lag(
    signal_dates: pd.Series,
    execution_dates: pd.Series,
) -> pd.DataFrame:
    """
    Ensure execution occurs strictly after signal generation.

    For this project, same-day execution is prohibited.
    """
    signal = pd.to_datetime(
        signal_dates,
        errors="coerce",
    )

    execution = pd.to_datetime(
        execution_dates,
        errors="coerce",
    )

    if signal.isna().any() or execution.isna().any():
        raise ValueError(
            "Signal or execution dates contain invalid values."
        )

    result = pd.DataFrame(
        {
            "signal_date": signal,
            "execution_date": execution,
        }
    )

    result["execution_after_signal"] = (
        result["execution_date"]
        > result["signal_date"]
    )

    result["calendar_lag_days"] = (
        result["execution_date"]
        - result["signal_date"]
    ).dt.days

    return result


def build_alignment_summary(
    momentum: pd.DataFrame,
    proxy: pd.DataFrame,
    policy: pd.DataFrame,
    narrative_v2: pd.DataFrame,
) -> pd.DataFrame:
    """Build a dataset-level time-alignment summary."""
    datasets = {
        "momentum": momentum,
        "market_attention_proxy": proxy,
        "policy_narrative": policy,
        "narrative_v2": narrative_v2,
    }

    rows = []

    for name, data in datasets.items():
        if "date" not in data.columns:
            raise ValueError(
                f"{name} dataset is missing date column."
            )

        dates = pd.to_datetime(
            data["date"],
            errors="coerce",
        )

        if dates.isna().any():
            raise ValueError(
                f"{name} contains invalid dates."
            )

        key_columns = ["date"]

        if "ticker" in data.columns:
            key_columns.append("ticker")

        elif "theme_id" in data.columns:
            key_columns.append("theme_id")

        duplicate_audit = audit_duplicate_signal_rows(
            data,
            key_columns=key_columns,
        )

        rows.append(
            {
                "dataset": name,
                "rows": len(data),
                "first_date": dates.min(),
                "last_date": dates.max(),
                "unique_dates": dates.nunique(),
                "duplicate_rows": duplicate_audit[
                    "duplicate_rows"
                ],
                "has_duplicates": duplicate_audit[
                    "has_duplicates"
                ],
            }
        )

    return pd.DataFrame(rows)

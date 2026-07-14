from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_POLICY_FEATURE_COLUMNS = {
    "date",
    "theme_id",
    "policy_count",
    "issuing_authority_count",
    "policy_type_count",
}


def cross_sectional_zscore(
    values: pd.Series,
) -> pd.Series:
    """Calculate a cross-sectional z-score for one date."""
    standard_deviation = values.std(ddof=0)

    if (
        pd.isna(standard_deviation)
        or np.isclose(standard_deviation, 0.0)
    ):
        return pd.Series(
            0.0,
            index=values.index,
        )

    return (
        values - values.mean()
    ) / standard_deviation


def build_policy_theme_panel(
    daily_policy_features: pd.DataFrame,
    theme_ids: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Build a complete daily theme panel.

    Missing rows receive zero policy observations. This represents
    no validated policy document observed for that theme-date and
    does not create synthetic documents.
    """
    missing = (
        REQUIRED_POLICY_FEATURE_COLUMNS
        - set(daily_policy_features.columns)
    )

    if missing:
        raise ValueError(
            "Daily policy features are missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = daily_policy_features.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    if frame["date"].isna().any():
        raise ValueError(
            "Daily policy features contain invalid dates."
        )

    if frame.duplicated(
        subset=["date", "theme_id"]
    ).any():
        raise ValueError(
            "Daily policy features contain duplicate "
            "date-theme rows."
        )

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
    )

    clean_theme_ids = sorted(
        {
            str(theme_id).strip()
            for theme_id in theme_ids
            if str(theme_id).strip()
        }
    )

    if not clean_theme_ids:
        raise ValueError(
            "At least one theme_id is required."
        )

    complete_index = pd.MultiIndex.from_product(
        [
            dates,
            clean_theme_ids,
        ],
        names=[
            "date",
            "theme_id",
        ],
    )

    panel = (
        frame.set_index(
            ["date", "theme_id"]
        )
        .reindex(complete_index)
        .fillna(
            {
                "policy_count": 0,
                "issuing_authority_count": 0,
                "policy_type_count": 0,
            }
        )
        .reset_index()
    )

    integer_columns = [
        "policy_count",
        "issuing_authority_count",
        "policy_type_count",
    ]

    for column in integer_columns:
        panel[column] = panel[column].astype(int)

    return panel


def engineer_policy_signal_features(
    policy_panel: pd.DataFrame,
    short_window: int = 30,
    long_window: int = 90,
) -> pd.DataFrame:
    """
    Engineer trailing policy-intensity features.

    policy_intensity_30:
        Number of validated policy documents in the trailing
        short window.

    policy_breadth_30:
        Rolling sum of authority and document-type breadth.

    policy_acceleration:
        Short-window policy frequency relative to the
        long-window daily average.
    """
    if short_window <= 1:
        raise ValueError(
            "short_window must be greater than 1."
        )

    if long_window <= short_window:
        raise ValueError(
            "long_window must be greater than short_window."
        )

    required = {
        "date",
        "theme_id",
        "policy_count",
        "issuing_authority_count",
        "policy_type_count",
    }

    missing = required - set(policy_panel.columns)

    if missing:
        raise ValueError(
            "Policy panel is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = policy_panel.copy()

    frame["date"] = pd.to_datetime(frame["date"])

    frame = frame.sort_values(
        ["theme_id", "date"]
    ).reset_index(drop=True)

    groups = []

    for _, group in frame.groupby(
        "theme_id",
        sort=False,
    ):
        group = group.copy()

        short_policy_sum = (
            group["policy_count"]
            .rolling(
                window=short_window,
                min_periods=1,
            )
            .sum()
        )

        long_policy_mean = (
            group["policy_count"]
            .rolling(
                window=long_window,
                min_periods=1,
            )
            .mean()
        )

        group["policy_intensity_30"] = (
            short_policy_sum
        )

        group["policy_breadth_30"] = (
            group[
                [
                    "issuing_authority_count",
                    "policy_type_count",
                ]
            ]
            .sum(axis=1)
            .rolling(
                window=short_window,
                min_periods=1,
            )
            .sum()
        )

        expected_short_count = (
            long_policy_mean * short_window
        )

        group["policy_acceleration"] = (
            short_policy_sum
            / expected_short_count.replace(
                0,
                np.nan,
            )
            - 1.0
        )

        group["policy_acceleration"] = (
            group["policy_acceleration"]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

        groups.append(group)

    return pd.concat(
        groups,
        ignore_index=True,
    )


def calculate_policy_narrative_score(
    feature_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily cross-sectional policy Narrative Score.

    Score:
        50% policy intensity
        25% policy breadth
        25% policy acceleration
    """
    required = {
        "date",
        "theme_id",
        "policy_intensity_30",
        "policy_breadth_30",
        "policy_acceleration",
    }

    missing = required - set(feature_data.columns)

    if missing:
        raise ValueError(
            "Policy feature data is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = feature_data.copy()

    score_columns = [
        "policy_intensity_30",
        "policy_breadth_30",
        "policy_acceleration",
    ]

    for column in score_columns:
        frame[f"z_{column}"] = (
            frame.groupby("date")[column]
            .transform(cross_sectional_zscore)
        )

    frame["policy_narrative_score"] = (
        0.50
        * frame["z_policy_intensity_30"]
        + 0.25
        * frame["z_policy_breadth_30"]
        + 0.25
        * frame["z_policy_acceleration"]
    )

    frame["policy_narrative_rank"] = (
        frame.groupby("date")[
            "policy_narrative_score"
        ]
        .rank(
            ascending=False,
            method="first",
        )
    )

    return frame

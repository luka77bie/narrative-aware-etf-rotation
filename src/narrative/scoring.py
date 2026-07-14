from typing import Dict, Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "theme_id",
    "news_count",
    "policy_count",
    "attention_index",
}


def cross_sectional_zscore(values: pd.Series) -> pd.Series:
    """Calculate a cross-sectional z-score for one date."""
    standard_deviation = values.std(ddof=0)

    if (
        pd.isna(standard_deviation)
        or np.isclose(standard_deviation, 0.0)
    ):
        return pd.Series(0.0, index=values.index)

    return (values - values.mean()) / standard_deviation


def engineer_narrative_features(
    data: pd.DataFrame,
    lookback: int = 7,
) -> pd.DataFrame:
    """Transform raw narrative observations into model features."""
    missing = REQUIRED_COLUMNS - set(data.columns)

    if missing:
        raise ValueError(
            "Narrative data is missing columns: "
            + ", ".join(sorted(missing))
        )

    if lookback <= 1:
        raise ValueError("lookback must be greater than 1.")

    frame = data.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    if frame["date"].isna().any():
        raise ValueError(
            "Narrative data contains invalid dates."
        )

    numeric_columns = [
        "news_count",
        "policy_count",
        "attention_index",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if frame[numeric_columns].isna().any().any():
        raise ValueError(
            "Narrative data contains invalid numeric values."
        )

    if (frame[numeric_columns] < 0).any().any():
        raise ValueError(
            "Narrative features cannot be negative."
        )

    frame = (
        frame.sort_values(["theme_id", "date"])
        .drop_duplicates(
            subset=["date", "theme_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    groups = []

    for _, group in frame.groupby(
        "theme_id",
        sort=False,
    ):
        group = group.copy()

        previous_news_mean = (
            group["news_count"]
            .rolling(
                window=lookback,
                min_periods=3,
            )
            .mean()
            .shift(1)
        )

        group["news_growth"] = (
            np.log1p(group["news_count"])
            - np.log1p(previous_news_mean)
        )

        group["policy_intensity"] = (
            group["policy_count"]
            .rolling(
                window=lookback,
                min_periods=1,
            )
            .sum()
        )

        group["attention_change"] = (
            group["attention_index"]
            .pct_change(
                periods=lookback,
                fill_method=None,
            )
        )

        groups.append(group)

    return pd.concat(
        groups,
        ignore_index=True,
    )


def calculate_narrative_scores(
    feature_data: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Calculate cross-sectional Narrative Score and ranking."""
    if weights is None:
        weights = {
            "news_growth": 0.40,
            "policy_intensity": 0.30,
            "attention_change": 0.30,
        }

    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(
            "Narrative weights must sum to 1."
        )

    missing = set(weights) - set(feature_data.columns)

    if missing:
        raise ValueError(
            "Narrative feature data is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = feature_data.copy()

    for feature in weights:
        zscore_column = f"z_{feature}"

        frame[zscore_column] = (
            frame.groupby("date")[feature]
            .transform(cross_sectional_zscore)
        )

    frame["narrative_score"] = 0.0

    for feature, weight in weights.items():
        frame["narrative_score"] += (
            weight * frame[f"z_{feature}"]
        )

    frame["narrative_rank"] = (
        frame.groupby("date")["narrative_score"]
        .rank(
            ascending=False,
            method="first",
        )
    )

    return frame

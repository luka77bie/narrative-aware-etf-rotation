from typing import Optional

import numpy as np
import pandas as pd


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


def map_policy_scores_to_etfs(
    policy_scores: pd.DataFrame,
    theme_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map theme-level policy signals to ETF-level signals.

    If one ETF is mapped to multiple themes, use the mean score.
    """
    required_policy = {
        "date",
        "theme_id",
        "policy_narrative_score",
    }

    required_mapping = {
        "theme_id",
        "ticker",
    }

    missing_policy = (
        required_policy
        - set(policy_scores.columns)
    )

    missing_mapping = (
        required_mapping
        - set(theme_mapping.columns)
    )

    if missing_policy:
        raise ValueError(
            "Policy scores are missing columns: "
            + ", ".join(sorted(missing_policy))
        )

    if missing_mapping:
        raise ValueError(
            "Theme mapping is missing columns: "
            + ", ".join(sorted(missing_mapping))
        )

    merged = policy_scores.merge(
        theme_mapping[
            [
                "theme_id",
                "ticker",
            ]
        ],
        on="theme_id",
        how="inner",
        validate="many_to_many",
    )

    if merged.empty:
        raise ValueError(
            "No policy scores could be mapped to ETFs."
        )

    result = (
        merged.groupby(
            [
                "date",
                "ticker",
            ],
            as_index=False,
        )
        .agg(
            policy_narrative_score=(
                "policy_narrative_score",
                "mean",
            ),
            policy_theme_count=(
                "theme_id",
                "nunique",
            ),
        )
    )

    return result


def combine_narrative_components(
    proxy_scores: pd.DataFrame,
    policy_etf_scores: pd.DataFrame,
    proxy_weight: float = 0.70,
    policy_weight: float = 0.30,
    research_status: str = (
        "pipeline_validation_only"
    ),
) -> pd.DataFrame:
    """
    Combine market-attention proxy and policy Narrative Score.

    Missing policy observations receive a neutral zero score.
    This does not create policy documents.
    """
    if not np.isclose(
        proxy_weight + policy_weight,
        1.0,
    ):
        raise ValueError(
            "Narrative component weights must sum to 1."
        )

    required_proxy = {
        "date",
        "ticker",
        "narrative_proxy_score",
    }

    required_policy = {
        "date",
        "ticker",
        "policy_narrative_score",
    }

    missing_proxy = (
        required_proxy
        - set(proxy_scores.columns)
    )

    missing_policy = (
        required_policy
        - set(policy_etf_scores.columns)
    )

    if missing_proxy:
        raise ValueError(
            "Proxy scores are missing columns: "
            + ", ".join(sorted(missing_proxy))
        )

    if missing_policy:
        raise ValueError(
            "Policy ETF scores are missing columns: "
            + ", ".join(sorted(missing_policy))
        )

    proxy = proxy_scores.copy()
    policy = policy_etf_scores.copy()

    proxy["date"] = pd.to_datetime(
        proxy["date"],
        errors="coerce",
    )

    policy["date"] = pd.to_datetime(
        policy["date"],
        errors="coerce",
    )

    combined = proxy.merge(
        policy[
            [
                "date",
                "ticker",
                "policy_narrative_score",
                "policy_theme_count",
            ]
        ],
        on=[
            "date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    combined["policy_narrative_score"] = (
        combined["policy_narrative_score"]
        .fillna(0.0)
    )

    combined["policy_theme_count"] = (
        combined["policy_theme_count"]
        .fillna(0)
        .astype(int)
    )

    combined["z_narrative_proxy"] = (
        combined.groupby("date")[
            "narrative_proxy_score"
        ]
        .transform(cross_sectional_zscore)
    )

    combined["z_policy_narrative"] = (
        combined.groupby("date")[
            "policy_narrative_score"
        ]
        .transform(cross_sectional_zscore)
    )

    combined["narrative_v2_score"] = (
        proxy_weight
        * combined["z_narrative_proxy"]
        + policy_weight
        * combined["z_policy_narrative"]
    )

    combined["narrative_v2_rank"] = (
        combined.groupby("date")[
            "narrative_v2_score"
        ]
        .rank(
            ascending=False,
            method="first",
        )
    )

    combined["research_status"] = (
        research_status
    )

    return combined

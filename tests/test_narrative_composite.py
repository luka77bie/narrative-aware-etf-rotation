import pandas as pd
import pytest

from src.narrative.composite import (
    combine_narrative_components,
    map_policy_scores_to_etfs,
)


def make_policy_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-07-10",
                    "2026-07-10",
                ]
            ),
            "theme_id": [
                "ai",
                "power",
            ],
            "policy_narrative_score": [
                1.5,
                -0.5,
            ],
        }
    )


def make_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "theme_id": [
                "ai",
                "power",
            ],
            "ticker": [
                "159363",
                "159611",
            ],
        }
    )


def make_proxy_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-07-10",
                    "2026-07-10",
                    "2026-07-10",
                ]
            ),
            "ticker": [
                "159363",
                "159611",
                "518880",
            ],
            "narrative_proxy_score": [
                1.0,
                0.5,
                -0.5,
            ],
        }
    )


def test_policy_scores_map_to_etfs() -> None:
    result = map_policy_scores_to_etfs(
        policy_scores=make_policy_scores(),
        theme_mapping=make_mapping(),
    )

    assert len(result) == 2

    ai = result.loc[
        result["ticker"] == "159363"
    ].iloc[0]

    assert (
        ai["policy_narrative_score"]
        == 1.5
    )


def test_composite_score_created() -> None:
    policy_etf = map_policy_scores_to_etfs(
        make_policy_scores(),
        make_mapping(),
    )

    result = combine_narrative_components(
        proxy_scores=make_proxy_scores(),
        policy_etf_scores=policy_etf,
    )

    assert "narrative_v2_score" in result.columns
    assert "narrative_v2_rank" in result.columns
    assert (
        result["research_status"]
        == "pipeline_validation_only"
    ).all()


def test_missing_policy_is_neutral() -> None:
    policy_etf = map_policy_scores_to_etfs(
        make_policy_scores(),
        make_mapping(),
    )

    result = combine_narrative_components(
        proxy_scores=make_proxy_scores(),
        policy_etf_scores=policy_etf,
    )

    gold = result.loc[
        result["ticker"] == "518880"
    ].iloc[0]

    assert (
        gold["policy_narrative_score"]
        == 0.0
    )

    assert gold["policy_theme_count"] == 0


def test_invalid_weights_fail() -> None:
    policy_etf = map_policy_scores_to_etfs(
        make_policy_scores(),
        make_mapping(),
    )

    with pytest.raises(
        ValueError,
        match="sum to 1",
    ):
        combine_narrative_components(
            proxy_scores=make_proxy_scores(),
            policy_etf_scores=policy_etf,
            proxy_weight=0.8,
            policy_weight=0.4,
        )

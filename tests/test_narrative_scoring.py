import numpy as np
import pandas as pd
import pytest

from src.narrative.scoring import (
    calculate_narrative_scores,
    cross_sectional_zscore,
    engineer_narrative_features,
)


def make_narrative_data() -> pd.DataFrame:
    dates = pd.date_range(
        "2024-01-01",
        periods=15,
        freq="D",
    )

    rows = []

    for theme_index, theme_id in enumerate(
        ["ai", "gold", "bank"]
    ):
        for date_index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "theme_id": theme_id,
                    "news_count": (
                        10
                        + theme_index * 5
                        + date_index
                    ),
                    "policy_count": (
                        theme_index
                        + date_index % 3
                    ),
                    "attention_index": (
                        100
                        + theme_index * 10
                        + date_index
                        * (theme_index + 1)
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_cross_sectional_zscore_mean_is_zero() -> None:
    values = pd.Series([1.0, 2.0, 3.0])

    result = cross_sectional_zscore(values)

    assert np.isclose(result.mean(), 0.0)


def test_engineer_features_creates_columns() -> None:
    result = engineer_narrative_features(
        make_narrative_data(),
        lookback=7,
    )

    assert "news_growth" in result.columns
    assert "policy_intensity" in result.columns
    assert "attention_change" in result.columns


def test_narrative_scores_created() -> None:
    features = engineer_narrative_features(
        make_narrative_data(),
        lookback=7,
    )

    result = calculate_narrative_scores(features)

    assert "narrative_score" in result.columns
    assert "narrative_rank" in result.columns


def test_narrative_weights_must_sum_to_one() -> None:
    features = engineer_narrative_features(
        make_narrative_data(),
        lookback=7,
    )

    with pytest.raises(
        ValueError,
        match="sum to 1",
    ):
        calculate_narrative_scores(
            features,
            weights={
                "news_growth": 0.5,
                "policy_intensity": 0.5,
                "attention_change": 0.5,
            },
        )


def test_duplicate_theme_dates_removed() -> None:
    data = make_narrative_data()

    duplicated = pd.concat(
        [data, data.iloc[[0]]],
        ignore_index=True,
    )

    result = engineer_narrative_features(
        duplicated,
        lookback=7,
    )

    assert not result.duplicated(
        subset=["date", "theme_id"]
    ).any()


def test_latest_date_has_cross_sectional_ranks() -> None:
    features = engineer_narrative_features(
        make_narrative_data(),
        lookback=7,
    )

    scored = calculate_narrative_scores(
        features
    )

    complete = scored.dropna(
        subset=[
            "news_growth",
            "attention_change",
            "narrative_score",
        ]
    )

    latest_date = complete["date"].max()

    latest = complete.loc[
        complete["date"] == latest_date
    ]

    assert latest["narrative_rank"].notna().all()
    assert latest["theme_id"].nunique() == 3

import pandas as pd

from src.narrative.policy_signal import (
    build_policy_theme_panel,
    calculate_policy_narrative_score,
    engineer_policy_signal_features,
)


def make_daily_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-06-10",
                    "2026-07-08",
                    "2026-07-10",
                ]
            ),
            "theme_id": [
                "power",
                "ai",
                "healthcare",
            ],
            "policy_count": [
                1,
                1,
                1,
            ],
            "issuing_authority_count": [
                1,
                4,
                1,
            ],
            "policy_type_count": [
                1,
                1,
                1,
            ],
        }
    )


def test_policy_panel_fills_missing_rows() -> None:
    panel = build_policy_theme_panel(
        daily_policy_features=make_daily_features(),
        theme_ids=[
            "ai",
            "power",
            "healthcare",
        ],
        start_date="2026-06-10",
        end_date="2026-07-10",
    )

    expected_rows = 31 * 3

    assert len(panel) == expected_rows

    missing_row = panel.loc[
        (panel["date"] == pd.Timestamp("2026-06-11"))
        & (panel["theme_id"] == "ai")
    ].iloc[0]

    assert missing_row["policy_count"] == 0


def test_policy_features_created() -> None:
    panel = build_policy_theme_panel(
        daily_policy_features=make_daily_features(),
        theme_ids=[
            "ai",
            "power",
            "healthcare",
        ],
        start_date="2026-06-10",
        end_date="2026-07-10",
    )

    result = engineer_policy_signal_features(
        panel,
        short_window=7,
        long_window=21,
    )

    assert "policy_intensity_30" in result.columns
    assert "policy_breadth_30" in result.columns
    assert "policy_acceleration" in result.columns


def test_recent_policy_increases_intensity() -> None:
    panel = build_policy_theme_panel(
        daily_policy_features=make_daily_features(),
        theme_ids=[
            "ai",
            "power",
            "healthcare",
        ],
        start_date="2026-06-10",
        end_date="2026-07-10",
    )

    result = engineer_policy_signal_features(
        panel,
        short_window=7,
        long_window=21,
    )

    latest_healthcare = result.loc[
        (result["date"] == pd.Timestamp("2026-07-10"))
        & (result["theme_id"] == "healthcare")
    ].iloc[0]

    assert latest_healthcare[
        "policy_intensity_30"
    ] >= 1


def test_policy_score_and_rank_created() -> None:
    panel = build_policy_theme_panel(
        daily_policy_features=make_daily_features(),
        theme_ids=[
            "ai",
            "power",
            "healthcare",
        ],
        start_date="2026-06-10",
        end_date="2026-07-10",
    )

    features = engineer_policy_signal_features(
        panel,
        short_window=7,
        long_window=21,
    )

    result = calculate_policy_narrative_score(
        features
    )

    assert "policy_narrative_score" in result.columns
    assert "policy_narrative_rank" in result.columns

import pandas as pd

from src.narrative.time_alignment import (
    audit_duplicate_signal_rows,
    audit_policy_point_in_time,
    audit_signal_execution_lag,
    build_alignment_summary,
)


def test_duplicate_signal_rows_detected() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                ]
            ),
            "ticker": [
                "510300",
                "510300",
            ],
        }
    )

    result = audit_duplicate_signal_rows(
        data,
        key_columns=["date", "ticker"],
    )

    assert result["has_duplicates"]
    assert result["duplicate_rows"] == 2


def test_policy_point_in_time_valid() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-07-10"]
            ),
            "published_at": [
                "2026-07-10T01:00:00Z"
            ],
            "retrieved_at": [
                "2026-07-10T02:00:00Z"
            ],
        }
    )

    result = audit_policy_point_in_time(data)

    assert result.iloc[0][
        "point_in_time_valid"
    ]


def test_policy_retrieved_late_fails() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-07-10"]
            ),
            "published_at": [
                "2026-07-10T01:00:00Z"
            ],
            "retrieved_at": [
                "2026-07-11T02:00:00Z"
            ],
        }
    )

    result = audit_policy_point_in_time(data)

    assert not result.iloc[0][
        "point_in_time_valid"
    ]


def test_execution_must_follow_signal() -> None:
    result = audit_signal_execution_lag(
        signal_dates=pd.Series(
            pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                ]
            )
        ),
        execution_dates=pd.Series(
            pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-02",
                ]
            )
        ),
    )

    assert result.iloc[0][
        "execution_after_signal"
    ]

    assert not result.iloc[1][
        "execution_after_signal"
    ]


def test_alignment_summary_created() -> None:
    momentum = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01"]
            ),
            "ticker": ["510300"],
        }
    )

    proxy = momentum.copy()

    policy = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01"]
            ),
            "theme_id": ["ai"],
        }
    )

    narrative_v2 = momentum.copy()

    result = build_alignment_summary(
        momentum=momentum,
        proxy=proxy,
        policy=policy,
        narrative_v2=narrative_v2,
    )

    assert len(result) == 4
    assert not result["has_duplicates"].any()


def test_policy_available_after_close_requires_later_signal() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-07-15"]
            ),
            "published_at": [
                "2026-07-10T01:00:00Z"
            ],
            "retrieved_at": [
                "2026-07-14T10:00:00Z"
            ],
        }
    )

    result = audit_policy_point_in_time(
        data,
        market_close_hour=15,
    )

    assert result.iloc[0][
        "available_in_time"
    ]

    assert result.iloc[0][
        "point_in_time_valid"
    ]

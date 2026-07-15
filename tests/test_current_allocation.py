import pandas as pd

from src.reporting.allocation import (
    build_allocation_snapshot,
    compare_allocations,
    render_markdown,
    select_top_n,
)


def make_ranking() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2026-07-13",
                "2026-07-13",
                "2026-07-13",
                "2026-07-13",
            ],
            "momentum_rank": [
                1,
                2,
                3,
                4,
            ],
            "ticker": [
                "510300",
                "512480",
                "588000",
                "159915",
            ],
            "name": [
                "CSI 300 ETF",
                "Semiconductor ETF",
                "STAR 50 ETF",
                "ChiNext ETF",
            ],
            "secondary_theme": [
                "Broad Market",
                "Semiconductor",
                "Technology",
                "Growth",
            ],
            "mom_60_pct": [
                20.0,
                18.0,
                15.0,
                12.0,
            ],
            "momentum_score": [
                1.5,
                1.2,
                0.9,
                0.7,
            ],
        }
    )


def make_history() -> pd.DataFrame:
    rows = []

    for ticker, name, score in [
        (
            "510300",
            "CSI 300 ETF",
            1.3,
        ),
        (
            "512480",
            "Semiconductor ETF",
            1.1,
        ),
        (
            "159915",
            "ChiNext ETF",
            0.8,
        ),
        (
            "588000",
            "STAR 50 ETF",
            0.6,
        ),
    ]:
        rows.append(
            {
                "date": "2026-06-30",
                "ticker": ticker,
                "name": name,
                "mom_60": score / 10,
                "momentum_score": score,
            }
        )

    return pd.DataFrame(rows)


def test_select_top_three_equal_weight() -> None:
    selected = select_top_n(
        make_ranking(),
        top_n=3,
    )

    assert selected["ticker"].tolist() == [
        "510300",
        "512480",
        "588000",
    ]

    assert selected[
        "target_weight"
    ].sum() == 1.0


def test_compare_allocations() -> None:
    current = pd.DataFrame(
        {
            "ticker": [
                "A",
                "B",
                "C",
            ],
            "target_weight": [
                1 / 3,
                1 / 3,
                1 / 3,
            ],
        }
    )

    previous = pd.DataFrame(
        {
            "ticker": [
                "A",
                "B",
                "D",
            ],
            "target_weight": [
                1 / 3,
                1 / 3,
                1 / 3,
            ],
        }
    )

    result = compare_allocations(
        current,
        previous,
    )

    assert result["added"] == ["C"]
    assert result["removed"] == ["D"]
    assert result["retained"] == [
        "A",
        "B",
    ]

    assert round(
        result["estimated_turnover"],
        6,
    ) == round(1 / 3, 6)


def test_snapshot_contains_previous_changes() -> None:
    snapshot = build_allocation_snapshot(
        ranking=make_ranking(),
        history=make_history(),
        top_n=3,
        reference_date=pd.Timestamp(
            "2026-07-14"
        ),
    )

    assert snapshot["data_age_days"] == 1
    assert snapshot["is_stale"] is False
    assert snapshot["added"] == [
        "588000"
    ]
    assert snapshot["removed"] == [
        "159915"
    ]


def test_markdown_contains_selected_etfs() -> None:
    snapshot = build_allocation_snapshot(
        ranking=make_ranking(),
        history=make_history(),
        top_n=3,
        reference_date=pd.Timestamp(
            "2026-07-14"
        ),
    )

    result = render_markdown(
        snapshot
    )

    assert "# Current MOM60 Allocation" in result
    assert "510300" in result
    assert "33.33%" in result
    assert "Estimated one-way turnover" in result

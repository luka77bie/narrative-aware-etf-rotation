import pandas as pd
import pytest

from src.narrative.proxy_composite import (
    combine_momentum_and_proxy,
)


def make_momentum() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-31",
                    "2024-01-31",
                    "2024-01-31",
                ]
            ),
            "ticker": [
                "A",
                "B",
                "C",
            ],
            "mom_60": [
                0.20,
                0.10,
                -0.05,
            ],
            "z_mom_60": [
                1.2,
                0.2,
                -1.4,
            ],
        }
    )


def make_proxy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-31",
                    "2024-01-31",
                    "2024-01-31",
                ]
            ),
            "ticker": [
                "A",
                "B",
                "C",
            ],
            "narrative_proxy_score": [
                -0.5,
                2.0,
                0.1,
            ],
        }
    )


def test_proxy_composite_created() -> None:
    result = combine_momentum_and_proxy(
        momentum=make_momentum(),
        proxy=make_proxy(),
        proxy_weight=0.30,
    )

    assert "momentum_score" in result.columns
    assert "momentum_rank" in result.columns
    assert "z_narrative_proxy" in result.columns


def test_zero_proxy_weight_matches_momentum() -> None:
    result = combine_momentum_and_proxy(
        momentum=make_momentum(),
        proxy=make_proxy(),
        proxy_weight=0.0,
    )

    assert (
        result["momentum_score"]
        == result["z_mom_60"]
    ).all()


def test_proxy_changes_ranking() -> None:
    result = combine_momentum_and_proxy(
        momentum=make_momentum(),
        proxy=make_proxy(),
        proxy_weight=0.50,
    )

    best = result.sort_values(
        "momentum_rank"
    ).iloc[0]

    assert best["ticker"] == "B"


def test_invalid_proxy_weight_fails() -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        combine_momentum_and_proxy(
            momentum=make_momentum(),
            proxy=make_proxy(),
            proxy_weight=1.5,
        )

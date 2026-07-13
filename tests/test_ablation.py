import pandas as pd
import pytest

from src.evaluation.ablation import (
    build_ablation_signal,
    summarise_ablation_results,
)


def make_signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-31",
                    "2024-01-31",
                    "2024-01-31",
                ]
            ),
            "ticker": ["A", "B", "C"],
            "mom_20": [0.30, 0.20, 0.10],
            "mom_60": [0.10, 0.20, 0.30],
            "z_mom_20": [1.0, 0.0, -1.0],
            "z_mom_60": [-1.0, 0.0, 1.0],
        }
    )


def test_mom20_variant_uses_short_momentum() -> None:
    result = build_ablation_signal(
        make_signals(),
        variant="mom20_only",
    )

    best = result.sort_values(
        "momentum_rank"
    ).iloc[0]

    assert best["ticker"] == "A"


def test_mom60_variant_uses_long_momentum() -> None:
    result = build_ablation_signal(
        make_signals(),
        variant="mom60_only",
    )

    best = result.sort_values(
        "momentum_rank"
    ).iloc[0]

    assert best["ticker"] == "C"


def test_combined_variant_averages_zscores() -> None:
    result = build_ablation_signal(
        make_signals(),
        variant="mom20_mom60",
    )

    assert result["momentum_score"].eq(0.0).all()


def test_invalid_variant_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported",
    ):
        build_ablation_signal(
            make_signals(),
            variant="invalid",
        )


def test_summarise_ablation_results() -> None:
    metrics = {
        "MOM20 only": pd.DataFrame(
            [
                {
                    "total_return": 0.20,
                    "cagr": 0.10,
                    "sharpe": 0.60,
                }
            ]
        ),
        "MOM60 only": pd.DataFrame(
            [
                {
                    "total_return": 0.15,
                    "cagr": 0.08,
                    "sharpe": 0.50,
                }
            ]
        ),
    }

    summary = summarise_ablation_results(metrics)

    assert len(summary) == 2
    assert "model" in summary.columns

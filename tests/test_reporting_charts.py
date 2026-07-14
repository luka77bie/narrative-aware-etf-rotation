from pathlib import Path

import pandas as pd

from src.reporting.charts import (
    prepare_oos_returns,
    save_drawdown_chart,
    save_fold_sharpe_chart,
    save_nav_chart,
    save_subperiod_cagr_chart,
)


def make_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
            "model": [
                "MOM60",
                "MOM60",
                "Proxy",
                "Proxy",
            ],
            "net_return": [
                0.01,
                -0.005,
                0.008,
                -0.002,
            ],
        }
    )


def test_prepare_oos_returns() -> None:
    result = prepare_oos_returns(
        make_returns()
    )

    assert "nav" in result.columns
    assert "drawdown" in result.columns
    assert result["nav"].gt(0).all()


def test_nav_chart_created(
    tmp_path: Path,
) -> None:
    output = tmp_path / "nav.png"

    save_nav_chart(
        make_returns(),
        output,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_drawdown_chart_created(
    tmp_path: Path,
) -> None:
    output = tmp_path / "drawdown.png"

    save_drawdown_chart(
        make_returns(),
        output,
    )

    assert output.exists()


def test_fold_sharpe_chart_created(
    tmp_path: Path,
) -> None:
    data = pd.DataFrame(
        {
            "model": [
                "MOM60",
                "Proxy",
            ],
            "fold": [1, 1],
            "sharpe": [
                0.5,
                0.6,
            ],
        }
    )

    output = tmp_path / "folds.png"

    save_fold_sharpe_chart(
        data,
        output,
    )

    assert output.exists()


def test_subperiod_chart_created(
    tmp_path: Path,
) -> None:
    data = pd.DataFrame(
        {
            "model": [
                "MOM60",
                "Proxy",
            ],
            "period": [
                "Full Sample",
                "Full Sample",
            ],
            "transaction_cost_bps": [
                10,
                10,
            ],
            "cagr": [
                0.15,
                0.16,
            ],
        }
    )

    output = tmp_path / "subperiod.png"

    save_subperiod_cagr_chart(
        data,
        output,
    )

    assert output.exists()

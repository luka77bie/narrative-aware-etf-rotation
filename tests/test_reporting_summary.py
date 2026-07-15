from pathlib import Path

import pandas as pd
import pytest

from src.reporting.summary import (
    compare_models,
    performance_markdown_table,
    prepare_performance_table,
    select_model_row,
    write_markdown_report,
)


def make_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": [
                "MOM60",
                "MOM60 + 50% Proxy",
            ],
            "cagr": [
                0.14933,
                0.15830,
            ],
            "annual_volatility": [
                0.27498,
                0.27007,
            ],
            "sharpe": [
                0.644,
                0.680,
            ],
            "sortino": [
                0.861,
                0.913,
            ],
            "maximum_drawdown": [
                -0.45698,
                -0.42263,
            ],
            "calmar": [
                0.327,
                0.375,
            ],
            "average_turnover": [
                0.45732,
                0.49797,
            ],
        }
    )


def test_prepare_performance_table() -> None:
    result = prepare_performance_table(
        make_metrics()
    )

    assert len(result) == 2
    assert "sharpe" in result.columns


def test_compare_models() -> None:
    metrics = make_metrics()

    baseline = select_model_row(
        metrics,
        "MOM60",
    )

    candidate = select_model_row(
        metrics,
        "MOM60 + 50% Proxy",
    )

    result = compare_models(
        baseline,
        candidate,
    )

    assert result["cagr_difference"] > 0
    assert result["sharpe_difference"] > 0


def test_missing_model_fails() -> None:
    with pytest.raises(
        ValueError,
        match="Model not found",
    ):
        select_model_row(
            make_metrics(),
            "Unknown",
        )


def test_markdown_table_created() -> None:
    result = performance_markdown_table(
        make_metrics()
    )

    assert "| Model |" in result
    assert "MOM60" in result
    assert "14.93%" in result


def test_markdown_report_written(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.md"

    write_markdown_report(
        output,
        "# Test Report",
    )

    assert output.exists()
    assert "# Test Report" in output.read_text(
        encoding="utf-8"
    )

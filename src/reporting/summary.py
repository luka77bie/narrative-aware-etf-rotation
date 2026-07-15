from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


PERFORMANCE_COLUMNS = [
    "model",
    "cagr",
    "annual_volatility",
    "sharpe",
    "sortino",
    "maximum_drawdown",
    "calmar",
    "average_turnover",
]


def require_columns(
    data: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Validate that a reporting input contains required columns."""
    missing = set(required_columns) - set(data.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            + ", ".join(sorted(missing))
        )


def load_csv(
    path: Path,
    required_columns: Iterable[str],
    dataset_name: str,
) -> pd.DataFrame:
    """Load and validate one reporting input."""
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} not found: {path}"
        )

    data = pd.read_csv(path)

    require_columns(
        data=data,
        required_columns=required_columns,
        dataset_name=dataset_name,
    )

    return data


def prepare_performance_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardise performance metrics for reporting.

    Internal values remain decimal. Percentage formatting is applied
    only when rendering the report.
    """
    require_columns(
        data=data,
        required_columns=PERFORMANCE_COLUMNS,
        dataset_name="performance data",
    )

    frame = data[PERFORMANCE_COLUMNS].copy()

    numeric_columns = [
        column
        for column in PERFORMANCE_COLUMNS
        if column != "model"
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if frame[numeric_columns].isna().any().any():
        raise ValueError(
            "Performance data contains invalid numeric values."
        )

    return frame


def select_model_row(
    data: pd.DataFrame,
    model_name: str,
) -> pd.Series:
    """Select exactly one model row."""
    matched = data.loc[
        data["model"] == model_name
    ]

    if matched.empty:
        raise ValueError(
            f"Model not found: {model_name}"
        )

    if len(matched) > 1:
        raise ValueError(
            f"Multiple rows found for model: {model_name}"
        )

    return matched.iloc[0]


def compare_models(
    baseline: pd.Series,
    candidate: pd.Series,
) -> Dict[str, float]:
    """Calculate candidate-minus-baseline metric differences."""
    return {
        "cagr_difference": (
            candidate["cagr"]
            - baseline["cagr"]
        ),
        "sharpe_difference": (
            candidate["sharpe"]
            - baseline["sharpe"]
        ),
        "sortino_difference": (
            candidate["sortino"]
            - baseline["sortino"]
        ),
        "drawdown_difference": (
            candidate["maximum_drawdown"]
            - baseline["maximum_drawdown"]
        ),
        "calmar_difference": (
            candidate["calmar"]
            - baseline["calmar"]
        ),
        "turnover_difference": (
            candidate["average_turnover"]
            - baseline["average_turnover"]
        ),
    }


def format_percentage(value: float) -> str:
    """Format decimal values as percentages."""
    return f"{value * 100:.2f}%"


def format_decimal(value: float) -> str:
    """Format ratios consistently."""
    return f"{value:.3f}"


def performance_markdown_table(
    data: pd.DataFrame,
) -> str:
    """Render performance metrics as a Markdown table."""
    frame = prepare_performance_table(data)

    lines = [
        "| Model | CAGR | Volatility | Sharpe | "
        "Sortino | Max Drawdown | Calmar | Turnover |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in frame.itertuples(index=False):
        lines.append(
            "| "
            f"{row.model} | "
            f"{format_percentage(row.cagr)} | "
            f"{format_percentage(row.annual_volatility)} | "
            f"{format_decimal(row.sharpe)} | "
            f"{format_decimal(row.sortino)} | "
            f"{format_percentage(row.maximum_drawdown)} | "
            f"{format_decimal(row.calmar)} | "
            f"{format_percentage(row.average_turnover)} |"
        )

    return "\n".join(lines)


def write_markdown_report(
    output_path: Path,
    content: str,
) -> None:
    """Write a UTF-8 Markdown research report."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        content.rstrip() + "\n",
        encoding="utf-8",
    )

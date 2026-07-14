import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.reporting.summary import (
    compare_models,
    format_decimal,
    format_percentage,
    load_csv,
    performance_markdown_table,
    prepare_performance_table,
    select_model_row,
    write_markdown_report,
)


OUTPUT_DIRECTORY = Path(
    "outputs/reporting"
)

ABLATION_PATH = Path(
    "outputs/proxy_composite_ablation_metrics.csv"
)

ROBUSTNESS_PATH = Path(
    "outputs/proxy_robustness_metrics.csv"
)

WALK_FORWARD_PATH = Path(
    "outputs/walk_forward_aggregate_metrics.csv"
)

FOLD_PATH = Path(
    "outputs/walk_forward_fold_metrics.csv"
)


BASELINE_MODEL = "MOM60"
CANDIDATE_MODEL = "MOM60 + 50% Proxy"


def build_ablation_summary() -> pd.DataFrame:
    ablation = load_csv(
        path=ABLATION_PATH,
        required_columns=[
            "model",
            "cagr",
            "annual_volatility",
            "sharpe",
            "sortino",
            "maximum_drawdown",
            "calmar",
            "average_turnover",
        ],
        dataset_name="Proxy ablation metrics",
    )

    return prepare_performance_table(
        ablation
    )


def build_robustness_summary() -> pd.DataFrame:
    robustness = load_csv(
        path=ROBUSTNESS_PATH,
        required_columns=[
            "model",
            "transaction_cost_bps",
            "period",
            "cagr",
            "sharpe",
            "maximum_drawdown",
            "calmar",
        ],
        dataset_name="Proxy robustness metrics",
    )

    selected = robustness.loc[
        robustness["transaction_cost_bps"] == 10
    ].copy()

    return selected


def build_walk_forward_summary() -> pd.DataFrame:
    return load_csv(
        path=WALK_FORWARD_PATH,
        required_columns=[
            "model",
            "fold_count",
            "cagr",
            "annual_volatility",
            "sharpe",
            "sortino",
            "maximum_drawdown",
            "calmar",
            "mean_fold_sharpe",
            "median_fold_sharpe",
            "positive_sharpe_folds",
        ],
        dataset_name="Walk-forward aggregate metrics",
    )


def build_fold_summary() -> pd.DataFrame:
    return load_csv(
        path=FOLD_PATH,
        required_columns=[
            "model",
            "fold",
            "test_start",
            "test_end",
            "cagr",
            "sharpe",
            "maximum_drawdown",
            "calmar",
        ],
        dataset_name="Walk-forward fold metrics",
    )


def render_subperiod_table(
    robustness: pd.DataFrame,
) -> str:
    lines = [
        "| Model | Period | CAGR | Sharpe | "
        "Max Drawdown | Calmar |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for row in robustness.itertuples(
        index=False
    ):
        lines.append(
            "| "
            f"{row.model} | "
            f"{row.period} | "
            f"{format_percentage(row.cagr)} | "
            f"{format_decimal(row.sharpe)} | "
            f"{format_percentage(row.maximum_drawdown)} | "
            f"{format_decimal(row.calmar)} |"
        )

    return "\n".join(lines)


def render_walk_forward_table(
    walk_forward: pd.DataFrame,
) -> str:
    lines = [
        "| Model | Folds | CAGR | Sharpe | Sortino | "
        "Max Drawdown | Calmar | Mean Fold Sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in walk_forward.itertuples(
        index=False
    ):
        lines.append(
            "| "
            f"{row.model} | "
            f"{int(row.fold_count)} | "
            f"{format_percentage(row.cagr)} | "
            f"{format_decimal(row.sharpe)} | "
            f"{format_decimal(row.sortino)} | "
            f"{format_percentage(row.maximum_drawdown)} | "
            f"{format_decimal(row.calmar)} | "
            f"{format_decimal(row.mean_fold_sharpe)} |"
        )

    return "\n".join(lines)


def render_fold_table(
    folds: pd.DataFrame,
) -> str:
    lines = [
        "| Model | Fold | Test Start | Test End | "
        "CAGR | Sharpe | Max Drawdown | Calmar |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]

    for row in folds.itertuples(
        index=False
    ):
        lines.append(
            "| "
            f"{row.model} | "
            f"{int(row.fold)} | "
            f"{row.test_start} | "
            f"{row.test_end} | "
            f"{format_percentage(row.cagr)} | "
            f"{format_decimal(row.sharpe)} | "
            f"{format_percentage(row.maximum_drawdown)} | "
            f"{format_decimal(row.calmar)} |"
        )

    return "\n".join(lines)


def main() -> int:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ablation = build_ablation_summary()
    robustness = build_robustness_summary()
    walk_forward = build_walk_forward_summary()
    folds = build_fold_summary()

    baseline_full = select_model_row(
        ablation,
        "MOM60 + 0% Proxy",
    )

    candidate_full = select_model_row(
        ablation,
        CANDIDATE_MODEL,
    )

    baseline_oos = select_model_row(
        walk_forward,
        BASELINE_MODEL,
    )

    candidate_oos = select_model_row(
        walk_forward,
        CANDIDATE_MODEL,
    )

    full_difference = compare_models(
        baseline_full,
        candidate_full,
    )

    oos_difference = compare_models(
        baseline_oos,
        candidate_oos,
    )

    ablation.to_csv(
        OUTPUT_DIRECTORY
        / "model_summary.csv",
        index=False,
    )

    robustness.to_csv(
        OUTPUT_DIRECTORY
        / "robustness_summary.csv",
        index=False,
    )

    walk_forward.to_csv(
        OUTPUT_DIRECTORY
        / "walk_forward_summary.csv",
        index=False,
    )

    report = f"""# Narrative-Aware ETF Rotation Research Report

## Executive Decision

The selected primary strategy remains **MOM60**.

The **MOM60 + 50% Market Attention Proxy** model improved full-sample
performance but did not outperform MOM60 in aggregate walk-forward
out-of-sample evaluation. It is therefore retained as an exploratory
research variant rather than the selected model.

Policy-derived narrative signals remain validation-only and are
excluded from formal historical performance claims.

## Full-Sample Proxy Ablation

{performance_markdown_table(ablation)}

### Full-Sample Candidate Difference

- CAGR difference: {format_percentage(full_difference["cagr_difference"])}
- Sharpe difference: {format_decimal(full_difference["sharpe_difference"])}
- Drawdown difference: {format_percentage(full_difference["drawdown_difference"])}
- Calmar difference: {format_decimal(full_difference["calmar_difference"])}
- Turnover difference: {format_percentage(full_difference["turnover_difference"])}

## Subperiod Robustness at 10 bps

{render_subperiod_table(robustness)}

## Walk-Forward Aggregate OOS Results

{render_walk_forward_table(walk_forward)}

### Aggregate OOS Candidate Difference

- CAGR difference: {format_percentage(oos_difference["cagr_difference"])}
- Sharpe difference: {format_decimal(oos_difference["sharpe_difference"])}
- Drawdown difference: {format_percentage(oos_difference["drawdown_difference"])}
- Calmar difference: {format_decimal(oos_difference["calmar_difference"])}

## Walk-Forward Fold Results

{render_fold_table(folds)}

## Visual Results

### Walk-Forward Out-of-Sample NAV

![Walk-Forward OOS NAV](charts/nav_comparison.png)

### Walk-Forward Out-of-Sample Drawdown

![Walk-Forward OOS Drawdown](charts/drawdown_comparison.png)

### Sharpe Ratio by Fold

![Walk-Forward Sharpe Ratio by Fold](charts/walk_forward_sharpe.png)

### CAGR by Subperiod

![CAGR by Subperiod](charts/subperiod_cagr.png)

## Interpretation

The market-attention proxy showed useful defensive behaviour during
the 2022–2023 weak market period. However, it underperformed MOM60 in
the pre-2022 and 2024+ subperiods.

Walk-forward evaluation also favoured MOM60 on aggregate CAGR,
Sharpe, Sortino, maximum drawdown and Calmar.

The proxy therefore provides an informative explanatory signal, but
the current specification does not demonstrate sufficiently stable
out-of-sample alpha to replace the MOM60 baseline.

## Final Model Status

| Component | Status |
|---|---|
| MOM60 | Selected primary model |
| 50% Market Attention Proxy | Exploratory OOS-rejected candidate |
| 10–30% Proxy variants | Rejected |
| Policy Narrative | Pipeline validation only |
| Narrative V2 | Pipeline validation only |

## Methodological Controls

- Next-trading-day execution
- Transaction costs applied using turnover
- Minimum ETF coverage requirement
- Point-in-time policy availability
- Market-close timing control
- Non-overlapping walk-forward OOS windows
- Fixed candidate parameters during OOS evaluation
- No post-OOS proxy-weight optimisation
"""

    report_path = (
        OUTPUT_DIRECTORY
        / "research_report.md"
    )

    write_markdown_report(
        report_path,
        report,
    )

    print("Research Reporting V1")
    print("=" * 80)
    print(
        "Primary model:",
        BASELINE_MODEL,
    )
    print(
        "Exploratory candidate:",
        CANDIDATE_MODEL,
    )
    print(f"Report: {report_path}")
    print(
        "Model summary:",
        OUTPUT_DIRECTORY / "model_summary.csv",
    )
    print(
        "Robustness summary:",
        OUTPUT_DIRECTORY
        / "robustness_summary.csv",
    )
    print(
        "Walk-forward summary:",
        OUTPUT_DIRECTORY
        / "walk_forward_summary.csv",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

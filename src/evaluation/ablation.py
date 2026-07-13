from typing import Dict

import pandas as pd


VALID_VARIANTS = {
    "mom20_only",
    "mom60_only",
    "mom20_mom60",
}


def build_ablation_signal(
    scored_signals: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    """
    Build one momentum ablation variant.

    All variants use the common sample where both MOM20 and
    MOM60 are available, ensuring a fair comparison.
    """
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"Unsupported ablation variant: {variant}"
        )

    required_columns = {
        "date",
        "ticker",
        "mom_20",
        "mom_60",
        "z_mom_20",
        "z_mom_60",
    }

    missing = required_columns - set(scored_signals.columns)

    if missing:
        raise ValueError(
            "Signal data is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = scored_signals.dropna(
        subset=[
            "mom_20",
            "mom_60",
            "z_mom_20",
            "z_mom_60",
        ]
    ).copy()

    if variant == "mom20_only":
        frame["momentum_score"] = frame["z_mom_20"]

    elif variant == "mom60_only":
        frame["momentum_score"] = frame["z_mom_60"]

    else:
        frame["momentum_score"] = (
            0.5 * frame["z_mom_20"]
            + 0.5 * frame["z_mom_60"]
        )

    frame["momentum_rank"] = (
        frame.groupby("date")["momentum_score"]
        .rank(
            ascending=False,
            method="first",
        )
    )

    return frame


def summarise_ablation_results(
    results: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine metrics from multiple ablation runs."""
    rows = []

    for model_name, metrics in results.items():
        if metrics.empty:
            continue

        row = metrics.iloc[0].to_dict()
        row["model"] = model_name
        rows.append(row)

    if not rows:
        raise ValueError(
            "No ablation metrics were provided."
        )

    summary = pd.DataFrame(rows)

    preferred_columns = [
        "model",
        "total_return",
        "cagr",
        "annual_volatility",
        "sharpe",
        "sortino",
        "maximum_drawdown",
        "calmar",
        "average_turnover",
        "rebalance_count",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in summary.columns
    ]

    return summary[available_columns]

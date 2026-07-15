from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def require_columns(
    data: pd.DataFrame,
    required: Iterable[str],
    dataset_name: str,
) -> None:
    missing = set(required) - set(data.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            + ", ".join(sorted(missing))
        )


def prepare_oos_returns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare walk-forward OOS returns for charting."""
    require_columns(
        data,
        required=[
            "date",
            "model",
            "net_return",
        ],
        dataset_name="OOS returns",
    )

    frame = data.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame["net_return"] = pd.to_numeric(
        frame["net_return"],
        errors="coerce",
    )

    if frame[
        [
            "date",
            "net_return",
        ]
    ].isna().any().any():
        raise ValueError(
            "OOS returns contain invalid values."
        )

    frame = (
        frame.sort_values(
            [
                "model",
                "date",
            ]
        )
        .drop_duplicates(
            subset=[
                "model",
                "date",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    frame["nav"] = (
        frame.groupby("model")[
            "net_return"
        ]
        .transform(
            lambda values: (
                1.0 + values
            ).cumprod()
        )
    )

    frame["running_peak"] = (
        frame.groupby("model")["nav"]
        .cummax()
    )

    frame["drawdown"] = (
        frame["nav"]
        / frame["running_peak"]
        - 1.0
    )

    return frame


def save_nav_chart(
    returns: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save cumulative OOS NAV comparison."""
    frame = prepare_oos_returns(returns)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    for model, group in frame.groupby(
        "model"
    ):
        axis.plot(
            group["date"],
            group["nav"],
            label=model,
            linewidth=2,
        )

    axis.set_title(
        "Walk-Forward Out-of-Sample NAV"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel(
        "Growth of 1 Unit"
    )
    axis.grid(
        True,
        alpha=0.25,
    )
    axis.legend()
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def save_drawdown_chart(
    returns: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save OOS drawdown comparison."""
    frame = prepare_oos_returns(returns)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    for model, group in frame.groupby(
        "model"
    ):
        axis.plot(
            group["date"],
            group["drawdown"] * 100,
            label=model,
            linewidth=2,
        )

    axis.set_title(
        "Walk-Forward Out-of-Sample Drawdown"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown (%)")
    axis.grid(
        True,
        alpha=0.25,
    )
    axis.legend()
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def save_fold_sharpe_chart(
    fold_metrics: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save fold-level Sharpe comparison."""
    require_columns(
        fold_metrics,
        required=[
            "model",
            "fold",
            "sharpe",
        ],
        dataset_name="Fold metrics",
    )

    pivot = fold_metrics.pivot(
        index="fold",
        columns="model",
        values="sharpe",
    ).sort_index()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    pivot.plot(
        kind="bar",
        ax=axis,
    )

    axis.axhline(
        0.0,
        linewidth=1,
    )
    axis.set_title(
        "Walk-Forward Sharpe Ratio by Fold"
    )
    axis.set_xlabel("Fold")
    axis.set_ylabel("Sharpe Ratio")
    axis.grid(
        True,
        axis="y",
        alpha=0.25,
    )
    axis.legend(
        title="Model"
    )
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def save_subperiod_cagr_chart(
    robustness: pd.DataFrame,
    output_path: Path,
    transaction_cost_bps: float = 10,
) -> Path:
    """Save subperiod CAGR comparison."""
    require_columns(
        robustness,
        required=[
            "model",
            "period",
            "transaction_cost_bps",
            "cagr",
        ],
        dataset_name="Robustness metrics",
    )

    selected = robustness.loc[
        robustness[
            "transaction_cost_bps"
        ] == transaction_cost_bps
    ].copy()

    if selected.empty:
        raise ValueError(
            "No robustness rows found for "
            f"{transaction_cost_bps} bps."
        )

    pivot = selected.pivot(
        index="period",
        columns="model",
        values="cagr",
    )

    preferred_order = [
        "Pre-2022",
        "2022-2023",
        "2024+",
        "Full Sample",
    ]

    available_order = [
        period
        for period in preferred_order
        if period in pivot.index
    ]

    pivot = pivot.loc[
        available_order
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    (
        pivot * 100
    ).plot(
        kind="bar",
        ax=axis,
    )

    axis.axhline(
        0.0,
        linewidth=1,
    )
    axis.set_title(
        "CAGR by Subperiod at 10 bps"
    )
    axis.set_xlabel("Period")
    axis.set_ylabel("CAGR (%)")
    axis.grid(
        True,
        axis="y",
        alpha=0.25,
    )
    axis.legend(
        title="Model"
    )
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path

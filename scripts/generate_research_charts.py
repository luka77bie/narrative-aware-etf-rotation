import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.reporting.charts import (
    save_drawdown_chart,
    save_fold_sharpe_chart,
    save_nav_chart,
    save_subperiod_cagr_chart,
)


REPORTING_DIRECTORY = Path(
    "outputs/reporting"
)

CHART_DIRECTORY = (
    REPORTING_DIRECTORY / "charts"
)

OOS_RETURNS_PATH = Path(
    "outputs/walk_forward_oos_returns.csv"
)

FOLD_METRICS_PATH = Path(
    "outputs/walk_forward_fold_metrics.csv"
)

ROBUSTNESS_PATH = Path(
    "outputs/proxy_robustness_metrics.csv"
)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required reporting input missing: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"Required reporting input is empty: {path}"
        )


def main() -> int:
    print("Research Charts V1")
    print("=" * 80)

    input_paths = [
        OOS_RETURNS_PATH,
        FOLD_METRICS_PATH,
        ROBUSTNESS_PATH,
    ]

    for path in input_paths:
        require_file(path)
        print(f"[INPUT] {path}")

    returns = pd.read_csv(
        OOS_RETURNS_PATH,
        parse_dates=["date"],
    )

    folds = pd.read_csv(
        FOLD_METRICS_PATH,
    )

    robustness = pd.read_csv(
        ROBUSTNESS_PATH,
    )

    print(
        f"[LOADED] OOS returns: {len(returns)} rows"
    )
    print(
        f"[LOADED] Fold metrics: {len(folds)} rows"
    )
    print(
        f"[LOADED] Robustness metrics: "
        f"{len(robustness)} rows"
    )

    CHART_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = []

    print("[CHART] Generating OOS NAV...")
    outputs.append(
        save_nav_chart(
            returns,
            CHART_DIRECTORY
            / "nav_comparison.png",
        )
    )

    print("[CHART] Generating drawdown...")
    outputs.append(
        save_drawdown_chart(
            returns,
            CHART_DIRECTORY
            / "drawdown_comparison.png",
        )
    )

    print("[CHART] Generating fold Sharpe...")
    outputs.append(
        save_fold_sharpe_chart(
            folds,
            CHART_DIRECTORY
            / "walk_forward_sharpe.png",
        )
    )

    print("[CHART] Generating subperiod CAGR...")
    outputs.append(
        save_subperiod_cagr_chart(
            robustness,
            CHART_DIRECTORY
            / "subperiod_cagr.png",
            transaction_cost_bps=10,
        )
    )

    print("")
    print("Created charts:")

    for output in outputs:
        print(
            f"- {output} "
            f"({output.stat().st_size:,} bytes)"
        )

    print("")
    print("Chart generation status: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

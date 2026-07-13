import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.backtest.engine import (
    calculate_performance_metrics,
)
from src.data.market_data import (
    load_cached_price_data,
)
from src.risk.volatility_overlay import (
    apply_volatility_target_overlay,
)


BASELINE_PATH = Path(
    "outputs/momentum_backtest_returns.csv"
)

OUTPUT_DIRECTORY = Path("outputs")

TARGETS = [
    0.10,
    0.15,
    0.20,
]


def load_cash_returns(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    cash = load_cached_price_data(
        ticker="159001",
        cache_directory="data/raw",
    )

    cash = cash.loc[
        (cash["date"] >= start_date)
        & (cash["date"] <= end_date)
    ].copy()

    cash["cash_return"] = (
        cash["adjusted_close"]
        .pct_change(fill_method=None)
        .fillna(0.0)
    )

    return cash[
        [
            "date",
            "cash_return",
        ]
    ]


def main() -> int:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            "Momentum baseline returns are missing. Run:\n"
            "python3 scripts/run_momentum_backtest.py"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = pd.read_csv(
        BASELINE_PATH,
        parse_dates=["date"],
    )

    cash_returns = load_cash_returns(
        start_date=baseline["date"].min(),
        end_date=baseline["date"].max(),
    )

    metrics_rows = []

    for target in TARGETS:
        print(
            f"[RUN] Target volatility: "
            f"{target:.0%}"
        )

        overlay = apply_volatility_target_overlay(
            strategy_returns=baseline[
                [
                    "date",
                    "net_return",
                ]
            ],
            cash_returns=cash_returns,
            target_volatility=target,
            lookback=60,
        )

        metrics = calculate_performance_metrics(
            overlay["overlay_return"]
        )

        metrics_rows.append(
            {
                "model": (
                    f"MOM60 Vol Target "
                    f"{target:.0%}"
                ),
                "target_volatility": target,
                **metrics,
                "average_risky_exposure": (
                    overlay[
                        "risky_exposure"
                    ].mean()
                ),
                "minimum_risky_exposure": (
                    overlay[
                        "risky_exposure"
                    ].min()
                ),
                "cash_days": int(
                    overlay[
                        "cash_exposure"
                    ].gt(0).sum()
                ),
            }
        )

        output_name = (
            f"vol_target_"
            f"{int(target * 100)}pct_returns.csv"
        )

        overlay.to_csv(
            OUTPUT_DIRECTORY / output_name,
            index=False,
        )

    metrics_frame = pd.DataFrame(
        metrics_rows
    )

    metrics_frame.to_csv(
        OUTPUT_DIRECTORY
        / "volatility_target_sensitivity.csv",
        index=False,
    )

    display = metrics_frame.copy()

    for column in [
        "target_volatility",
        "cagr",
        "annual_volatility",
        "maximum_drawdown",
        "average_risky_exposure",
        "minimum_risky_exposure",
    ]:
        display[column] *= 100

    print("")
    print("Volatility Target Sensitivity")
    print("=" * 130)

    print(
        display[
            [
                "model",
                "cagr",
                "annual_volatility",
                "sharpe",
                "sortino",
                "maximum_drawdown",
                "calmar",
                "average_risky_exposure",
                "minimum_risky_exposure",
                "cash_days",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("")
    print(
        "Output: "
        "outputs/volatility_target_sensitivity.csv"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

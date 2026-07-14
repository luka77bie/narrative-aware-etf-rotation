import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.risk.regime import (
    apply_market_regime,
    build_market_regime,
)


INPUT_PATH = Path(
    "outputs/momentum_signal_history.csv"
)

OUTPUT_PATH = Path(
    "outputs/market_regime_signal_history.csv"
)

REGIME_PATH = Path(
    "outputs/market_regime_history.csv"
)


def main() -> int:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Momentum signal history not found. Run:\n"
            "python3 scripts/run_momentum_signal.py"
        )

    signals = pd.read_csv(
        INPUT_PATH,
        dtype={"ticker": "string"},
        parse_dates=["date"],
    )

    # MOM60 is the selected baseline after ablation.
    signals["momentum_score"] = (
        signals["z_mom_60"]
    )

    signals["momentum_rank"] = (
        signals.groupby("date")["momentum_score"]
        .rank(
            ascending=False,
            method="first",
        )
    )

    regime = build_market_regime(
        signal_data=signals,
        benchmark_ticker="510300",
        momentum_column="mom_60",
        threshold=0.0,
    )

    adjusted = apply_market_regime(
        signal_data=signals,
        regime_data=regime,
        absolute_momentum_column="mom_60",
    )

    regime.to_csv(
        REGIME_PATH,
        index=False,
    )

    adjusted.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("Market Regime Summary")
    print("=" * 60)
    print(
        regime["regime"]
        .value_counts()
        .to_string()
    )

    print("")
    print(
        "Risk-off ratio:",
        round(
            regime["risk_on"].eq(False).mean(),
            4,
        ),
    )

    print(f"Regime history: {REGIME_PATH}")
    print(f"Adjusted signals: {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

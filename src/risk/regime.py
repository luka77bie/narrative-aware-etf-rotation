import pandas as pd


def build_market_regime(
    signal_data: pd.DataFrame,
    benchmark_ticker: str = "510300",
    momentum_column: str = "mom_60",
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Build a daily market regime from benchmark absolute momentum.

    risk_on:
        benchmark MOM60 > threshold

    risk_off:
        benchmark MOM60 <= threshold
    """
    required_columns = {
        "date",
        "ticker",
        momentum_column,
    }

    missing = required_columns - set(signal_data.columns)

    if missing:
        raise ValueError(
            "Signal data is missing columns: "
            + ", ".join(sorted(missing))
        )

    benchmark = signal_data.loc[
        signal_data["ticker"].astype(str)
        == str(benchmark_ticker)
    ][
        [
            "date",
            momentum_column,
        ]
    ].copy()

    if benchmark.empty:
        raise ValueError(
            f"Benchmark ticker not found: {benchmark_ticker}"
        )

    benchmark = (
        benchmark.sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .rename(
            columns={
                momentum_column: "benchmark_momentum",
            }
        )
    )

    benchmark["risk_on"] = (
        benchmark["benchmark_momentum"] > threshold
    )

    benchmark["regime"] = benchmark["risk_on"].map(
        {
            True: "RISK_ON",
            False: "RISK_OFF",
        }
    )

    return benchmark.reset_index(drop=True)


def apply_market_regime(
    signal_data: pd.DataFrame,
    regime_data: pd.DataFrame,
    absolute_momentum_column: str = "mom_60",
) -> pd.DataFrame:
    """
    Apply portfolio-level market regime to ETF signals.

    On risk-off dates, all risky ETFs are made ineligible for
    the existing absolute-momentum cash filter.
    """
    required_regime_columns = {
        "date",
        "risk_on",
        "regime",
        "benchmark_momentum",
    }

    missing = (
        required_regime_columns
        - set(regime_data.columns)
    )

    if missing:
        raise ValueError(
            "Regime data is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = signal_data.merge(
        regime_data,
        on="date",
        how="left",
        validate="many_to_one",
    )

    frame["original_absolute_momentum"] = (
        frame[absolute_momentum_column]
    )

    risk_off = frame["risk_on"].eq(False)

    # The cash-filter selector requires positive MOM60.
    # Force every risky ETF below zero during risk-off regimes.
    frame.loc[
        risk_off,
        absolute_momentum_column,
    ] = -1.0

    return frame

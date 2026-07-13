import pandas as pd

from src.risk.regime import (
    apply_market_regime,
    build_market_regime,
)


def make_signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-31",
                    "2024-02-29",
                    "2024-01-31",
                    "2024-02-29",
                ]
            ),
            "ticker": [
                "510300",
                "510300",
                "ETF_A",
                "ETF_A",
            ],
            "mom_60": [
                0.10,
                -0.05,
                0.20,
                0.15,
            ],
            "momentum_score": [
                1.0,
                -1.0,
                2.0,
                1.5,
            ],
        }
    )


def test_build_market_regime() -> None:
    regime = build_market_regime(
        make_signals(),
        benchmark_ticker="510300",
    )

    assert regime["risk_on"].tolist() == [
        True,
        False,
    ]

    assert regime["regime"].tolist() == [
        "RISK_ON",
        "RISK_OFF",
    ]


def test_risk_off_forces_negative_momentum() -> None:
    signals = make_signals()

    regime = build_market_regime(
        signals,
        benchmark_ticker="510300",
    )

    result = apply_market_regime(
        signal_data=signals,
        regime_data=regime,
    )

    risk_off_rows = result.loc[
        result["regime"] == "RISK_OFF"
    ]

    assert (
        risk_off_rows["mom_60"] < 0
    ).all()


def test_risk_on_preserves_original_momentum() -> None:
    signals = make_signals()

    regime = build_market_regime(signals)

    result = apply_market_regime(
        signal_data=signals,
        regime_data=regime,
    )

    risk_on_etf = result.loc[
        (result["regime"] == "RISK_ON")
        & (result["ticker"] == "ETF_A")
    ].iloc[0]

    assert risk_on_etf["mom_60"] == 0.20

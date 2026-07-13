from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs")
RETURNS_PATH = OUTPUT_DIR / "momentum_backtest_returns.csv"
HOLDINGS_PATH = OUTPUT_DIR / "momentum_backtest_holdings.csv"
REBALANCES_PATH = OUTPUT_DIR / "momentum_backtest_rebalances.csv"


def main() -> int:
    returns = pd.read_csv(
        RETURNS_PATH,
        parse_dates=["date"],
    )

    holdings = pd.read_csv(
        HOLDINGS_PATH,
        dtype={"ticker": "string"},
        parse_dates=["date"],
    )

    rebalances = pd.read_csv(
        REBALANCES_PATH,
        parse_dates=[
            "signal_date",
            "execution_date",
        ],
    )

    returns = returns.sort_values("date").copy()

    returns["running_max"] = (
        returns["net_nav"].cummax()
    )

    returns["drawdown"] = (
        returns["net_nav"]
        / returns["running_max"]
        - 1.0
    )

    worst_index = returns["drawdown"].idxmin()
    worst_row = returns.loc[worst_index]

    print("Backtest Integrity Diagnostics")
    print("=" * 80)

    print("\nDate range")
    print(
        returns["date"].min().date(),
        "to",
        returns["date"].max().date(),
    )

    print("\nReturn summary")
    print(
        returns[
            [
                "gross_return",
                "transaction_cost",
                "net_return",
            ]
        ].describe().to_string()
    )

    print("\nLargest negative daily returns")
    print(
        returns.nsmallest(
            15,
            "net_return",
        )[
            [
                "date",
                "gross_return",
                "transaction_cost",
                "net_return",
                "net_nav",
                "drawdown",
            ]
        ].to_string(index=False)
    )

    print("\nLargest positive daily returns")
    print(
        returns.nlargest(
            10,
            "net_return",
        )[
            [
                "date",
                "gross_return",
                "net_return",
                "net_nav",
            ]
        ].to_string(index=False)
    )

    print("\nMaximum drawdown")
    print(
        f"Date: {worst_row['date'].date()}"
    )
    print(
        f"Drawdown: {worst_row['drawdown']:.2%}"
    )
    print(
        f"NAV: {worst_row['net_nav']:.4f}"
    )

    extreme_returns = returns.loc[
        returns["net_return"].abs() > 0.15
    ]

    print("\nExtreme return days above 15% absolute value")
    print(f"Count: {len(extreme_returns)}")

    if not extreme_returns.empty:
        print(
            extreme_returns[
                [
                    "date",
                    "gross_return",
                    "transaction_cost",
                    "net_return",
                ]
            ].to_string(index=False)
        )

    weight_sums = (
        holdings.groupby("date")["weight"]
        .sum()
    )

    print("\nPortfolio weight totals")
    print(weight_sums.describe().to_string())

    invalid_weights = weight_sums.loc[
        (weight_sums < 0.999)
        | (weight_sums > 1.001)
    ]

    print(
        "\nDates with weights not equal to 100%:",
        len(invalid_weights),
    )

    duplicate_holdings = int(
        holdings.duplicated(
            subset=["date", "ticker"]
        ).sum()
    )

    print(
        "Duplicate date/ticker holdings:",
        duplicate_holdings,
    )

    print("\nRebalance summary")
    print(
        rebalances[
            [
                "turnover",
                "transaction_cost",
            ]
        ].describe().to_string()
    )

    diagnostic_path = (
        OUTPUT_DIR
        / "momentum_backtest_diagnostics.csv"
    )

    returns.to_csv(
        diagnostic_path,
        index=False,
    )

    print(
        f"\nDetailed output: {diagnostic_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MOMENTUM_PATH = Path(
    "outputs/momentum_signal_history.csv"
)

PROXY_PATH = Path(
    "outputs/narrative_proxy_signal_history.csv"
)

OUTPUT_PATH = Path(
    "outputs/proxy_composite_coverage_audit.csv"
)


def main() -> int:
    momentum = pd.read_csv(
        MOMENTUM_PATH,
        dtype={"ticker": "string"},
        parse_dates=["date"],
    )

    proxy = pd.read_csv(
        PROXY_PATH,
        dtype={"ticker": "string"},
        parse_dates=["date"],
    )

    momentum["ticker"] = (
        momentum["ticker"]
        .astype(str)
        .str.zfill(6)
    )

    proxy["ticker"] = (
        proxy["ticker"]
        .astype(str)
        .str.zfill(6)
    )

    momentum_valid = momentum.dropna(
        subset=[
            "mom_60",
            "z_mom_60",
        ]
    )[
        [
            "date",
            "ticker",
        ]
    ].copy()

    proxy_valid = proxy.dropna(
        subset=[
            "narrative_proxy_score",
        ]
    )[
        [
            "date",
            "ticker",
        ]
    ].copy()

    overlap = momentum_valid.merge(
        proxy_valid,
        on=[
            "date",
            "ticker",
        ],
        how="inner",
        validate="one_to_one",
    )

    momentum_daily = (
        momentum_valid.groupby("date")["ticker"]
        .nunique()
        .rename("momentum_assets")
    )

    proxy_daily = (
        proxy_valid.groupby("date")["ticker"]
        .nunique()
        .rename("proxy_assets")
    )

    overlap_daily = (
        overlap.groupby("date")["ticker"]
        .nunique()
        .rename("overlap_assets")
    )

    coverage = pd.concat(
        [
            momentum_daily,
            proxy_daily,
            overlap_daily,
        ],
        axis=1,
    ).fillna(0)

    coverage = coverage.reset_index()

    for column in [
        "momentum_assets",
        "proxy_assets",
        "overlap_assets",
    ]:
        coverage[column] = (
            coverage[column].astype(int)
        )

    coverage["eligible_10_assets"] = (
        coverage["overlap_assets"] >= 10
    )

    coverage["year_month"] = (
        coverage["date"].dt.to_period("M")
    )

    monthly = (
        coverage.groupby("year_month")
        .agg(
            maximum_overlap_assets=(
                "overlap_assets",
                "max",
            ),
            eligible_days=(
                "eligible_10_assets",
                "sum",
            ),
            observations=(
                "date",
                "size",
            ),
        )
        .reset_index()
    )

    monthly["eligible_month"] = (
        monthly["maximum_overlap_assets"] >= 10
    )

    monthly["year_month"] = (
        monthly["year_month"].astype(str)
    )

    monthly.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("Proxy Composite Coverage Audit")
    print("=" * 90)
    print(
        "Daily date range:",
        coverage["date"].min().date(),
        "to",
        coverage["date"].max().date(),
    )
    print(
        "Total months:",
        len(monthly),
    )
    print(
        "Months with >=10 overlapping assets:",
        int(monthly["eligible_month"].sum()),
    )
    print(
        "Months below 10 assets:",
        int((~monthly["eligible_month"]).sum()),
    )
    print("")
    print("Overlap asset-count distribution:")
    print(
        coverage["overlap_assets"]
        .describe()
        .round(2)
        .to_string()
    )

    print("")
    print("First eligible month:")
    eligible = monthly.loc[
        monthly["eligible_month"]
    ]

    if eligible.empty:
        print("None")
    else:
        print(
            eligible.iloc[0][
                "year_month"
            ]
        )

    print("")
    print("Recent monthly coverage:")
    print(
        monthly.tail(24)
        .to_string(index=False)
    )

    print("")
    print(f"Output: {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

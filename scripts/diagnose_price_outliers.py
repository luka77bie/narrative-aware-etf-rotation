from pathlib import Path

import pandas as pd


UNIVERSE_PATH = Path(
    "config/etf_universe.csv"
)

CACHE_DIR = Path("data/raw")
OUTPUT_PATH = Path(
    "outputs/price_outlier_report.csv"
)


def main() -> int:
    universe = pd.read_csv(
        UNIVERSE_PATH,
        dtype={"ticker": "string"},
    )

    universe = universe.loc[
        universe["include"]
        .astype(str)
        .str.lower()
        .eq("true")
    ]

    rows = []

    for row in universe.itertuples(index=False):
        ticker = str(row.ticker).zfill(6)
        path = CACHE_DIR / f"{ticker}.csv"

        if not path.exists():
            continue

        data = pd.read_csv(
            path,
            parse_dates=["date"],
        )

        data = data.sort_values("date")
        data["return"] = (
            data["adjusted_close"]
            .pct_change(fill_method=None)
        )

        extreme = data.loc[
            data["return"].abs() > 0.15
        ]

        for observation in extreme.itertuples(
            index=False
        ):
            rows.append(
                {
                    "ticker": ticker,
                    "name": row.name,
                    "date": observation.date,
                    "adjusted_close": (
                        observation.adjusted_close
                    ),
                    "return": getattr(observation, "return"),
                }
            )

        minimum_return = data["return"].min()
        maximum_return = data["return"].max()

        print(
            f"{ticker} {row.name}: "
            f"min={minimum_return:.2%}, "
            f"max={maximum_return:.2%}, "
            f"extreme={len(extreme)}"
        )

    report = pd.DataFrame(rows)

    report.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("")
    print(f"Extreme observations: {len(report)}")
    print(f"Output: {OUTPUT_PATH}")

    if not report.empty:
        print(
            report.sort_values("return")
            .head(30)
            .to_string(index=False)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

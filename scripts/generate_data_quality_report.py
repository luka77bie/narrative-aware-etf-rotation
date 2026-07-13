import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.market_data import (
    get_price_data,
    load_etf_universe,
)
from src.data.validation import validate_price_data


def main() -> int:
    universe = load_etf_universe(
        included_only=True
    )

    results = []

    for row in universe.itertuples(index=False):
        ticker = str(row.ticker)

        print(f"[CHECK] {ticker} {row.name}")

        try:
            data, source = get_price_data(
                ticker=ticker,
                start_date=pd.Timestamp(
                    row.active_from
                ).strftime("%Y%m%d"),
                use_online=False,
            )

            result = validate_price_data(
                data=data,
                ticker=ticker,
                min_history_days=int(
                    row.min_history_days
                ),
            )

            result["name"] = row.name
            result["primary_theme"] = (
                row.primary_theme
            )
            result["retrieval_source"] = source

            results.append(result)

            print(
                f"[{result['status']}] "
                f"{ticker}: "
                f"{result['observations']} rows"
            )

        except Exception as exc:
            results.append(
                {
                    "ticker": ticker,
                    "name": row.name,
                    "primary_theme": (
                        row.primary_theme
                    ),
                    "first_date": "",
                    "last_date": "",
                    "observations": 0,
                    "missing_close": "",
                    "duplicate_dates": "",
                    "zero_volume_days": "",
                    "stale_price_days": "",
                    "min_history_days": (
                        row.min_history_days
                    ),
                    "source": "",
                    "retrieval_source": "",
                    "status": "FAIL",
                    "message": str(exc),
                }
            )

            print(
                f"[FAIL] {ticker}: {exc}"
            )

    report = pd.DataFrame(results)

    output_path = Path(
        "outputs/data_quality_report.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        output_path,
        index=False,
    )

    print("")
    print("Data quality report complete")
    print(report["status"].value_counts())
    print(f"Report: {output_path}")

    return (
        1
        if (report["status"] == "FAIL").any()
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())

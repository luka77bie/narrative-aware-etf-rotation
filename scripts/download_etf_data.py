import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.market_data import get_price_data, load_etf_universe


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and cache ETF historical market data."
    )

    parser.add_argument(
        "--config",
        default="config/etf_universe.csv",
        help="Path to ETF universe CSV.",
    )

    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime("%Y%m%d"),
        help="End date in YYYYMMDD format.",
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        help="Disable AKShare and use cache/sample fallback only.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    universe = load_etf_universe(args.config, included_only=True)
    results = []

    for row in universe.itertuples(index=False):
        ticker = str(row.ticker)
        start_date = pd.Timestamp(row.active_from).strftime("%Y%m%d")

        print(f"[START] {ticker} {row.name}")

        try:
            data, source = get_price_data(
                ticker=ticker,
                start_date=start_date,
                end_date=args.end_date,
                use_online=not args.offline,
            )

            results.append(
                {
                    "ticker": ticker,
                    "name": row.name,
                    "status": "SUCCESS",
                    "source": source,
                    "first_date": data["date"].min(),
                    "last_date": data["date"].max(),
                    "observations": len(data),
                    "error": "",
                }
            )

            print(
                f"[SUCCESS] {ticker}: "
                f"{len(data)} rows from {source}"
            )

        except Exception as exc:
            results.append(
                {
                    "ticker": ticker,
                    "name": row.name,
                    "status": "FAILED",
                    "source": "",
                    "first_date": "",
                    "last_date": "",
                    "observations": 0,
                    "error": str(exc),
                }
            )

            print(f"[FAILED] {ticker}: {exc}")

    output_directory = Path("outputs")
    output_directory.mkdir(parents=True, exist_ok=True)

    report_path = output_directory / "download_report.csv"
    report = pd.DataFrame(results)
    report.to_csv(report_path, index=False)

    successful = int((report["status"] == "SUCCESS").sum())
    failed = int((report["status"] == "FAILED").sum())

    print("")
    print("Download complete")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Report: {report_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

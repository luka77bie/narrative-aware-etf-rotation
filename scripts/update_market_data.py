"""
Refresh Chinese ETF price caches using AkShare's Sina ETF history API.

Important:
- Sina symbols require an exchange prefix:
    SSE  -> sh510300
    SZSE -> sz159915
- The script appends only dates newer than each validated local cache.
- Existing data are preserved unless refresh coverage reaches 80%.
- Sina does not provide historical turnover, so newly appended turnover
  values remain missing. This live refresh is intended for the MOM60
  allocation signal, which uses adjusted_close.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import akshare as ak
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


UNIVERSE_PATH = (
    PROJECT_ROOT
    / "config"
    / "etf_universe.csv"
)

CACHE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
)

MINIMUM_REFRESH_COVERAGE = 0.80
REQUEST_DELAY_SECONDS = 0.8
MAX_ATTEMPTS = 3

CACHE_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "turnover",
    "source",
]


def load_universe() -> pd.DataFrame:
    universe = pd.read_csv(
        UNIVERSE_PATH,
        dtype={"ticker": "string"},
    )

    required = {
        "ticker",
        "name",
        "exchange",
        "include",
        "style",
    }

    missing = required - set(universe.columns)

    if missing:
        raise ValueError(
            "ETF universe missing columns: "
            + ", ".join(sorted(missing))
        )

    included = (
        universe["include"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    universe = universe.loc[included].copy()

    # Cash ETF is not part of the momentum ranking.
    universe = universe.loc[
        universe["style"]
        .astype(str)
        .str.lower()
        .ne("cash")
    ].copy()

    universe["ticker"] = (
        universe["ticker"]
        .astype("string")
        .str.strip()
        .str.zfill(6)
    )

    universe["exchange"] = (
        universe["exchange"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if universe.empty:
        raise ValueError(
            "No included non-cash ETFs found."
        )

    return universe.reset_index(drop=True)


def sina_symbol(
    ticker: str,
    exchange: str,
) -> str:
    if exchange == "SSE":
        return f"sh{ticker}"

    if exchange == "SZSE":
        return f"sz{ticker}"

    raise ValueError(
        f"Unsupported exchange for {ticker}: "
        f"{exchange}"
    )


def load_existing_cache(
    ticker: str,
) -> pd.DataFrame:
    path = (
        CACHE_DIRECTORY
        / f"{ticker}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found: {path}"
        )

    frame = pd.read_csv(
        path,
        dtype={"ticker": "string"},
    )

    required = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "turnover",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Cache {path} missing columns: "
            + ", ".join(sorted(missing))
        )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame = (
        frame.dropna(subset=["date"])
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    if "ticker" not in frame.columns:
        frame["ticker"] = ticker

    frame["ticker"] = (
        frame["ticker"]
        .astype("string")
        .fillna(ticker)
        .str.zfill(6)
    )

    if "source" not in frame.columns:
        frame["source"] = "existing_cache"

    return frame


def download_sina_history(
    symbol: str,
) -> pd.DataFrame:
    last_error = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):
        try:
            frame = ak.fund_etf_hist_sina(
                symbol=symbol
            )

            if frame is None or frame.empty:
                raise ValueError(
                    f"Sina returned empty data "
                    f"for {symbol}"
                )

            required = {
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            }

            missing = required - set(
                frame.columns
            )

            if missing:
                raise ValueError(
                    f"Sina output for {symbol} "
                    "missing columns: "
                    + ", ".join(
                        sorted(missing)
                    )
                )

            frame = frame[
                [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            ].copy()

            frame["date"] = pd.to_datetime(
                frame["date"],
                errors="coerce",
            )

            for column in [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]:
                frame[column] = (
                    pd.to_numeric(
                        frame[column],
                        errors="coerce",
                    )
                )

            frame = (
                frame.dropna(
                    subset=[
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                )
                .drop_duplicates(
                    subset=["date"],
                    keep="last",
                )
                .sort_values("date")
                .reset_index(drop=True)
            )

            if frame.empty:
                raise ValueError(
                    f"No valid rows returned "
                    f"for {symbol}"
                )

            return frame

        except Exception as exc:
            last_error = exc

            print(
                f"    attempt "
                f"{attempt}/{MAX_ATTEMPTS} "
                f"failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            if attempt < MAX_ATTEMPTS:
                time.sleep(
                    attempt * 1.5
                )

    raise RuntimeError(
        f"All Sina attempts failed "
        f"for {symbol}: {last_error}"
    )


def align_adjusted_close(
    existing: pd.DataFrame,
    downloaded: pd.DataFrame,
    ticker: str,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    overlap = sorted(
        set(existing["date"])
        & set(downloaded["date"])
    )

    if not overlap:
        raise ValueError(
            f"No overlapping date between "
            f"Sina and cache for {ticker}"
        )

    anchor_date = pd.Timestamp(
        overlap[-1]
    )

    old_adjusted_close = (
        pd.to_numeric(
            existing.loc[
                existing["date"]
                == anchor_date,
                "adjusted_close",
            ].iloc[-1],
            errors="coerce",
        )
    )

    sina_close = (
        pd.to_numeric(
            downloaded.loc[
                downloaded["date"]
                == anchor_date,
                "close",
            ].iloc[-1],
            errors="coerce",
        )
    )

    if (
        pd.isna(old_adjusted_close)
        or pd.isna(sina_close)
        or float(sina_close) <= 0
    ):
        raise ValueError(
            f"Invalid adjustment anchor "
            f"for {ticker}"
        )

    scale = (
        float(old_adjusted_close)
        / float(sina_close)
    )

    aligned = downloaded.copy()

    aligned["adjusted_close"] = (
        aligned["close"] * scale
    )

    return aligned, anchor_date


def build_updated_cache(
    existing: pd.DataFrame,
    downloaded: pd.DataFrame,
    ticker: str,
) -> Tuple[pd.DataFrame, int]:
    aligned, _ = align_adjusted_close(
        existing=existing,
        downloaded=downloaded,
        ticker=ticker,
    )

    old_latest = existing[
        "date"
    ].max()

    new_rows = aligned.loc[
        aligned["date"] > old_latest
    ].copy()

    if new_rows.empty:
        return existing.copy(), 0

    new_rows["ticker"] = ticker

    # Sina historical ETF API does not expose daily turnover.
    # Keep this field missing rather than inventing exact values.
    new_rows["turnover"] = pd.NA

    new_rows["source"] = (
        "akshare_sina_price_only"
    )

    new_rows = new_rows[
        CACHE_COLUMNS
    ]

    combined = pd.concat(
        [
            existing[CACHE_COLUMNS],
            new_rows,
        ],
        ignore_index=True,
    )

    combined = (
        combined.drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return combined, len(new_rows)


def main() -> int:
    universe = load_universe()

    staged: Dict[
        str,
        pd.DataFrame,
    ] = {}

    summaries = []
    failures = []

    print(
        f"Refreshing {len(universe)} "
        "momentum ETFs with "
        "AkShare Sina..."
    )
    print("")

    for index, row in enumerate(
        universe.itertuples(
            index=False
        ),
        start=1,
    ):
        ticker = str(row.ticker)
        exchange = str(row.exchange)
        name = str(row.name)

        symbol = sina_symbol(
            ticker=ticker,
            exchange=exchange,
        )

        print(
            f"[{index}/{len(universe)}] "
            f"{ticker} {name} "
            f"({symbol})"
        )

        try:
            existing = (
                load_existing_cache(
                    ticker=ticker,
                )
            )

            downloaded = (
                download_sina_history(
                    symbol=symbol,
                )
            )

            updated, added_rows = (
                build_updated_cache(
                    existing=existing,
                    downloaded=downloaded,
                    ticker=ticker,
                )
            )

            staged[ticker] = updated

            summaries.append(
                {
                    "ticker": ticker,
                    "name": name,
                    "sina_symbol": symbol,
                    "old_latest": (
                        existing[
                            "date"
                        ].max().date()
                    ),
                    "new_latest": (
                        updated[
                            "date"
                        ].max().date()
                    ),
                    "added_rows": (
                        added_rows
                    ),
                    "status": "SUCCESS",
                }
            )

            print(
                "    PASS: "
                f"{existing['date'].max().date()} "
                "-> "
                f"{updated['date'].max().date()} "
                f"(+{added_rows} rows)"
            )

        except Exception as exc:
            failures.append(
                {
                    "ticker": ticker,
                    "name": name,
                    "sina_symbol": symbol,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

            print(
                "    FAILED: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    coverage = (
        len(staged)
        / len(universe)
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIRECTORY
        / "market_refresh_summary.csv"
    )

    failure_path = (
        OUTPUT_DIRECTORY
        / "market_refresh_failures.csv"
    )

    pd.DataFrame(
        summaries
    ).to_csv(
        summary_path,
        index=False,
    )

    pd.DataFrame(
        failures
    ).to_csv(
        failure_path,
        index=False,
    )

    print("")
    print(
        f"Refresh coverage: "
        f"{len(staged)}/{len(universe)} "
        f"({coverage:.1%})"
    )

    if (
        coverage
        < MINIMUM_REFRESH_COVERAGE
    ):
        raise RuntimeError(
            "Refresh coverage is below "
            f"{MINIMUM_REFRESH_COVERAGE:.0%}. "
            "No cache files were overwritten. "
            f"Failure log: {failure_path}"
        )

    # Write only after minimum cross-sectional
    # coverage is confirmed.
    for ticker, frame in (
        staged.items()
    ):
        output_path = (
            CACHE_DIRECTORY
            / f"{ticker}.csv"
        )

        frame.to_csv(
            output_path,
            index=False,
        )

    latest_dates = [
        pd.Timestamp(
            frame["date"].max()
        )
        for frame in staged.values()
    ]

    print("")
    print("=" * 72)
    print("MARKET REFRESH COMPLETE")
    print("=" * 72)
    print(
        f"Updated caches: "
        f"{len(staged)}"
    )
    print(
        "Earliest latest date: "
        f"{min(latest_dates).date()}"
    )
    print(
        "Latest latest date: "
        f"{max(latest_dates).date()}"
    )
    print(
        f"Summary: {summary_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

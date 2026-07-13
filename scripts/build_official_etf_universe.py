import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.market_data import (
    download_akshare_price_data,
    load_cached_price_data,
    save_price_cache,
)


CANDIDATE_PATH = Path(
    "config/etf_universe_candidates.csv"
)

OUTPUT_PATH = Path(
    "config/etf_universe.csv"
)

FAILURE_REPORT_PATH = Path(
    "outputs/universe_build_failures.csv"
)


STYLE_MAP: Dict[str, str] = {
    "Broad Market": "Core",
    "Technology": "Growth",
    "Commodity Cyclical": "Cyclical",
    "Defensive Value": "Defensive",
    "Healthcare Consumer": "Defensive",
    "Cash Fixed Income": "Cash",
}


BENCHMARK_MAP: Dict[str, str] = {
    "CSI 300": "沪深300指数",
    "CSI 500": "中证500指数",
    "CSI 1000": "中证1000指数",
    "ChiNext": "创业板指数",
    "STAR 50": "上证科创板50成份指数",
    "Artificial Intelligence": "人工智能主题指数",
    "Communication": "通信主题指数",
    "Semiconductor": "半导体主题指数",
    "Robot": "机器人主题指数",
    "Computer": "计算机主题指数",
    "Gold": "黄金现货合约",
    "Nonferrous Metals": "有色金属主题指数",
    "Rare Earth": "稀土产业主题指数",
    "Coal": "煤炭主题指数",
    "Dividend": "红利主题指数",
    "Low Volatility": "红利低波指数",
    "Bank": "银行主题指数",
    "Power Utilities": "电力及公用事业主题指数",
    "Healthcare": "医疗健康主题指数",
    "Consumer": "消费主题指数",
    "Money Market": "货币市场工具",
}


def infer_exchange(ticker: str) -> str:
    ticker = str(ticker).zfill(6)

    if ticker.startswith(("5", "6")):
        return "SSE"

    if ticker.startswith(("1", "3")):
        return "SZSE"

    raise ValueError(
        f"Unable to infer exchange for ticker {ticker}"
    )


def get_data_with_retry(
    ticker: str,
    max_attempts: int = 3,
    retry_delay: int = 3,
) -> pd.DataFrame:
    """
    Load cached data first. Download only when cache is missing.

    If online download fails, retry with exponential backoff.
    """
    try:
        data = load_cached_price_data(
            ticker=ticker,
            cache_directory="data/raw",
        )

        print(
            f"[CACHE] {ticker}: {len(data)} rows"
        )

        return data

    except FileNotFoundError:
        pass

    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            print(
                f"[DOWNLOAD] {ticker}: "
                f"attempt {attempt}/{max_attempts}"
            )

            data = download_akshare_price_data(
                ticker=ticker,
                start_date="19900101",
            )

            save_price_cache(
                data=data,
                ticker=ticker,
            )

            return data

        except Exception as exc:
            last_error = exc

            print(
                f"[RETRY] {ticker}: "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < max_attempts:
                time.sleep(
                    retry_delay * attempt
                )

    raise RuntimeError(
        f"All download attempts failed for "
        f"{ticker}: {last_error}"
    )


def load_previous_universe() -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        OUTPUT_PATH,
        dtype={"ticker": "string"},
    )


def main() -> int:
    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"Candidate file not found: "
            f"{CANDIDATE_PATH}"
        )

    candidates = pd.read_csv(
        CANDIDATE_PATH,
        dtype={"ticker": "string"},
    )

    selected_mask = (
        candidates["selected"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    selected = candidates.loc[
        selected_mask
    ].copy()

    if selected.empty:
        raise ValueError(
            "No selected ETFs found."
        )

    if selected["ticker"].duplicated().any():
        raise ValueError(
            "Selected candidates contain duplicate tickers."
        )

    previous_universe = load_previous_universe()

    previous_by_ticker = {}

    if not previous_universe.empty:
        previous_by_ticker = (
            previous_universe
            .set_index("ticker")
            .to_dict("index")
        )

    rows = []
    failures = []

    for row in selected.itertuples(index=False):
        ticker = str(row.ticker).zfill(6)

        print(
            f"\n[START] {ticker} {row.name}"
        )

        data = None
        error_message = ""

        try:
            data = get_data_with_retry(
                ticker=ticker,
            )

        except Exception as exc:
            error_message = str(exc)

            print(
                f"[FAILED] {ticker}: "
                f"{error_message}"
            )

        if data is not None and not data.empty:
            first_date = pd.Timestamp(
                data["date"].min()
            )

            observations = len(data)
            include = observations >= 252
            data_status = "AVAILABLE"

        elif ticker in previous_by_ticker:
            previous = previous_by_ticker[ticker]

            first_date = pd.Timestamp(
                previous["active_from"]
            )

            observations = int(
                previous.get(
                    "observations",
                    0,
                )
            )

            include = bool(
                previous.get(
                    "include",
                    True,
                )
            )

            data_status = "PRESERVED_PREVIOUS"

        else:
            first_date = pd.NaT
            observations = 0
            include = False
            data_status = "DOWNLOAD_FAILED"

        rows.append(
            {
                "ticker": ticker,
                "name": row.name,
                "exchange": infer_exchange(ticker),
                "primary_theme": row.primary_theme,
                "secondary_theme": row.exposure,
                "style": STYLE_MAP.get(
                    row.primary_theme,
                    "Other",
                ),
                "benchmark_index": BENCHMARK_MAP.get(
                    row.exposure,
                    row.exposure,
                ),
                "inception_date": (
                    first_date.date()
                    if pd.notna(first_date)
                    else ""
                ),
                "active_from": (
                    first_date.date()
                    if pd.notna(first_date)
                    else ""
                ),
                "active_to": "",
                "min_history_days": 252,
                "include": include,
                "observations": observations,
                "data_status": data_status,
            }
        )

        if error_message:
            failures.append(
                {
                    "ticker": ticker,
                    "name": row.name,
                    "error": error_message,
                    "data_status": data_status,
                }
            )

    universe = pd.DataFrame(rows)

    universe = universe.sort_values(
        [
            "primary_theme",
            "secondary_theme",
        ]
    ).reset_index(drop=True)

    universe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    FAILURE_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(failures).to_csv(
        FAILURE_REPORT_PATH,
        index=False,
    )

    print("")
    print("Official ETF universe generated")
    print(f"Rows: {len(universe)}")
    print(
        f"Included: "
        f"{int(universe['include'].sum())}"
    )
    print(
        "Available: "
        f"{int((universe['data_status'] == 'AVAILABLE').sum())}"
    )
    print(
        "Preserved previous: "
        f"{int((universe['data_status'] == 'PRESERVED_PREVIOUS').sum())}"
    )
    print(
        "Download failed: "
        f"{int((universe['data_status'] == 'DOWNLOAD_FAILED').sum())}"
    )
    print(f"Output: {OUTPUT_PATH}")
    print(
        f"Failure report: "
        f"{FAILURE_REPORT_PATH}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

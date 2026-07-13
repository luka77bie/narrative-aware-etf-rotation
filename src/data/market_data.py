from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from src.data.preprocessing import standardise_price_data


REQUIRED_UNIVERSE_COLUMNS = {
    "ticker",
    "name",
    "exchange",
    "primary_theme",
    "secondary_theme",
    "style",
    "benchmark_index",
    "inception_date",
    "active_from",
    "active_to",
    "min_history_days",
    "include",
}


AKSHARE_RAW_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "turnover",
}


def load_etf_universe(
    config_path: Union[str, Path] = "config/etf_universe.csv",
    included_only: bool = True,
) -> pd.DataFrame:
    """Load and validate the ETF universe configuration."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"ETF universe file not found: {path}")

    universe = pd.read_csv(path, dtype={"ticker": "string"})

    missing_columns = REQUIRED_UNIVERSE_COLUMNS - set(universe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"ETF universe is missing required columns: {missing}")

    if universe.empty:
        raise ValueError("ETF universe is empty.")

    universe["ticker"] = universe["ticker"].str.strip()
    universe["exchange"] = universe["exchange"].str.upper().str.strip()

    for column in ["inception_date", "active_from", "active_to"]:
        universe[column] = pd.to_datetime(universe[column], errors="coerce")

    universe["min_history_days"] = pd.to_numeric(
        universe["min_history_days"],
        errors="coerce",
    )

    universe["include"] = (
        universe["include"]
        .astype("string")
        .str.lower()
        .map({"true": True, "false": False})
    )

    if universe["ticker"].isna().any() or universe["ticker"].eq("").any():
        raise ValueError("ETF universe contains an empty ticker.")

    if universe["ticker"].duplicated().any():
        duplicates = universe.loc[
            universe["ticker"].duplicated(keep=False),
            "ticker",
        ].tolist()
        raise ValueError(f"Duplicate ETF tickers found: {duplicates}")

    invalid_exchanges = set(universe["exchange"].dropna()) - {"SSE", "SZSE"}
    if invalid_exchanges:
        raise ValueError(
            f"Unsupported exchange values: {sorted(invalid_exchanges)}"
        )

    if universe["active_from"].isna().any():
        raise ValueError("Every ETF must have a valid active_from date.")

    if universe["min_history_days"].isna().any():
        raise ValueError("Every ETF must have min_history_days.")

    if (universe["min_history_days"] <= 0).any():
        raise ValueError("min_history_days must be positive.")

    if universe["include"].isna().any():
        raise ValueError("The include column must contain true or false.")

    if included_only:
        universe = universe.loc[universe["include"]].copy()

    return universe.reset_index(drop=True)


def load_local_price_data(
    ticker: str,
    data_directory: Union[str, Path] = "data/sample",
) -> pd.DataFrame:
    """Load one ETF from a local CSV file."""
    ticker = str(ticker).strip()
    file_path = Path(data_directory) / f"{ticker}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Local price file not found for ticker {ticker}: {file_path}"
        )

    raw_data = pd.read_csv(file_path)

    return standardise_price_data(
        data=raw_data,
        ticker=ticker,
        source="local_csv",
    )


def _prepare_akshare_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """Convert unadjusted AKShare ETF output into internal English columns."""
    if data is None or data.empty:
        raise ValueError("AKShare returned empty unadjusted ETF data.")

    frame = data.rename(columns=AKSHARE_RAW_COLUMN_MAP).copy()

    required = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    }

    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "AKShare unadjusted data is missing columns: "
            + ", ".join(sorted(missing))
        )

    return frame[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ]
    ]


def _prepare_akshare_adjusted_data(data: pd.DataFrame) -> pd.DataFrame:
    """Extract adjusted close from adjusted AKShare ETF output."""
    if data is None or data.empty:
        raise ValueError("AKShare returned empty adjusted ETF data.")

    frame = data.rename(
        columns={
            "日期": "date",
            "收盘": "adjusted_close",
        }
    ).copy()

    required = {"date", "adjusted_close"}
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            "AKShare adjusted data is missing columns: "
            + ", ".join(sorted(missing))
        )

    return frame[["date", "adjusted_close"]]


def download_akshare_price_data(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download unadjusted OHLCV and back-adjusted close from AKShare.

    The unadjusted request preserves observable market prices.
    The adjusted request provides adjusted_close for return calculations.
    """
    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError(
            "AKShare is not installed. Run: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    ticker = str(ticker).strip()
    end_date = end_date or datetime.now().strftime("%Y%m%d")

    raw_response = ak.fund_etf_hist_em(
        symbol=ticker,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="",
    )

    adjusted_response = ak.fund_etf_hist_em(
        symbol=ticker,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="hfq",
    )

    raw_data = _prepare_akshare_raw_data(raw_response)
    adjusted_data = _prepare_akshare_adjusted_data(adjusted_response)

    merged = raw_data.merge(
        adjusted_data,
        on="date",
        how="left",
        validate="one_to_one",
    )

    if merged["adjusted_close"].isna().any():
        raise ValueError(
            f"Adjusted close could not be aligned for ticker {ticker}."
        )

    return standardise_price_data(
        data=merged,
        ticker=ticker,
        source="akshare",
    )


def save_price_cache(
    data: pd.DataFrame,
    ticker: str,
    cache_directory: Union[str, Path] = "data/raw",
) -> Path:
    """Save standardised ETF data to the local raw-data cache."""
    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)

    output_path = directory / f"{ticker}.csv"
    data.to_csv(output_path, index=False)

    return output_path


def load_cached_price_data(
    ticker: str,
    cache_directory: Union[str, Path] = "data/raw",
) -> pd.DataFrame:
    """Load a previously downloaded ETF cache."""
    ticker = str(ticker).strip()
    file_path = Path(cache_directory) / f"{ticker}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cached price file not found for ticker {ticker}: {file_path}"
        )

    data = pd.read_csv(file_path)

    data = data.drop(
        columns=["ticker", "source"],
        errors="ignore",
    )

    return standardise_price_data(
        data=data,
        ticker=ticker,
        source="raw_cache",
    )


def get_price_data(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
    cache_directory: Union[str, Path] = "data/raw",
    sample_directory: Union[str, Path] = "data/sample",
    use_online: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Retrieve ETF data through the full fallback chain.

    Priority:
    1. AKShare online download
    2. Raw local cache
    3. Bundled sample CSV
    """
    errors = []

    if use_online:
        try:
            data = download_akshare_price_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            save_price_cache(
                data=data,
                ticker=ticker,
                cache_directory=cache_directory,
            )
            return data, "akshare"
        except Exception as exc:
            errors.append(f"akshare={type(exc).__name__}: {exc}")

    try:
        data = load_cached_price_data(
            ticker=ticker,
            cache_directory=cache_directory,
        )
        return data, "raw_cache"
    except Exception as exc:
        errors.append(f"raw_cache={type(exc).__name__}: {exc}")

    try:
        data = load_local_price_data(
            ticker=ticker,
            data_directory=sample_directory,
        )
        return data, "sample_csv"
    except Exception as exc:
        errors.append(f"sample_csv={type(exc).__name__}: {exc}")

    error_message = " | ".join(errors)

    raise RuntimeError(
        f"All data sources failed for ticker {ticker}. {error_message}"
    )

from pathlib import Path
from typing import Union

import pandas as pd


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


def load_etf_universe(
    config_path: "Union[str, Path]" = "config/etf_universe.csv",
    included_only: bool = True,
) -> pd.DataFrame:
    """
    Load and validate the ETF universe configuration.

    Parameters
    ----------
    config_path:
        Path to the ETF universe CSV.
    included_only:
        If True, return only rows where include is True.

    Returns
    -------
    pd.DataFrame
        Validated ETF metadata.
    """
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
    data_directory: "Union[str, Path]" = "data/sample",
) -> pd.DataFrame:
    """Load one ETF from a local CSV fallback file."""
    from src.data.preprocessing import standardise_price_data

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

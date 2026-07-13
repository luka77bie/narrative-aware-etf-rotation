from typing import List

import pandas as pd


REQUIRED_PRICE_COLUMNS = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "turnover",
}

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "turnover",
]


def standardise_price_data(
    data: pd.DataFrame,
    ticker: str,
    source: str,
) -> pd.DataFrame:
    """Validate and standardise one ETF price DataFrame."""
    frame = data.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    missing_columns = REQUIRED_PRICE_COLUMNS - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Price data is missing required columns: {missing}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["ticker"] = str(ticker).strip()
    frame["source"] = str(source).strip()

    frame = (
        frame.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

    invalid_dates = frame["date"].isna()
    if invalid_dates.any():
        raise ValueError("Price data contains invalid dates.")

    price_columns: List[str] = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
    ]

    if frame[price_columns].isna().any().any():
        raise ValueError("Price data contains missing or non-numeric prices.")

    if (frame[price_columns] <= 0).any().any():
        raise ValueError("Price data contains non-positive prices.")

    if (frame["volume"] < 0).any():
        raise ValueError("Price data contains negative volume.")

    invalid_ohlc = (
        (frame["high"] < frame["low"])
        | (frame["high"] < frame["open"])
        | (frame["high"] < frame["close"])
        | (frame["low"] > frame["open"])
        | (frame["low"] > frame["close"])
    )

    if invalid_ohlc.any():
        raise ValueError("Price data contains inconsistent OHLC values.")

    output_columns = [
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

    return frame[output_columns]

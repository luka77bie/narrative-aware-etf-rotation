from typing import Dict

import pandas as pd


def count_stale_price_days(
    data: pd.DataFrame,
    price_column: str = "adjusted_close",
) -> int:
    """
    Count trading days where the selected price is unchanged
    from the previous observation.
    """
    if price_column not in data.columns:
        raise ValueError(
            f"Price column not found: {price_column}"
        )

    return int(
        data[price_column]
        .diff()
        .eq(0)
        .sum()
    )


def validate_price_data(
    data: pd.DataFrame,
    ticker: str,
    min_history_days: int,
) -> Dict[str, object]:
    """
    Produce a data-quality summary for one ETF.
    """
    required_columns = {
        "date",
        "ticker",
        "close",
        "adjusted_close",
        "volume",
        "source",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        return {
            "ticker": ticker,
            "first_date": "",
            "last_date": "",
            "observations": len(data),
            "missing_close": "",
            "duplicate_dates": "",
            "zero_volume_days": "",
            "stale_price_days": "",
            "min_history_days": min_history_days,
            "source": "",
            "status": "FAIL",
            "message": (
                "Missing required columns: "
                + ", ".join(sorted(missing_columns))
            ),
        }

    observations = len(data)

    missing_close = int(
        data["adjusted_close"].isna().sum()
    )

    duplicate_dates = int(
        data["date"].duplicated().sum()
    )

    zero_volume_days = int(
        data["volume"].eq(0).sum()
    )

    stale_price_days = count_stale_price_days(
        data=data,
        price_column="adjusted_close",
    )

    source = (
        str(data["source"].iloc[0])
        if not data.empty
        else ""
    )

    messages = []

    if observations < min_history_days:
        messages.append(
            f"Insufficient history: "
            f"{observations} < {min_history_days}"
        )

    if missing_close > 0:
        messages.append(
            f"Missing adjusted_close: {missing_close}"
        )

    if duplicate_dates > 0:
        messages.append(
            f"Duplicate dates: {duplicate_dates}"
        )

    if zero_volume_days > 0:
        messages.append(
            f"Zero-volume days: {zero_volume_days}"
        )

    if stale_price_days > 5:
        messages.append(
            f"High stale-price count: {stale_price_days}"
        )

    if missing_close > 0 or duplicate_dates > 0:
        status = "FAIL"

    elif (
        observations < min_history_days
        or zero_volume_days > 0
        or stale_price_days > 5
        or source == "sample_csv"
    ):
        status = "WARNING"

    else:
        status = "PASS"

    return {
        "ticker": ticker,
        "first_date": (
            data["date"].min()
            if not data.empty
            else ""
        ),
        "last_date": (
            data["date"].max()
            if not data.empty
            else ""
        ),
        "observations": observations,
        "missing_close": missing_close,
        "duplicate_dates": duplicate_dates,
        "zero_volume_days": zero_volume_days,
        "stale_price_days": stale_price_days,
        "min_history_days": min_history_days,
        "source": source,
        "status": status,
        "message": (
            " | ".join(messages)
            if messages
            else "Data quality checks passed."
        ),
    }

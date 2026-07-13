import pandas as pd

from src.data.validation import (
    count_stale_price_days,
    validate_price_data,
)


def make_valid_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "ticker": [
                "510300",
                "510300",
                "510300",
            ],
            "close": [
                3.40,
                3.45,
                3.50,
            ],
            "adjusted_close": [
                3.40,
                3.45,
                3.50,
            ],
            "volume": [
                1000,
                1200,
                1100,
            ],
            "source": [
                "raw_cache",
                "raw_cache",
                "raw_cache",
            ],
        }
    )


def test_count_stale_price_days() -> None:
    data = make_valid_data()
    data.loc[2, "adjusted_close"] = 3.45

    assert count_stale_price_days(data) == 1


def test_valid_data_passes() -> None:
    result = validate_price_data(
        data=make_valid_data(),
        ticker="510300",
        min_history_days=3,
    )

    assert result["status"] == "PASS"
    assert result["observations"] == 3


def test_insufficient_history_warns() -> None:
    result = validate_price_data(
        data=make_valid_data(),
        ticker="510300",
        min_history_days=252,
    )

    assert result["status"] == "WARNING"


def test_duplicate_dates_fail() -> None:
    data = make_valid_data()
    data.loc[2, "date"] = data.loc[1, "date"]

    result = validate_price_data(
        data=data,
        ticker="510300",
        min_history_days=3,
    )

    assert result["status"] == "FAIL"


def test_missing_adjusted_close_fails() -> None:
    data = make_valid_data()
    data.loc[1, "adjusted_close"] = None

    result = validate_price_data(
        data=data,
        ticker="510300",
        min_history_days=3,
    )

    assert result["status"] == "FAIL"

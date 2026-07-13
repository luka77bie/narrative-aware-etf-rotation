import pandas as pd
import pytest

from src.data.preprocessing import standardise_price_data


def make_valid_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02"],
            "open": [3.40, 3.30],
            "high": [3.50, 3.40],
            "low": [3.35, 3.25],
            "close": [3.45, 3.35],
            "adjusted_close": [3.45, 3.35],
            "volume": [1000, 900],
            "turnover": [3450, 3015],
        }
    )


def test_standardise_price_data_sorts_dates() -> None:
    result = standardise_price_data(
        make_valid_data(),
        ticker="510300",
        source="test",
    )

    assert result["date"].is_monotonic_increasing
    assert result["ticker"].eq("510300").all()
    assert result["source"].eq("test").all()


def test_non_positive_price_raises_error() -> None:
    data = make_valid_data()
    data.loc[0, "close"] = 0

    with pytest.raises(ValueError, match="non-positive"):
        standardise_price_data(data, "510300", "test")


def test_invalid_ohlc_raises_error() -> None:
    data = make_valid_data()
    data.loc[0, "high"] = 3.20

    with pytest.raises(ValueError, match="inconsistent OHLC"):
        standardise_price_data(data, "510300", "test")

from pathlib import Path

import pandas as pd
import pytest

from src.data.market_data import load_etf_universe


def test_load_etf_universe_returns_included_rows_only() -> None:
    universe = load_etf_universe()

    assert not universe.empty
    assert universe["include"].all()
    assert "512660" not in universe["ticker"].tolist()


def test_ticker_is_loaded_as_string() -> None:
    universe = load_etf_universe()

    assert pd.api.types.is_string_dtype(universe["ticker"])
    assert "510300" in universe["ticker"].tolist()


def test_missing_config_file_raises_error(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_etf_universe(missing_file)


def test_duplicate_tickers_raise_error(tmp_path: Path) -> None:
    test_file = tmp_path / "duplicate_universe.csv"

    test_file.write_text(
        "ticker,name,exchange,primary_theme,secondary_theme,style,"
        "benchmark_index,inception_date,active_from,active_to,"
        "min_history_days,include\n"
        "510300,ETF A,SSE,Broad,Large,Core,Index,"
        "2012-05-04,2012-05-28,,252,true\n"
        "510300,ETF B,SSE,Broad,Large,Core,Index,"
        "2012-05-04,2012-05-28,,252,true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate ETF tickers"):
        load_etf_universe(test_file)

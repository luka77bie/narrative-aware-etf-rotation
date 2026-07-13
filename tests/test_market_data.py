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


def test_load_local_price_data() -> None:
    from src.data.market_data import load_local_price_data

    data = load_local_price_data("510300")

    assert not data.empty
    assert data["ticker"].eq("510300").all()
    assert data["source"].eq("local_csv").all()
    assert data["date"].is_monotonic_increasing


def test_missing_local_price_file_raises_error() -> None:
    from src.data.market_data import load_local_price_data

    with pytest.raises(FileNotFoundError):
        load_local_price_data("999999")


def test_get_price_data_uses_sample_when_offline(
    tmp_path: Path,
) -> None:
    from src.data.market_data import get_price_data

    data, source = get_price_data(
        ticker="510300",
        start_date="20240101",
        cache_directory=tmp_path / "empty_cache",
        sample_directory="data/sample",
        use_online=False,
    )

    assert not data.empty
    assert source == "sample_csv"


def test_get_price_data_uses_raw_cache_before_sample(
    tmp_path: Path,
) -> None:
    from src.data.market_data import get_price_data

    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()

    sample_data = pd.read_csv("data/sample/510300.csv")
    sample_data.to_csv(
        cache_directory / "510300.csv",
        index=False,
    )

    data, source = get_price_data(
        ticker="510300",
        start_date="20240101",
        cache_directory=cache_directory,
        sample_directory="data/sample",
        use_online=False,
    )

    assert not data.empty
    assert source == "raw_cache"


def test_get_price_data_raises_when_all_sources_fail(
    tmp_path: Path,
) -> None:
    from src.data.market_data import get_price_data

    with pytest.raises(RuntimeError, match="All data sources failed"):
        get_price_data(
            ticker="999999",
            start_date="20240101",
            cache_directory=tmp_path / "cache",
            sample_directory=tmp_path / "sample",
            use_online=False,
        )

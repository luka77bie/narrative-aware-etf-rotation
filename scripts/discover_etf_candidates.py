import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


THEME_KEYWORDS: Dict[str, List[str]] = {
    "Broad Market": [
        "沪深300",
        "中证500",
        "中证1000",
        "中证A500",
        "创业板",
        "科创50",
    ],
    "Technology": [
        "人工智能",
        "AI",
        "通信",
        "半导体",
        "芯片",
        "机器人",
        "计算机",
    ],
    "Commodity Cyclical": [
        "黄金",
        "有色",
        "稀土",
        "煤炭",
        "石油",
    ],
    "Defensive Value": [
        "红利",
        "低波",
        "银行",
        "电力",
        "公用事业",
    ],
    "Healthcare Consumer": [
        "创新药",
        "医药",
        "医疗",
        "消费",
        "食品饮料",
    ],
    "Cash Fixed Income": [
        "货币",
        "短债",
        "国债",
        "政金债",
    ],
}


def normalise_catalog(data: pd.DataFrame) -> pd.DataFrame:
    """Normalise the ETF spot catalog returned by AKShare."""
    column_map = {
        "代码": "ticker",
        "名称": "name",
        "最新价": "latest_price",
        "成交量": "volume",
        "成交额": "turnover",
    }

    catalog = data.rename(columns=column_map).copy()

    required = {"ticker", "name"}
    missing = required - set(catalog.columns)

    if missing:
        raise ValueError(
            "ETF catalog missing required columns: "
            + ", ".join(sorted(missing))
        )

    catalog["ticker"] = (
        catalog["ticker"]
        .astype("string")
        .str.strip()
        .str.zfill(6)
    )

    catalog["name"] = (
        catalog["name"]
        .astype("string")
        .str.strip()
    )

    catalog = (
        catalog
        .drop_duplicates(subset=["ticker"])
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    return catalog


def build_shortlist(catalog: pd.DataFrame) -> pd.DataFrame:
    """Match ETFs against the theme keyword dictionary."""
    results = []

    for theme, keywords in THEME_KEYWORDS.items():
        for keyword in keywords:
            mask = catalog["name"].str.contains(
                keyword,
                case=False,
                na=False,
                regex=False,
            )

            matches = catalog.loc[mask].copy()

            if matches.empty:
                continue

            matches["candidate_theme"] = theme
            matches["matched_keyword"] = keyword
            results.append(matches)

    if not results:
        return pd.DataFrame(
            columns=[
                "ticker",
                "name",
                "candidate_theme",
                "matched_keyword",
            ]
        )

    shortlist = pd.concat(
        results,
        ignore_index=True,
    )

    shortlist = (
        shortlist
        .drop_duplicates(
            subset=[
                "ticker",
                "candidate_theme",
            ]
        )
        .sort_values(
            [
                "candidate_theme",
                "turnover",
            ],
            ascending=[
                True,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    return shortlist


def main() -> int:
    import akshare as ak

    output_directory = Path("outputs")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("[START] Downloading current ETF catalog")

    raw_catalog = ak.fund_etf_spot_em()
    catalog = normalise_catalog(raw_catalog)
    shortlist = build_shortlist(catalog)

    catalog_path = output_directory / "etf_catalog.csv"
    shortlist_path = output_directory / "etf_candidate_shortlist.csv"

    catalog.to_csv(
        catalog_path,
        index=False,
    )

    shortlist.to_csv(
        shortlist_path,
        index=False,
    )

    print(f"[SUCCESS] ETF catalog rows: {len(catalog)}")
    print(f"[SUCCESS] Candidate rows: {len(shortlist)}")
    print(f"Catalog: {catalog_path}")
    print(f"Shortlist: {shortlist_path}")

    if not shortlist.empty:
        print("")
        print(
            shortlist[
                [
                    "ticker",
                    "name",
                    "candidate_theme",
                    "matched_keyword",
                ]
            ]
            .head(100)
            .to_string(index=False)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

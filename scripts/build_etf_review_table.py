from pathlib import Path
from typing import Dict, List

import pandas as pd


TARGET_EXPOSURES: Dict[str, Dict[str, object]] = {
    "CSI 300": {
        "theme": "Broad Market",
        "keywords": ["沪深300"],
        "target_count": 1,
    },
    "CSI 500": {
        "theme": "Broad Market",
        "keywords": ["中证500"],
        "target_count": 1,
    },
    "CSI 1000": {
        "theme": "Broad Market",
        "keywords": ["中证1000"],
        "target_count": 1,
    },
    "ChiNext": {
        "theme": "Broad Market",
        "keywords": ["创业板"],
        "target_count": 1,
    },
    "STAR 50": {
        "theme": "Broad Market",
        "keywords": ["科创50"],
        "target_count": 1,
    },
    "Artificial Intelligence": {
        "theme": "Technology",
        "keywords": ["人工智能", "AI"],
        "target_count": 1,
    },
    "Communication": {
        "theme": "Technology",
        "keywords": ["通信"],
        "target_count": 1,
    },
    "Semiconductor": {
        "theme": "Technology",
        "keywords": ["半导体", "芯片"],
        "target_count": 1,
    },
    "Robot": {
        "theme": "Technology",
        "keywords": ["机器人"],
        "target_count": 1,
    },
    "Computer": {
        "theme": "Technology",
        "keywords": ["计算机"],
        "target_count": 1,
    },
    "Gold": {
        "theme": "Commodity Cyclical",
        "keywords": ["黄金"],
        "target_count": 1,
    },
    "Nonferrous Metals": {
        "theme": "Commodity Cyclical",
        "keywords": ["有色"],
        "target_count": 1,
    },
    "Rare Earth": {
        "theme": "Commodity Cyclical",
        "keywords": ["稀土"],
        "target_count": 1,
    },
    "Coal": {
        "theme": "Commodity Cyclical",
        "keywords": ["煤炭"],
        "target_count": 1,
    },
    "Dividend": {
        "theme": "Defensive Value",
        "keywords": ["红利"],
        "target_count": 1,
    },
    "Low Volatility": {
        "theme": "Defensive Value",
        "keywords": ["低波"],
        "target_count": 1,
    },
    "Bank": {
        "theme": "Defensive Value",
        "keywords": ["银行"],
        "target_count": 1,
    },
    "Power Utilities": {
        "theme": "Defensive Value",
        "keywords": ["电力", "公用事业"],
        "target_count": 1,
    },
    "Innovative Drug": {
        "theme": "Healthcare Consumer",
        "keywords": ["创新药"],
        "target_count": 1,
    },
    "Healthcare": {
        "theme": "Healthcare Consumer",
        "keywords": ["医疗", "医药"],
        "target_count": 1,
    },
    "Consumer": {
        "theme": "Healthcare Consumer",
        "keywords": ["消费", "食品饮料"],
        "target_count": 1,
    },
    "Money Market": {
        "theme": "Cash Fixed Income",
        "keywords": ["货币"],
        "target_count": 1,
    },
    "Short Bond": {
        "theme": "Cash Fixed Income",
        "keywords": ["短债"],
        "target_count": 1,
    },
    "Government Bond": {
        "theme": "Cash Fixed Income",
        "keywords": ["国债", "政金债"],
        "target_count": 1,
    },
}


def match_exposure(
    catalog: pd.DataFrame,
    keywords: List[str],
) -> pd.DataFrame:
    mask = pd.Series(False, index=catalog.index)

    for keyword in keywords:
        mask = mask | catalog["name"].str.contains(
            keyword,
            case=False,
            na=False,
            regex=False,
        )

    matches = catalog.loc[mask].copy()

    if "turnover" in matches.columns:
        matches["turnover"] = pd.to_numeric(
            matches["turnover"],
            errors="coerce",
        )

        matches = matches.sort_values(
            "turnover",
            ascending=False,
            na_position="last",
        )

    return matches


def main() -> int:
    catalog_path = Path("outputs/etf_catalog.csv")

    if not catalog_path.exists():
        raise FileNotFoundError(
            "ETF catalog not found. Run "
            "scripts/discover_etf_candidates.py first."
        )

    catalog = pd.read_csv(
        catalog_path,
        dtype={"ticker": "string"},
    )

    catalog["ticker"] = catalog["ticker"].str.zfill(6)

    review_rows = []

    for exposure, specification in TARGET_EXPOSURES.items():
        matches = match_exposure(
            catalog=catalog,
            keywords=specification["keywords"],
        )

        top_matches = matches.head(5)

        if top_matches.empty:
            review_rows.append(
                {
                    "exposure": exposure,
                    "primary_theme": specification["theme"],
                    "ticker": "",
                    "name": "",
                    "turnover": "",
                    "candidate_rank": "",
                    "selected": False,
                    "review_status": "NO_MATCH",
                    "review_notes": "",
                }
            )
            continue

        for rank, row in enumerate(
            top_matches.itertuples(index=False),
            start=1,
        ):
            review_rows.append(
                {
                    "exposure": exposure,
                    "primary_theme": specification["theme"],
                    "ticker": row.ticker,
                    "name": row.name,
                    "turnover": getattr(row, "turnover", ""),
                    "candidate_rank": rank,
                    "selected": rank == 1,
                    "review_status": "PENDING",
                    "review_notes": "",
                }
            )

    review = pd.DataFrame(review_rows)

    output_path = Path(
        "config/etf_universe_candidates.csv"
    )

    review.to_csv(
        output_path,
        index=False,
    )

    print(f"Review rows: {len(review)}")
    print(f"Output: {output_path}")
    print("")
    print(
        review[
            [
                "exposure",
                "ticker",
                "name",
                "turnover",
                "candidate_rank",
                "selected",
            ]
        ].to_string(index=False)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

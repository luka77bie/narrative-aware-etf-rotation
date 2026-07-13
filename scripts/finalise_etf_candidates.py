from pathlib import Path
from typing import List, Set

import pandas as pd


CANDIDATE_PATH = Path("config/etf_universe_candidates.csv")
CATALOG_PATH = Path("outputs/etf_catalog.csv")

CROSS_BORDER_KEYWORDS: List[str] = [
    "中韩",
    "韩国",
    "港股",
    "恒生",
    "纳斯达克",
    "标普",
    "日经",
    "德国",
    "法国",
    "海外",
    "全球",
]

EXPOSURE_PRIORITY = [
    "CSI 300",
    "CSI 500",
    "CSI 1000",
    "ChiNext",
    "STAR 50",
    "Artificial Intelligence",
    "Communication",
    "Semiconductor",
    "Robot",
    "Computer",
    "Gold",
    "Nonferrous Metals",
    "Rare Earth",
    "Coal",
    # 先选 Low Volatility，保留 512890 给这一 exposure
    "Low Volatility",
    "Dividend",
    "Bank",
    "Power Utilities",
    "Innovative Drug",
    "Healthcare",
    "Consumer",
    "Money Market",
]


def contains_excluded_keyword(name: str) -> bool:
    name = str(name)

    return any(
        keyword in name
        for keyword in CROSS_BORDER_KEYWORDS
    )


def normalise_selected_column(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()

    frame["selected"] = (
        frame["selected"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    return frame


def select_best_candidate(
    candidates: pd.DataFrame,
    used_tickers: Set[str],
) -> int:
    """
    Return the row index of the best valid candidate.

    Selection priority:
    1. Lower candidate_rank
    2. Higher turnover
    3. Not already selected elsewhere
    4. Not a cross-border product
    """
    frame = candidates.copy()

    frame["candidate_rank"] = pd.to_numeric(
        frame["candidate_rank"],
        errors="coerce",
    )

    frame["turnover"] = pd.to_numeric(
        frame["turnover"],
        errors="coerce",
    )

    frame = frame.sort_values(
        ["candidate_rank", "turnover"],
        ascending=[True, False],
        na_position="last",
    )

    for index, row in frame.iterrows():
        ticker = str(row["ticker"]).zfill(6)
        name = str(row["name"])

        if ticker in used_tickers:
            continue

        if contains_excluded_keyword(name):
            continue

        return index

    raise ValueError(
        f"No valid candidate available for "
        f"{candidates['exposure'].iloc[0]}"
    )


def add_bond_candidates(
    candidates: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Append up to five domestic bond ETF candidates."""
    bond_keywords = [
        "政金债",
        "国债",
        "短融",
        "短债",
        "信用债",
        "债券",
    ]

    mask = pd.Series(False, index=catalog.index)

    for keyword in bond_keywords:
        mask = mask | catalog["name"].str.contains(
            keyword,
            na=False,
            regex=False,
        )

    bond_candidates = catalog.loc[mask].copy()

    # 排除可转债与跨境债券产品
    bond_candidates = bond_candidates.loc[
        ~bond_candidates["name"].str.contains(
            "可转债|转债|海外|全球|美元",
            na=False,
            regex=True,
        )
    ]

    bond_candidates["turnover"] = pd.to_numeric(
        bond_candidates.get("turnover"),
        errors="coerce",
    )

    bond_candidates = (
        bond_candidates
        .sort_values(
            "turnover",
            ascending=False,
            na_position="last",
        )
        .drop_duplicates(subset=["ticker"])
        .head(5)
        .reset_index(drop=True)
    )

    rows = []

    for rank, row in bond_candidates.iterrows():
        rows.append(
            {
                "exposure": "Bond",
                "primary_theme": "Cash Fixed Income",
                "ticker": str(row["ticker"]).zfill(6),
                "name": row["name"],
                "turnover": row.get("turnover", ""),
                "candidate_rank": rank + 1,
                "selected": False,
                "review_status": "PENDING",
                "review_notes": "",
            }
        )

    # 删除旧 Bond 行，防止重复执行脚本时反复追加
    candidates = candidates.loc[
        candidates["exposure"] != "Bond"
    ].copy()

    if rows:
        candidates = pd.concat(
            [candidates, pd.DataFrame(rows)],
            ignore_index=True,
        )

    return candidates


def main() -> int:
    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {CANDIDATE_PATH}"
        )

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"ETF catalog not found: {CATALOG_PATH}"
        )

    candidates = pd.read_csv(
        CANDIDATE_PATH,
        dtype={"ticker": "string"},
    )

    catalog = pd.read_csv(
        CATALOG_PATH,
        dtype={"ticker": "string"},
    )

    candidates["ticker"] = (
        candidates["ticker"]
        .astype("string")
        .str.zfill(6)
    )

    catalog["ticker"] = (
        catalog["ticker"]
        .astype("string")
        .str.zfill(6)
    )

    candidates = normalise_selected_column(candidates)
    candidates = add_bond_candidates(candidates, catalog)

    # 重置所有选择，由脚本统一重新决策
    candidates["selected"] = False
    candidates["review_status"] = "PENDING"
    candidates["review_notes"] = ""

    used_tickers: Set[str] = set()

    selection_order = EXPOSURE_PRIORITY + ["Bond"]

    for exposure in selection_order:
        group = candidates.loc[
            candidates["exposure"] == exposure
        ]

        if group.empty:
            print(f"[WARNING] No candidates for {exposure}")
            continue

        try:
            selected_index = select_best_candidate(
                candidates=group,
                used_tickers=used_tickers,
            )

        except ValueError as exc:
            print(f"[WARNING] {exc}")
            continue

        ticker = str(
            candidates.loc[selected_index, "ticker"]
        ).zfill(6)

        candidates.loc[selected_index, "selected"] = True
        candidates.loc[
            selected_index,
            "review_status",
        ] = "APPROVED"

        candidates.loc[
            selected_index,
            "review_notes",
        ] = "Automatically selected after duplicate and exposure checks"

        used_tickers.add(ticker)

    # 标记被排除的跨境产品
    cross_border_mask = candidates["name"].apply(
        contains_excluded_keyword
    )

    not_selected_mask = ~candidates["selected"]

    candidates.loc[
        cross_border_mask & not_selected_mask,
        "review_status",
    ] = "REJECTED"

    candidates.loc[
        cross_border_mask & not_selected_mask,
        "review_notes",
    ] = "Cross-border exposure excluded from domestic ETF universe"

    candidates.to_csv(
        CANDIDATE_PATH,
        index=False,
    )

    selected = candidates.loc[
        candidates["selected"]
    ].copy()

    print("")
    print("Final selected ETFs")
    print("=" * 90)

    print(
        selected[
            [
                "exposure",
                "primary_theme",
                "ticker",
                "name",
                "turnover",
            ]
        ].to_string(index=False)
    )

    duplicate_count = int(
        selected["ticker"].duplicated().sum()
    )

    print("")
    print(f"Selected count: {len(selected)}")
    print(f"Unique tickers: {selected['ticker'].nunique()}")
    print(f"Duplicate tickers: {duplicate_count}")

    if duplicate_count > 0:
        raise RuntimeError(
            "Duplicate tickers remain after finalisation."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

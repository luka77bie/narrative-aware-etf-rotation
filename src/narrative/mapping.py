from typing import List

import pandas as pd


def expand_theme_ticker_mapping(
    themes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand pipe-separated mapped_tickers into one row per ETF.
    """
    required_columns = {
        "theme_id",
        "theme_name",
        "mapped_tickers",
    }

    missing = required_columns - set(themes.columns)

    if missing:
        raise ValueError(
            "Theme mapping is missing columns: "
            + ", ".join(sorted(missing))
        )

    rows: List[dict] = []

    for row in themes.itertuples(index=False):
        tickers = str(row.mapped_tickers).split("|")

        for ticker in tickers:
            ticker = ticker.strip()

            if not ticker:
                continue

            rows.append(
                {
                    "theme_id": row.theme_id,
                    "theme_name": row.theme_name,
                    "ticker": ticker.zfill(6),
                }
            )

    mapping = pd.DataFrame(rows)

    if mapping.empty:
        raise ValueError(
            "No valid theme-to-ETF mappings were found."
        )

    return (
        mapping.drop_duplicates(
            subset=["theme_id", "ticker"]
        )
        .reset_index(drop=True)
    )


def map_narrative_scores_to_etfs(
    narrative_scores: pd.DataFrame,
    theme_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map theme-level Narrative Scores to ETF-level signals.

    If one ETF maps to multiple themes, use the mean score.
    """
    required_score_columns = {
        "date",
        "theme_id",
        "narrative_score",
    }

    missing = (
        required_score_columns
        - set(narrative_scores.columns)
    )

    if missing:
        raise ValueError(
            "Narrative scores are missing columns: "
            + ", ".join(sorted(missing))
        )

    merged = narrative_scores.merge(
        theme_mapping,
        on="theme_id",
        how="inner",
        validate="many_to_many",
    )

    if merged.empty:
        raise ValueError(
            "Narrative scores could not be mapped to ETFs."
        )

    etf_scores = (
        merged.groupby(
            ["date", "ticker"],
            as_index=False,
        )
        .agg(
            narrative_score=(
                "narrative_score",
                "mean",
            ),
            narrative_theme_count=(
                "theme_id",
                "nunique",
            ),
        )
    )

    return etf_scores

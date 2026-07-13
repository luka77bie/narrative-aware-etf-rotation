from typing import List

import pandas as pd


def select_top_n_equal_weight(
    ranking: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Select the top-N ETFs and assign equal weights.
    """
    required_columns = {
        "ticker",
        "momentum_score",
    }

    missing = required_columns - set(ranking.columns)

    if missing:
        raise ValueError(
            "Ranking is missing columns: "
            + ", ".join(sorted(missing))
        )

    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    eligible = ranking.dropna(
        subset=["momentum_score"]
    ).copy()

    if eligible.empty:
        return pd.DataFrame(
            columns=["ticker", "weight"]
        )

    selected = (
        eligible.sort_values(
            ["momentum_score", "ticker"],
            ascending=[False, True],
        )
        .head(top_n)
        .copy()
    )

    selected["weight"] = 1.0 / len(selected)

    return selected[
        [
            "ticker",
            "weight",
        ]
    ].reset_index(drop=True)


def weights_to_dict(
    weights: pd.DataFrame,
) -> dict:
    """Convert ticker/weight DataFrame into a dictionary."""
    if weights.empty:
        return {}

    return dict(
        zip(
            weights["ticker"],
            weights["weight"],
        )
    )


def calculate_turnover(
    old_weights: dict,
    new_weights: dict,
) -> float:
    """
    One-way portfolio turnover.

    Turnover = 0.5 * sum(abs(new_weight - old_weight))
    """
    tickers: List[str] = sorted(
        set(old_weights) | set(new_weights)
    )

    return 0.5 * sum(
        abs(
            new_weights.get(ticker, 0.0)
            - old_weights.get(ticker, 0.0)
        )
        for ticker in tickers
    )

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


def select_top_n_with_cash_filter(
    ranking: pd.DataFrame,
    top_n: int = 3,
    cash_ticker: str = "159001",
    absolute_momentum_column: str = "mom_60",
    score_column: str = "momentum_score",
) -> pd.DataFrame:
    """
    Select positive-momentum ETFs and allocate unused slots to cash.

    Each of the top_n slots receives 1 / top_n weight.
    If fewer than top_n ETFs have positive absolute momentum,
    the remaining weight is assigned to the cash ETF.
    """
    required_columns = {
        "ticker",
        absolute_momentum_column,
        score_column,
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
        subset=[
            absolute_momentum_column,
            score_column,
        ]
    ).copy()

    eligible = eligible.loc[
        eligible[absolute_momentum_column] > 0
    ]

    selected = (
        eligible.sort_values(
            [score_column, "ticker"],
            ascending=[False, True],
        )
        .head(top_n)
        .copy()
    )

    slot_weight = 1.0 / top_n

    rows = [
        {
            "ticker": str(ticker),
            "weight": slot_weight,
        }
        for ticker in selected["ticker"]
    ]

    unused_slots = top_n - len(selected)

    if unused_slots > 0:
        rows.append(
            {
                "ticker": str(cash_ticker),
                "weight": unused_slots * slot_weight,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["ticker", "weight"],
    )

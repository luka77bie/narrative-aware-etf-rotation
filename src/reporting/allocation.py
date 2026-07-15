from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


REQUIRED_RANKING_COLUMNS = {
    "date",
    "momentum_rank",
    "ticker",
    "name",
    "mom_60_pct",
    "momentum_score",
}


def require_columns(
    data: pd.DataFrame,
    required: Iterable[str],
    dataset_name: str,
) -> None:
    missing = set(required) - set(data.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            + ", ".join(sorted(missing))
        )


def normalise_ticker(
    series: pd.Series,
) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.zfill(6)
    )


def prepare_latest_ranking(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        ranking,
        REQUIRED_RANKING_COLUMNS,
        "Latest momentum ranking",
    )

    frame = ranking.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame["ticker"] = normalise_ticker(
        frame["ticker"]
    )

    numeric_columns = [
        "momentum_rank",
        "mom_60_pct",
        "momentum_score",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if frame[
        [
            "date",
            "ticker",
            *numeric_columns,
        ]
    ].isna().any().any():
        raise ValueError(
            "Latest momentum ranking contains invalid values."
        )

    if frame.duplicated(
        subset=["date", "ticker"]
    ).any():
        raise ValueError(
            "Latest momentum ranking contains duplicate "
            "date/ticker rows."
        )

    latest_date = frame["date"].max()

    frame = frame.loc[
        frame["date"] == latest_date
    ].copy()

    frame = frame.sort_values(
        [
            "momentum_rank",
            "ticker",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(drop=True)

    return frame


def prepare_signal_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "date",
        "ticker",
        "name",
        "mom_60",
        "momentum_score",
    }

    require_columns(
        history,
        required,
        "Momentum signal history",
    )

    frame = history.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame["ticker"] = normalise_ticker(
        frame["ticker"]
    )

    for column in [
        "mom_60",
        "momentum_score",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna(
        subset=[
            "date",
            "ticker",
            "mom_60",
            "momentum_score",
        ]
    )

    if frame.empty:
        raise ValueError(
            "Momentum signal history contains no valid rows."
        )

    return frame


def select_top_n(
    ranking: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError(
            "top_n must be positive."
        )

    frame = prepare_latest_ranking(
        ranking
    )

    if len(frame) < top_n:
        raise ValueError(
            f"Only {len(frame)} eligible ETFs are available; "
            f"{top_n} are required."
        )

    selected = frame.head(top_n).copy()

    selected["target_weight"] = (
        1.0 / top_n
    )

    return selected


def select_previous_top_n(
    history: pd.DataFrame,
    current_signal_date: pd.Timestamp,
    top_n: int = 3,
) -> pd.DataFrame:
    frame = prepare_signal_history(
        history
    )

    historical = frame.loc[
        frame["date"] < current_signal_date
    ].copy()

    if historical.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "name",
                "mom_60",
                "momentum_score",
                "momentum_rank",
                "target_weight",
            ]
        )

    valid = historical.dropna(
        subset=[
            "mom_60",
            "momentum_score",
        ]
    )

    coverage = (
        valid.groupby("date")["ticker"]
        .nunique()
        .sort_index()
    )

    eligible_dates = coverage.loc[
        coverage >= top_n
    ]

    if eligible_dates.empty:
        return pd.DataFrame()

    previous_date = (
        eligible_dates.index.max()
    )

    previous = valid.loc[
        valid["date"] == previous_date
    ].copy()

    previous = previous.sort_values(
        [
            "momentum_score",
            "ticker",
        ],
        ascending=[
            False,
            True,
        ],
    ).head(top_n)

    previous["momentum_rank"] = range(
        1,
        len(previous) + 1,
    )

    previous["target_weight"] = (
        1.0 / top_n
    )

    return previous.reset_index(
        drop=True
    )


def compare_allocations(
    current: pd.DataFrame,
    previous: pd.DataFrame,
) -> Dict[str, object]:
    current_tickers = set(
        current["ticker"]
    )

    previous_tickers = set(
        previous["ticker"]
    )

    added = sorted(
        current_tickers - previous_tickers
    )

    removed = sorted(
        previous_tickers - current_tickers
    )

    retained = sorted(
        current_tickers & previous_tickers
    )

    all_tickers = sorted(
        current_tickers | previous_tickers
    )

    current_weights = (
        current.set_index("ticker")[
            "target_weight"
        ].to_dict()
    )

    previous_weights = (
        previous.set_index("ticker")[
            "target_weight"
        ].to_dict()
        if not previous.empty
        else {}
    )

    one_way_turnover = (
        sum(
            abs(
                current_weights.get(
                    ticker,
                    0.0,
                )
                - previous_weights.get(
                    ticker,
                    0.0,
                )
            )
            for ticker in all_tickers
        )
        / 2.0
    )

    return {
        "added": added,
        "removed": removed,
        "retained": retained,
        "estimated_turnover": (
            one_way_turnover
        ),
    }


def calculate_data_age_days(
    signal_date: pd.Timestamp,
    reference_date: Optional[pd.Timestamp] = None,
) -> int:
    if reference_date is None:
        reference_date = pd.Timestamp.now(
            tz=None
        ).normalize()

    signal = pd.Timestamp(
        signal_date
    ).tz_localize(None).normalize()

    reference = pd.Timestamp(
        reference_date
    ).tz_localize(None).normalize()

    return int(
        (reference - signal).days
    )


def build_allocation_snapshot(
    ranking: pd.DataFrame,
    history: pd.DataFrame,
    top_n: int = 3,
    stale_after_days: int = 7,
    reference_date: Optional[pd.Timestamp] = None,
) -> Dict[str, object]:
    current = select_top_n(
        ranking,
        top_n=top_n,
    )

    signal_date = pd.Timestamp(
        current["date"].iloc[0]
    )

    previous = select_previous_top_n(
        history=history,
        current_signal_date=signal_date,
        top_n=top_n,
    )

    comparison = compare_allocations(
        current=current,
        previous=previous,
    )

    age_days = calculate_data_age_days(
        signal_date=signal_date,
        reference_date=reference_date,
    )

    snapshot = {
        "signal_date": signal_date,
        "data_age_days": age_days,
        "is_stale": (
            age_days > stale_after_days
        ),
        "execution_policy": (
            "Execute on the next available trading day "
            "after the signal date."
        ),
        "current": current,
        "previous": previous,
        **comparison,
    }

    return snapshot


def ticker_list_text(
    tickers: Iterable[str],
) -> str:
    values = list(tickers)

    if not values:
        return "None"

    return ", ".join(values)


def render_markdown(
    snapshot: Dict[str, object],
) -> str:
    current = snapshot["current"]
    previous = snapshot["previous"]

    signal_date = pd.Timestamp(
        snapshot["signal_date"]
    ).date()

    stale_status = (
        "WARNING: data may be stale"
        if snapshot["is_stale"]
        else "PASS"
    )

    lines = [
        "# Current MOM60 Allocation",
        "",
        f"- Data as of: **{signal_date}**",
        (
            f"- Data age: "
            f"**{snapshot['data_age_days']} calendar days**"
        ),
        f"- Freshness check: **{stale_status}**",
        (
            "- Execution policy: "
            f"**{snapshot['execution_policy']}**"
        ),
        "",
        "## Current Target Allocation",
        "",
        (
            "| Rank | Ticker | ETF Name | Theme | "
            "MOM60 | Score | Target Weight |"
        ),
        (
            "|---:|---|---|---|---:|---:|---:|"
        ),
    ]

    for row in current.itertuples(
        index=False
    ):
        theme = getattr(
            row,
            "secondary_theme",
            "",
        )

        lines.append(
            "| "
            f"{int(row.momentum_rank)} | "
            f"{row.ticker} | "
            f"{row.name} | "
            f"{theme} | "
            f"{row.mom_60_pct:.2f}% | "
            f"{row.momentum_score:.3f} | "
            f"{row.target_weight * 100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Rebalance Summary",
            "",
            (
                f"- Added: "
                f"**{ticker_list_text(snapshot['added'])}**"
            ),
            (
                f"- Removed: "
                f"**{ticker_list_text(snapshot['removed'])}**"
            ),
            (
                f"- Retained: "
                f"**{ticker_list_text(snapshot['retained'])}**"
            ),
            (
                "- Estimated one-way turnover: "
                f"**{snapshot['estimated_turnover'] * 100:.2f}%**"
            ),
            "",
            "## Previous Target Allocation",
            "",
        ]
    )

    if previous.empty:
        lines.append(
            "No prior eligible allocation was found."
        )
    else:
        previous_date = pd.Timestamp(
            previous["date"].iloc[0]
        ).date()

        lines.extend(
            [
                f"Previous signal date: **{previous_date}**",
                "",
                (
                    "| Rank | Ticker | ETF Name | "
                    "MOM60 | Score | Weight |"
                ),
                "|---:|---|---|---:|---:|---:|",
            ]
        )

        for row in previous.itertuples(
            index=False
        ):
            lines.append(
                "| "
                f"{int(row.momentum_rank)} | "
                f"{row.ticker} | "
                f"{row.name} | "
                f"{row.mom_60 * 100:.2f}% | "
                f"{row.momentum_score:.3f} | "
                f"{row.target_weight * 100:.2f}% |"
            )

    lines.extend(
        [
            "",
            "## Important Notes",
            "",
            (
                "- This is a model-generated signal, "
                "not an automatic trade instruction."
            ),
            (
                "- Confirm the latest market-data date "
                "before using the allocation."
            ),
            (
                "- Check suspension, price limits, "
                "liquidity, fees and slippage before execution."
            ),
            (
                "- The strategy uses next-trading-day execution "
                "to avoid same-day look-ahead bias."
            ),
        ]
    )

    return "\n".join(lines) + "\n"


def render_html(
    markdown_text: str,
) -> str:
    import markdown

    body = markdown.markdown(
        markdown_text,
        extensions=[
            "tables",
            "fenced_code",
        ],
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>
<title>Current MOM60 Allocation</title>
<style>
body {{
    max-width: 1050px;
    margin: 40px auto;
    padding: 0 24px;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
    line-height: 1.55;
    color: #202124;
}}
h1, h2 {{
    line-height: 1.25;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 18px 0 30px;
}}
th, td {{
    border: 1px solid #d0d7de;
    padding: 9px 11px;
}}
th {{
    background: #f6f8fa;
}}
th:not(:nth-child(2)):not(:nth-child(3)):not(:nth-child(4)),
td:not(:nth-child(2)):not(:nth-child(3)):not(:nth-child(4)) {{
    text-align: right;
}}
li {{
    margin: 6px 0;
}}
</style>
</head>
<body>
{body}
</body>
</html>
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.portfolio.construction import (
    calculate_turnover,
    select_top_n_equal_weight,
    weights_to_dict,
)


def build_price_matrix(
    prices: pd.DataFrame,
    price_column: str = "adjusted_close",
) -> pd.DataFrame:
    """Convert long-format ETF prices into a wide price matrix."""
    required_columns = {
        "date",
        "ticker",
        price_column,
    }

    missing = required_columns - set(prices.columns)

    if missing:
        raise ValueError(
            "Price data is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = prices.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    if frame["date"].isna().any():
        raise ValueError("Price data contains invalid dates.")

    matrix = (
        frame.pivot_table(
            index="date",
            columns="ticker",
            values=price_column,
            aggfunc="last",
        )
        .sort_index()
    )

    return matrix


def get_month_end_signal_dates(
    dates: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """
    Return the last available trading date in each calendar month.
    """
    date_series = pd.Series(
        dates,
        index=dates,
    )

    month_end_dates = (
        date_series.groupby(
            dates.to_period("M")
        )
        .max()
    )

    return pd.DatetimeIndex(
        month_end_dates.values
    )


def get_next_trading_date(
    dates: pd.DatetimeIndex,
    signal_date: pd.Timestamp,
) -> pd.Timestamp:
    """
    Return the first available trading date after signal_date.
    """
    future_dates = dates[
        dates > signal_date
    ]

    if len(future_dates) == 0:
        return pd.NaT

    return future_dates[0]


def calculate_performance_metrics(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Calculate core performance statistics."""
    returns = returns.dropna()

    if returns.empty:
        raise ValueError(
            "Cannot calculate metrics from empty returns."
        )

    equity_curve = (1.0 + returns).cumprod()

    total_return = equity_curve.iloc[-1] - 1.0

    years = len(returns) / periods_per_year

    cagr = (
        equity_curve.iloc[-1] ** (1.0 / years) - 1.0
        if years > 0
        else np.nan
    )

    volatility = (
        returns.std(ddof=0)
        * np.sqrt(periods_per_year)
    )

    sharpe = (
        returns.mean()
        / returns.std(ddof=0)
        * np.sqrt(periods_per_year)
        if not np.isclose(
            returns.std(ddof=0),
            0.0,
        )
        else np.nan
    )

    downside_returns = returns.loc[
        returns < 0
    ]

    downside_volatility = (
        downside_returns.std(ddof=0)
        * np.sqrt(periods_per_year)
    )

    sortino = (
        returns.mean()
        / downside_returns.std(ddof=0)
        * np.sqrt(periods_per_year)
        if (
            not downside_returns.empty
            and not np.isclose(
                downside_returns.std(ddof=0),
                0.0,
            )
        )
        else np.nan
    )

    running_max = equity_curve.cummax()

    drawdown = (
        equity_curve / running_max - 1.0
    )

    maximum_drawdown = drawdown.min()

    calmar = (
        cagr / abs(maximum_drawdown)
        if maximum_drawdown < 0
        else np.nan
    )

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "maximum_drawdown": maximum_drawdown,
        "calmar": calmar,
    }


def run_monthly_top_n_backtest(
    prices: pd.DataFrame,
    scored_signals: pd.DataFrame,
    top_n: int = 3,
    transaction_cost_rate: float = 0.001,
) -> Dict[str, pd.DataFrame]:
    """
    Run a monthly Top-N equal-weight momentum backtest.

    Signal timing:
    - Signal generated at month-end close.
    - New weights become active on the next trading day.
    - Returns are calculated using prior-day weights.
    """
    price_matrix = build_price_matrix(
        prices=prices,
    )

    daily_returns = (
        price_matrix.pct_change(
            fill_method=None
        )
    )

    trading_dates = price_matrix.index

    signal_dates = get_month_end_signal_dates(
        trading_dates
    )

    target_weights_by_execution_date = {}
    rebalance_rows: List[dict] = []

    previous_target_weights = {}

    for signal_date in signal_dates:
        execution_date = get_next_trading_date(
            trading_dates,
            signal_date,
        )

        if pd.isna(execution_date):
            continue

        cross_section = scored_signals.loc[
            scored_signals["date"]
            == signal_date
        ].copy()

        cross_section = cross_section.dropna(
            subset=[
                "mom_20",
                "mom_60",
                "momentum_score",
            ]
        )

        if cross_section.empty:
            continue

        selected = select_top_n_equal_weight(
            ranking=cross_section,
            top_n=top_n,
        )

        new_target_weights = weights_to_dict(
            selected
        )

        turnover = calculate_turnover(
            old_weights=previous_target_weights,
            new_weights=new_target_weights,
        )

        target_weights_by_execution_date[
            execution_date
        ] = new_target_weights

        rebalance_rows.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "selected_tickers": ",".join(
                    selected["ticker"].tolist()
                ),
                "turnover": turnover,
                "transaction_cost": (
                    turnover
                    * transaction_cost_rate
                ),
            }
        )

        previous_target_weights = (
            new_target_weights
        )

    current_weights = {}
    portfolio_returns = []
    holdings_rows = []

    for date in trading_dates:
        transaction_cost = 0.0

        if date in target_weights_by_execution_date:
            new_weights = (
                target_weights_by_execution_date[
                    date
                ]
            )

            turnover = calculate_turnover(
                old_weights=current_weights,
                new_weights=new_weights,
            )

            transaction_cost = (
                turnover
                * transaction_cost_rate
            )

            current_weights = new_weights

        daily_return = 0.0

        for ticker, weight in current_weights.items():
            asset_return = daily_returns.loc[
                date,
                ticker,
            ]

            if pd.notna(asset_return):
                daily_return += (
                    weight * asset_return
                )

        net_return = (
            daily_return - transaction_cost
        )

        portfolio_returns.append(
            {
                "date": date,
                "gross_return": daily_return,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
            }
        )

        for ticker, weight in current_weights.items():
            holdings_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "weight": weight,
                }
            )

    returns = pd.DataFrame(
        portfolio_returns
    ).set_index("date")

    returns["gross_nav"] = (
        1.0 + returns["gross_return"]
    ).cumprod()

    returns["net_nav"] = (
        1.0 + returns["net_return"]
    ).cumprod()

    holdings = pd.DataFrame(
        holdings_rows
    )

    rebalances = pd.DataFrame(
        rebalance_rows
    )

    metrics = calculate_performance_metrics(
        returns["net_return"]
    )

    metrics_frame = pd.DataFrame(
        [
            {
                **metrics,
                "top_n": top_n,
                "transaction_cost_rate": (
                    transaction_cost_rate
                ),
                "rebalance_count": len(
                    rebalances
                ),
                "average_turnover": (
                    rebalances["turnover"].mean()
                    if not rebalances.empty
                    else 0.0
                ),
            }
        ]
    )

    return {
        "returns": returns,
        "holdings": holdings,
        "rebalances": rebalances,
        "metrics": metrics_frame,
    }

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_walk_forward_windows(
    start_date: str,
    end_date: str,
    train_months: int = 36,
    test_months: int = 12,
    step_months: int = 12,
) -> List[WalkForwardWindow]:
    """
    Generate non-overlapping walk-forward test windows.

    Training windows may overlap. Test windows advance by step_months.
    """
    if train_months <= 0:
        raise ValueError(
            "train_months must be positive."
        )

    if test_months <= 0:
        raise ValueError(
            "test_months must be positive."
        )

    if step_months <= 0:
        raise ValueError(
            "step_months must be positive."
        )

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if end <= start:
        raise ValueError(
            "end_date must be later than start_date."
        )

    windows = []
    fold = 1

    train_start = start

    while True:
        train_end = (
            train_start
            + pd.DateOffset(months=train_months)
            - pd.Timedelta(days=1)
        )

        test_start = train_end + pd.Timedelta(days=1)

        test_end = (
            test_start
            + pd.DateOffset(months=test_months)
            - pd.Timedelta(days=1)
        )

        if test_start > end:
            break

        clipped_test_end = min(
            test_end,
            end,
        )

        windows.append(
            WalkForwardWindow(
                fold=fold,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=clipped_test_end,
            )
        )

        fold += 1

        train_start = (
            train_start
            + pd.DateOffset(months=step_months)
        )

        if clipped_test_end >= end:
            break

    return windows


def filter_window(
    data: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    date_column: str = "date",
) -> pd.DataFrame:
    """Return rows inside an inclusive date window."""
    if date_column not in data.columns:
        raise ValueError(
            f"Data is missing date column: {date_column}"
        )

    frame = data.copy()

    frame[date_column] = pd.to_datetime(
        frame[date_column],
        errors="coerce",
    )

    if frame[date_column].isna().any():
        raise ValueError(
            "Data contains invalid dates."
        )

    return frame.loc[
        (frame[date_column] >= start_date)
        & (frame[date_column] <= end_date)
    ].copy()

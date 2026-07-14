import pandas as pd
import pytest

from src.evaluation.walk_forward import (
    filter_window,
    generate_walk_forward_windows,
)


def test_generate_walk_forward_windows() -> None:
    windows = generate_walk_forward_windows(
        start_date="2019-10-08",
        end_date="2026-07-13",
        train_months=36,
        test_months=12,
        step_months=12,
    )

    assert len(windows) >= 3

    first = windows[0]

    assert first.train_start == pd.Timestamp(
        "2019-10-08"
    )

    assert first.test_start > first.train_end


def test_test_windows_move_forward() -> None:
    windows = generate_walk_forward_windows(
        start_date="2019-10-08",
        end_date="2026-07-13",
    )

    for previous, current in zip(
        windows[:-1],
        windows[1:],
    ):
        assert (
            current.test_start
            > previous.test_start
        )


def test_invalid_window_configuration_fails() -> None:
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        generate_walk_forward_windows(
            start_date="2020-01-01",
            end_date="2024-01-01",
            train_months=0,
        )


def test_filter_window() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                ]
            ),
            "value": [1, 2, 3],
        }
    )

    result = filter_window(
        data=data,
        start_date=pd.Timestamp(
            "2024-02-01"
        ),
        end_date=pd.Timestamp(
            "2024-03-01"
        ),
    )

    assert len(result) == 2
    assert result["value"].tolist() == [2, 3]

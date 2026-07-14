from pathlib import Path

import pandas as pd


PROFILE_PATH = Path(
    "config/narrative_profiles.csv"
)


def test_selected_formal_profile_excludes_policy() -> None:
    profiles = pd.read_csv(PROFILE_PATH)

    selected_formal = profiles.loc[
        (profiles["status"] == "selected")
        & (profiles["research_use"] == "formal")
    ]

    assert not selected_formal.empty
    assert (
        selected_formal["policy_weight"] == 0.0
    ).all()


def test_validation_only_profiles_are_not_formal() -> None:
    profiles = pd.read_csv(PROFILE_PATH)

    validation_only = profiles.loc[
        profiles["status"] == "validation_only"
    ]

    assert not validation_only.empty
    assert (
        validation_only["research_use"]
        != "formal"
    ).all()


def test_weights_sum_to_one() -> None:
    profiles = pd.read_csv(PROFILE_PATH)

    total = (
        profiles["momentum_weight"]
        + profiles["proxy_weight"]
        + profiles["policy_weight"]
    )

    assert total.round(10).eq(1.0).all()

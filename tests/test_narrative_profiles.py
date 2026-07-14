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


def test_oos_rejected_proxy_is_not_selected() -> None:
    profiles = pd.read_csv(PROFILE_PATH)

    proxy = profiles.loc[
        profiles["profile"]
        == "market_attention_50pct"
    ].iloc[0]

    assert proxy["status"] != "selected"
    assert proxy["policy_weight"] == 0.0

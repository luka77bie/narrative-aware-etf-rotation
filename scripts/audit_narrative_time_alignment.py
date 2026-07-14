import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.narrative.policy_archive import (
    load_policy_archive,
)
from src.narrative.time_alignment import (
    audit_policy_point_in_time,
    build_alignment_summary,
)


MOMENTUM_PATH = Path(
    "outputs/momentum_signal_history.csv"
)

PROXY_PATH = Path(
    "outputs/narrative_proxy_signal_history.csv"
)

POLICY_SIGNAL_PATH = Path(
    "outputs/policy_narrative_signal_history.csv"
)

POLICY_METADATA_PATH = Path(
    "data/raw/policy/"
    "manual_state_council_gazette.csv"
)

NARRATIVE_V2_PATH = Path(
    "outputs/narrative_v2_signal_history.csv"
)

OUTPUT_DIRECTORY = Path("outputs")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required audit input is missing: {path}"
        )


def main() -> int:
    for path in [
        MOMENTUM_PATH,
        PROXY_PATH,
        POLICY_SIGNAL_PATH,
        POLICY_METADATA_PATH,
        NARRATIVE_V2_PATH,
    ]:
        require_file(path)

    momentum = pd.read_csv(
        MOMENTUM_PATH,
        dtype={"ticker": "string"},
        parse_dates=["date"],
    )

    proxy = pd.read_csv(
        PROXY_PATH,
        dtype={"ticker": "string"},
        parse_dates=["date"],
    )

    policy_signal = pd.read_csv(
        POLICY_SIGNAL_PATH,
        dtype={"theme_id": "string"},
        parse_dates=["date"],
    )

    narrative_v2 = pd.read_csv(
        NARRATIVE_V2_PATH,
        dtype={"ticker": "string"},
        parse_dates=["date"],
    )

    policy_metadata = load_policy_archive(
        POLICY_METADATA_PATH
    )

    # A manually collected document becomes available only after
    # both publication and retrieval have occurred.
    published_at = pd.to_datetime(
        policy_metadata["published_at"],
        errors="coerce",
        utc=True,
    )

    retrieved_at = pd.to_datetime(
        policy_metadata["retrieved_at"],
        errors="coerce",
        utc=True,
    )

    if published_at.isna().any():
        raise ValueError(
            "Policy metadata contains invalid published_at values."
        )

    if retrieved_at.isna().any():
        raise ValueError(
            "Policy metadata contains invalid retrieved_at values."
        )

    policy_metadata["available_at"] = pd.concat(
        [
            published_at.rename("published_at"),
            retrieved_at.rename("retrieved_at"),
        ],
        axis=1,
    ).max(axis=1)

    available_shanghai = (
        policy_metadata["available_at"]
        .dt.tz_convert("Asia/Shanghai")
    )

    local_day = available_shanghai.dt.normalize()

    market_close = (
        local_day
        + pd.Timedelta(hours=15)
    )

    after_market_close = (
        available_shanghai > market_close
    )

    effective_day = local_day.where(
        ~after_market_close,
        local_day + pd.Timedelta(days=1),
    )

    policy_metadata["date"] = (
        effective_day
        .dt.tz_localize(None)
    )

    policy_pit = audit_policy_point_in_time(
        policy_metadata,
        signal_date_column="date",
        publication_column="published_at",
        retrieval_column="retrieved_at",
        market_close_hour=15,
    )

    summary = build_alignment_summary(
        momentum=momentum,
        proxy=proxy,
        policy=policy_signal,
        narrative_v2=narrative_v2,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIRECTORY
        / "narrative_time_alignment_summary.csv"
    )

    policy_path = (
        OUTPUT_DIRECTORY
        / "policy_point_in_time_audit.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    policy_pit.to_csv(
        policy_path,
        index=False,
    )

    duplicate_failures = summary.loc[
        summary["has_duplicates"]
    ]

    invalid_policy = policy_pit.loc[
        ~policy_pit["point_in_time_valid"]
    ]

    print("Narrative Time Alignment Audit")
    print("=" * 110)

    print("")
    print("Dataset Summary")
    print(
        summary.to_string(index=False)
    )

    print("")
    print("Policy Point-in-Time Audit")
    print(
        policy_pit[
            [
                "document_id",
                "date",
                "published_at",
                "retrieved_at",
                "available_at",
                "signal_cutoff_at",
                "available_in_time",
                "point_in_time_valid",
            ]
        ].to_string(index=False)
    )

    print("")
    print(
        "Duplicate dataset failures:",
        len(duplicate_failures),
    )

    print(
        "Invalid policy timing rows:",
        len(invalid_policy),
    )

    print(f"Summary output: {summary_path}")
    print(f"Policy audit output: {policy_path}")

    if not duplicate_failures.empty:
        raise RuntimeError(
            "Time alignment audit failed: "
            "duplicate signal keys found."
        )

    if not invalid_policy.empty:
        raise RuntimeError(
            "Time alignment audit failed: "
            "policy records were not available "
            "at their assumed signal dates."
        )

    print("")
    print("Audit status: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

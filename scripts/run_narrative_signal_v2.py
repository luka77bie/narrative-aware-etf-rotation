import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.narrative.composite import (
    combine_narrative_components,
    map_policy_scores_to_etfs,
)
from src.narrative.mapping import (
    expand_theme_ticker_mapping,
)


PROXY_PATH = Path(
    "outputs/narrative_proxy_signal_history.csv"
)

POLICY_PATH = Path(
    "outputs/policy_narrative_signal_history.csv"
)

THEME_PATH = Path(
    "config/narrative_themes.csv"
)

OUTPUT_DIRECTORY = Path("outputs")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine market-attention proxy and policy "
            "signals into Narrative Signal V2."
        )
    )

    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Required while policy sources remain "
            "unapproved for formal research."
        ),
    )

    parser.add_argument(
        "--proxy-weight",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--policy-weight",
        type=float,
        default=0.30,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if not args.validation_only:
        raise RuntimeError(
            "Narrative V2 includes an unapproved policy "
            "source. Use --validation-only."
        )

    if not PROXY_PATH.exists():
        raise FileNotFoundError(
            "Narrative Proxy history is missing. Run:\n"
            "python3 scripts/run_narrative_proxy_signal.py"
        )

    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            "Policy Narrative history is missing. Run:\n"
            "python3 scripts/run_policy_signal_v1.py "
            "--validation-only"
        )

    proxy = pd.read_csv(
        PROXY_PATH,
        dtype={"ticker": "string"},
        parse_dates=["date"],
    )

    policy = pd.read_csv(
        POLICY_PATH,
        dtype={"theme_id": "string"},
        parse_dates=["date"],
    )

    themes = pd.read_csv(
        THEME_PATH,
        dtype={
            "theme_id": "string",
            "mapped_tickers": "string",
        },
    )

    included = (
        themes["include"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    themes = themes.loc[included].copy()

    mapping = expand_theme_ticker_mapping(
        themes
    )

    policy_etf = map_policy_scores_to_etfs(
        policy_scores=policy,
        theme_mapping=mapping,
    )

    combined = combine_narrative_components(
        proxy_scores=proxy,
        policy_etf_scores=policy_etf,
        proxy_weight=args.proxy_weight,
        policy_weight=args.policy_weight,
        research_status=(
            "pipeline_validation_only"
        ),
    )

    complete = combined.dropna(
        subset=[
            "narrative_proxy_score",
            "narrative_v2_score",
            "narrative_v2_rank",
        ]
    ).copy()

    if complete.empty:
        raise ValueError(
            "No complete Narrative V2 observations."
        )

    latest_date = complete["date"].max()

    latest = (
        complete.loc[
            complete["date"] == latest_date
        ]
        .sort_values(
            "narrative_v2_rank"
        )
        .reset_index(drop=True)
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        OUTPUT_DIRECTORY
        / "narrative_v2_signal_history.csv"
    )

    latest_path = (
        OUTPUT_DIRECTORY
        / "latest_narrative_v2_ranking.csv"
    )

    policy_etf_path = (
        OUTPUT_DIRECTORY
        / "policy_etf_signal_history.csv"
    )

    policy_etf.to_csv(
        policy_etf_path,
        index=False,
    )

    combined.to_csv(
        history_path,
        index=False,
    )

    latest.to_csv(
        latest_path,
        index=False,
    )

    print("Narrative Signal V2")
    print("=" * 110)
    print(
        "Research status: "
        "PIPELINE VALIDATION ONLY"
    )
    print(
        f"Proxy weight: "
        f"{args.proxy_weight:.2f}"
    )
    print(
        f"Policy weight: "
        f"{args.policy_weight:.2f}"
    )
    print(
        f"Latest date: "
        f"{pd.Timestamp(latest_date).date()}"
    )

    print("")
    print("Top 10 Narrative V2 Ranking")
    print("=" * 110)

    display_columns = [
        "narrative_v2_rank",
        "ticker",
        "name",
        "secondary_theme",
        "narrative_proxy_score",
        "policy_narrative_score",
        "policy_theme_count",
        "narrative_v2_score",
        "research_status",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in latest.columns
    ]

    print(
        latest[
            available_columns
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    print("")
    print(f"Policy ETF signals: {policy_etf_path}")
    print(f"V2 history: {history_path}")
    print(f"Latest ranking: {latest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

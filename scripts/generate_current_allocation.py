import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.reporting.allocation import (
    build_allocation_snapshot,
    render_html,
    render_markdown,
)


RANKING_PATH = Path(
    "outputs/latest_momentum_ranking.csv"
)

HISTORY_PATH = Path(
    "outputs/momentum_signal_history.csv"
)

OUTPUT_DIRECTORY = Path(
    "outputs/reporting"
)

CSV_PATH = (
    OUTPUT_DIRECTORY
    / "current_allocation.csv"
)

MARKDOWN_PATH = (
    OUTPUT_DIRECTORY
    / "current_allocation.md"
)

HTML_PATH = (
    OUTPUT_DIRECTORY
    / "current_allocation.html"
)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"Required input is empty: {path}"
        )


def main() -> int:
    require_file(RANKING_PATH)
    require_file(HISTORY_PATH)

    ranking = pd.read_csv(
        RANKING_PATH,
        dtype={"ticker": "string"},
    )

    history = pd.read_csv(
        HISTORY_PATH,
        dtype={"ticker": "string"},
    )

    snapshot = build_allocation_snapshot(
        ranking=ranking,
        history=history,
        top_n=3,
        stale_after_days=7,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    current = snapshot[
        "current"
    ].copy()

    current["signal_date"] = (
        snapshot["signal_date"]
    )

    current["data_age_days"] = (
        snapshot["data_age_days"]
    )

    current["is_stale"] = (
        snapshot["is_stale"]
    )

    current["estimated_turnover"] = (
        snapshot["estimated_turnover"]
    )

    current.to_csv(
        CSV_PATH,
        index=False,
    )

    markdown_text = render_markdown(
        snapshot
    )

    MARKDOWN_PATH.write_text(
        markdown_text,
        encoding="utf-8",
    )

    HTML_PATH.write_text(
        render_html(markdown_text),
        encoding="utf-8",
    )

    display = current[
        [
            "momentum_rank",
            "ticker",
            "name",
            "mom_60_pct",
            "momentum_score",
            "target_weight",
        ]
    ].copy()

    display["target_weight"] *= 100

    print("")
    print("Current MOM60 Allocation")
    print("=" * 90)
    print(
        f"Data as of: "
        f"{pd.Timestamp(snapshot['signal_date']).date()}"
    )
    print(
        f"Data age: "
        f"{snapshot['data_age_days']} days"
    )
    print(
        f"Stale: {snapshot['is_stale']}"
    )
    print("")

    print(
        display.round(
            {
                "mom_60_pct": 2,
                "momentum_score": 3,
                "target_weight": 2,
            }
        ).to_string(index=False)
    )

    print("")
    print(
        "Added:",
        snapshot["added"],
    )
    print(
        "Removed:",
        snapshot["removed"],
    )
    print(
        "Retained:",
        snapshot["retained"],
    )
    print(
        "Estimated turnover:",
        f"{snapshot['estimated_turnover'] * 100:.2f}%",
    )
    print("")
    print(f"CSV: {CSV_PATH}")
    print(f"Markdown: {MARKDOWN_PATH}")
    print(f"HTML: {HTML_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path(
    "data/sample/narrative/narrative_features.csv"
)

THEMES = [
    "ai",
    "semiconductor",
    "robotics",
    "gold",
    "nonferrous",
    "coal",
    "dividend",
    "bank",
    "power",
    "healthcare",
    "consumer",
]


def main() -> int:
    rng = np.random.default_rng(42)

    dates = pd.date_range(
        "2024-01-01",
        periods=180,
        freq="D",
    )

    rows = []

    for theme_index, theme_id in enumerate(THEMES):
        base_news = 8 + theme_index
        base_attention = 80 + theme_index * 4

        for date_index, date in enumerate(dates):
            trend = date_index / len(dates)

            cyclical_component = (
                np.sin(
                    date_index / 12
                    + theme_index
                )
            )

            narrative_spike = 0

            if (
                theme_id == "ai"
                and 90 <= date_index <= 115
            ):
                narrative_spike = 15

            if (
                theme_id == "gold"
                and 125 <= date_index <= 145
            ):
                narrative_spike = 10

            news_count = max(
                0,
                int(
                    base_news
                    + 4 * cyclical_component
                    + narrative_spike
                    + rng.normal(0, 2)
                ),
            )

            policy_count = max(
                0,
                int(
                    rng.poisson(
                        0.5
                        + 0.2 * theme_index
                    )
                ),
            )

            attention_index = max(
                1.0,
                (
                    base_attention
                    + 10 * trend
                    + 6 * cyclical_component
                    + 2 * narrative_spike
                    + rng.normal(0, 2)
                ),
            )

            rows.append(
                {
                    "date": date.date(),
                    "theme_id": theme_id,
                    "news_count": news_count,
                    "policy_count": policy_count,
                    "attention_index": round(
                        attention_index,
                        3,
                    ),
                    "source": "synthetic_sample",
                }
            )

    data = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Rows: {len(data)}")
    print(f"Themes: {data['theme_id'].nunique()}")
    print(
        f"Date range: "
        f"{data['date'].min()} to "
        f"{data['date'].max()}"
    )
    print(f"Output: {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

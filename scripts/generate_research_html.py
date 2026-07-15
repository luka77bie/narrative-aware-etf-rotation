import sys
from pathlib import Path

import markdown


REPORT_DIRECTORY = Path(
    "outputs/reporting"
)

MARKDOWN_PATH = (
    REPORT_DIRECTORY / "research_report.md"
)

HTML_PATH = (
    REPORT_DIRECTORY / "research_report.html"
)


def main() -> int:
    if not MARKDOWN_PATH.exists():
        raise FileNotFoundError(
            f"Markdown report not found: "
            f"{MARKDOWN_PATH}"
        )

    markdown_text = MARKDOWN_PATH.read_text(
        encoding="utf-8"
    )

    html_body = markdown.markdown(
        markdown_text,
        extensions=[
            "tables",
            "fenced_code",
        ],
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>
<title>
Narrative-Aware ETF Rotation Research Report
</title>
<style>
body {{
    max-width: 1100px;
    margin: 40px auto;
    padding: 0 24px;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
    line-height: 1.6;
    color: #202124;
}}
h1, h2, h3 {{
    line-height: 1.25;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0 32px;
    font-size: 14px;
}}
th, td {{
    border: 1px solid #d0d7de;
    padding: 8px 10px;
    text-align: right;
}}
th {{
    background: #f6f8fa;
}}
th:first-child,
td:first-child {{
    text-align: left;
}}
img {{
    display: block;
    max-width: 100%;
    height: auto;
    margin: 18px auto 36px;
}}
code {{
    background: #f6f8fa;
    padding: 2px 5px;
    border-radius: 4px;
}}
blockquote {{
    border-left: 4px solid #d0d7de;
    margin-left: 0;
    padding-left: 16px;
    color: #57606a;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    HTML_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    HTML_PATH.write_text(
        html,
        encoding="utf-8",
    )

    print("Research HTML Report")
    print("=" * 80)
    print(f"Input: {MARKDOWN_PATH}")
    print(f"Output: {HTML_PATH}")
    print(
        f"Size: {HTML_PATH.stat().st_size:,} bytes"
    )
    print("Status: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

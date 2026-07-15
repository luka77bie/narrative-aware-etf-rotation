import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: List[str]
    required_outputs: List[Path]
    optional: bool = False


PIPELINE_STEPS = [
    PipelineStep(
        name="Run test suite",
        command=[
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ],
        required_outputs=[],
    ),
    PipelineStep(
        name="Build momentum signal",
        command=[
            sys.executable,
            "scripts/run_momentum_signal.py",
        ],
        required_outputs=[
            Path(
                "outputs/momentum_signal_history.csv"
            ),
        ],
    ),
    PipelineStep(
        name="Generate current allocation report",
        command=[
            sys.executable,
            "scripts/generate_current_allocation.py",
        ],
        required_outputs=[
            Path(
                "outputs/reporting/"
                "current_allocation.csv"
            ),
            Path(
                "outputs/reporting/"
                "current_allocation.md"
            ),
            Path(
                "outputs/reporting/"
                "current_allocation.html"
            ),
        ],
    ),
    PipelineStep(
        name="Build market-attention proxy",
        command=[
            sys.executable,
            "scripts/run_narrative_proxy_signal.py",
        ],
        required_outputs=[
            Path(
                "outputs/"
                "narrative_proxy_signal_history.csv"
            ),
        ],
    ),
    PipelineStep(
        name="Run proxy ablation",
        command=[
            sys.executable,
            "scripts/"
            "run_proxy_composite_ablation.py",
        ],
        required_outputs=[
            Path(
                "outputs/"
                "proxy_composite_ablation_metrics.csv"
            ),
        ],
    ),
    PipelineStep(
        name="Run proxy robustness analysis",
        command=[
            sys.executable,
            "scripts/run_proxy_robustness.py",
        ],
        required_outputs=[
            Path(
                "outputs/proxy_robustness_metrics.csv"
            ),
        ],
    ),
    PipelineStep(
        name="Run walk-forward validation",
        command=[
            sys.executable,
            "scripts/"
            "run_walk_forward_validation.py",
        ],
        required_outputs=[
            Path(
                "outputs/"
                "walk_forward_aggregate_metrics.csv"
            ),
            Path(
                "outputs/"
                "walk_forward_fold_metrics.csv"
            ),
            Path(
                "outputs/"
                "walk_forward_oos_returns.csv"
            ),
        ],
    ),
    PipelineStep(
        name="Generate research charts",
        command=[
            sys.executable,
            "scripts/generate_research_charts.py",
        ],
        required_outputs=[
            Path(
                "outputs/reporting/charts/"
                "nav_comparison.png"
            ),
            Path(
                "outputs/reporting/charts/"
                "drawdown_comparison.png"
            ),
            Path(
                "outputs/reporting/charts/"
                "walk_forward_sharpe.png"
            ),
            Path(
                "outputs/reporting/charts/"
                "subperiod_cagr.png"
            ),
        ],
    ),
    PipelineStep(
        name="Generate research report",
        command=[
            sys.executable,
            "scripts/generate_research_report.py",
        ],
        required_outputs=[
            Path(
                "outputs/reporting/"
                "research_report.md"
            ),
            Path(
                "outputs/reporting/model_summary.csv"
            ),
            Path(
                "outputs/reporting/"
                "robustness_summary.csv"
            ),
            Path(
                "outputs/reporting/"
                "walk_forward_summary.csv"
            ),
        ],
    ),
    PipelineStep(
        name="Generate HTML report",
        command=[
            sys.executable,
            "scripts/generate_research_html.py",
        ],
        required_outputs=[
            Path(
                "outputs/reporting/"
                "research_report.html"
            ),
        ],
    ),
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete narrative-aware ETF "
            "rotation research pipeline."
        )
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the pytest stage.",
    )

    parser.add_argument(
        "--from-step",
        default=None,
        help=(
            "Start from a named pipeline step. "
            "Use --list-steps to inspect names."
        ),
    )

    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List pipeline steps and exit.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue after a failed optional step."
        ),
    )

    return parser.parse_args()


def verify_outputs(
    step: PipelineStep,
) -> None:
    missing_outputs = [
        output
        for output in step.required_outputs
        if (
            not output.exists()
            or output.stat().st_size == 0
        )
    ]

    if missing_outputs:
        missing_text = "\n".join(
            f"- {path}"
            for path in missing_outputs
        )

        raise RuntimeError(
            f"Step completed but required outputs "
            f"are missing or empty:\n{missing_text}"
        )


def run_step(
    step: PipelineStep,
) -> float:
    print("")
    print("=" * 100)
    print(f"[START] {step.name}")
    print(
        "[COMMAND] "
        + " ".join(step.command)
    )
    print("=" * 100)

    start_time = time.perf_counter()

    process = subprocess.run(
        step.command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    elapsed = time.perf_counter() - start_time

    if process.returncode != 0:
        raise RuntimeError(
            f"Pipeline step failed: {step.name}. "
            f"Exit code: {process.returncode}"
        )

    verify_outputs(step)

    print(
        f"[PASS] {step.name} "
        f"({elapsed:.2f} seconds)"
    )

    return elapsed


def select_steps(
    skip_tests: bool,
    from_step: Optional[str],
) -> List[PipelineStep]:
    steps = PIPELINE_STEPS.copy()

    if skip_tests:
        steps = [
            step
            for step in steps
            if step.name != "Run test suite"
        ]

    if from_step is None:
        return steps

    matching_indices = [
        index
        for index, step in enumerate(steps)
        if step.name.lower()
        == from_step.lower()
    ]

    if not matching_indices:
        available = "\n".join(
            f"- {step.name}"
            for step in steps
        )

        raise ValueError(
            f"Unknown pipeline step: {from_step}\n"
            f"Available steps:\n{available}"
        )

    return steps[matching_indices[0]:]


def main() -> int:
    args = parse_arguments()

    if args.list_steps:
        print("Available pipeline steps:")

        for index, step in enumerate(
            PIPELINE_STEPS,
            start=1,
        ):
            print(f"{index}. {step.name}")

        return 0

    steps = select_steps(
        skip_tests=args.skip_tests,
        from_step=args.from_step,
    )

    print("Narrative-Aware ETF Rotation")
    print("Reproducible Research Pipeline")
    print("=" * 100)
    print(
        f"Python: {sys.executable}"
    )
    print(
        f"Project root: {PROJECT_ROOT}"
    )
    print(
        f"Steps selected: {len(steps)}"
    )

    total_start = time.perf_counter()
    completed = []

    for step in steps:
        try:
            elapsed = run_step(step)

            completed.append(
                {
                    "name": step.name,
                    "seconds": elapsed,
                    "status": "PASS",
                }
            )

        except Exception as exc:
            completed.append(
                {
                    "name": step.name,
                    "seconds": 0.0,
                    "status": "FAILED",
                }
            )

            print("")
            print(
                f"[FAILED] {step.name}: {exc}"
            )

            if (
                not step.optional
                or not args.continue_on_error
            ):
                print("")
                print("Pipeline status: FAILED")
                return 1

    total_elapsed = (
        time.perf_counter() - total_start
    )

    print("")
    print("=" * 100)
    print("Pipeline Summary")
    print("=" * 100)

    for record in completed:
        print(
            f"{record['status']:>6}  "
            f"{record['seconds']:>8.2f}s  "
            f"{record['name']}"
        )

    print("")
    print(
        f"Total elapsed: {total_elapsed:.2f} seconds"
    )
    print(
        "Primary model: MOM60"
    )
    print(
        "Exploratory model: "
        "MOM60 + 50% Market Attention Proxy"
    )
    print(
        "Report: "
        "outputs/reporting/research_report.md"
    )
    print("")
    print("Pipeline status: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

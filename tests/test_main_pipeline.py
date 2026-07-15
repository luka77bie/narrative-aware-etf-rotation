import main


def test_pipeline_contains_required_steps() -> None:
    names = [
        step.name
        for step in main.PIPELINE_STEPS
    ]

    assert "Build momentum signal" in names
    assert (
        "Build market-attention proxy"
        in names
    )
    assert (
        "Run walk-forward validation"
        in names
    )
    assert (
        "Generate research charts"
        in names
    )
    assert (
        "Generate research report"
        in names
    )
    assert (
        "Generate HTML report"
        in names
    )


def test_skip_tests_removes_test_step() -> None:
    steps = main.select_steps(
        skip_tests=True,
        from_step=None,
    )

    names = [
        step.name
        for step in steps
    ]

    assert "Run test suite" not in names


def test_from_step_selects_remaining_steps() -> None:
    steps = main.select_steps(
        skip_tests=False,
        from_step="Generate research charts",
    )

    assert (
        steps[0].name
        == "Generate research charts"
    )


def test_all_required_outputs_are_paths() -> None:
    for step in main.PIPELINE_STEPS:
        for output in step.required_outputs:
            assert output.is_absolute() is False

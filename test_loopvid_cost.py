from scripts.loopvid.cost import (
    estimate_run_cost,
    cost_breakdown_lines,
    BudgetExceededError,
    enforce_budget,
)


def test_estimate_60min_run_is_in_expected_range():
    cost = estimate_run_cost(duration_sec=3600)
    assert 1.5 < cost < 6.0   # generous range — adjust as pricing changes


def test_estimate_only_remaining_steps():
    full = estimate_run_cost(duration_sec=3600)
    only_video_mux = estimate_run_cost(
        duration_sec=3600,
        skip=("plan", "music", "image"),
    )
    assert only_video_mux < full
    assert only_video_mux < 1.0


def test_breakdown_lines_includes_all_paid_steps():
    lines = cost_breakdown_lines(duration_sec=3600)
    body = "\n".join(lines)
    for step in ("LLM", "Image", "Music", "Video"):
        assert step in body


def test_enforce_budget_under_max_passes():
    enforce_budget(estimated=2.50, max_cost=5.00)


def test_enforce_budget_over_max_raises():
    import pytest
    with pytest.raises(BudgetExceededError, match="\\$5"):
        enforce_budget(estimated=6.00, max_cost=5.00)

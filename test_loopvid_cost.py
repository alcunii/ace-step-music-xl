from scripts.loopvid.cost import (
    estimate_run_cost,
    cost_breakdown_lines,
    BudgetExceededError,
    enforce_budget,
    segments_for_duration,
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


def test_segments_for_duration_60min_yields_11():
    assert segments_for_duration(3600) == 11


def test_segments_for_duration_5min_yields_1():
    assert segments_for_duration(300) == 1


def test_segments_for_duration_below_threshold_rounds_up():
    assert segments_for_duration(400) == 2   # 400/360 rounded up
    assert segments_for_duration(720) == 2   # 720/360 = 2.0 exactly


def test_segments_for_duration_minimum_is_one():
    assert segments_for_duration(60) == 1
    assert segments_for_duration(1) == 1

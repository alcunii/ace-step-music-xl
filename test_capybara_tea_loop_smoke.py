"""Opt-in live E2E smoke tests for capybara_tea_loop.py.

Two variants — gated independently:
  - 5-min full pipeline smoke: RUN_LIVE_SMOKE=1
  - 60-s turbo music-only smoke: ACESTEP_TURBO_SMOKE=1

Both cost real $ (Replicate + RunPod). Use to validate pipeline
before unattended 60-min runs.

Usage:
    RUN_LIVE_SMOKE=1 pytest test_capybara_tea_loop_smoke.py::test_capybara_tea_loop_5min_e2e -v -s
    ACESTEP_TURBO_SMOKE=1 pytest test_capybara_tea_loop_smoke.py::test_capybara_tea_loop_60s_turbo_smoke -v -s
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

LIVE = os.environ.get("RUN_LIVE_SMOKE", "0") == "1"
TURBO_LIVE = os.environ.get("ACESTEP_TURBO_SMOKE", "0") == "1"


@pytest.mark.skipif(not LIVE, reason="set RUN_LIVE_SMOKE=1 to run")
def test_capybara_tea_loop_5min_e2e(tmp_path):
    """5-min smoke: locked seed, locked setting → final.mp4 exists, ≈ 5 min."""
    out_dir = tmp_path / "out"
    cmd = [
        "python3", "scripts/capybara_tea_loop.py",
        "--setting", "forest_hot_spring",
        "--duration", "300",
        "--seed", "42",
        "--out-dir", str(out_dir),
        "--yes",
    ]
    result = subprocess.run(
        cmd, cwd="/root/ace-step-music-xl",
        capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, (
        f"capybara_tea_loop.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    runs = list(out_dir.glob("capybara-*"))
    assert len(runs) == 1, f"expected one run dir, got {runs}"
    run_dir = runs[0]

    final = run_dir / "final.mp4"
    assert final.exists(), f"no final.mp4 at {final}"

    # ffprobe duration check — within ±10s of 300
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(final)],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip())
    assert 290 <= duration <= 310, f"duration {duration}s out of bounds"

    # Manifest sanity check
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["steps"]["plan"]["status"] == "done"
    assert manifest["steps"]["video"]["status"] == "done"
    assert manifest["steps"]["mux"]["status"] == "done"

    # Plan.json used preset values
    plan = json.loads((run_dir / "plan.json").read_text())
    assert "Nujabes" in plan["music_palette"]
    assert plan["image_archetype_key"] == "capybara_tea"
    assert "capybara" in plan["seedream_scene"].lower()


@pytest.mark.skipif(not TURBO_LIVE, reason="set ACESTEP_TURBO_SMOKE=1 to run")
def test_capybara_tea_loop_60s_turbo_smoke(tmp_path):
    """60-s turbo music-only smoke against the new XL-turbo endpoint.

    Validates that capybara_tea_loop.py + ACE_STEP_TURBO_PRESET +
    DEFAULT_ACE_STEP_ENDPOINT round-trip a single ~60s segment in
    under ~3 minutes (cold-start + 8-step turbo gen).
    """
    out_dir = tmp_path / "out"
    cmd = [
        "python3", "scripts/capybara_tea_loop.py",
        "--setting", "forest_hot_spring",
        "--duration", "60",
        "--seed", "1",
        "--out-dir", str(out_dir),
        "--only", "music",
        "--yes",
    ]
    result = subprocess.run(
        cmd, cwd="/root/ace-step-music-xl",
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"capybara_tea_loop.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    runs = list(out_dir.glob("capybara-*"))
    assert len(runs) == 1, f"expected one run dir, got {runs}"
    run_dir = runs[0]

    seg = run_dir / "music" / "seg_01.mp3"
    assert seg.exists(), f"no music segment at {seg}"
    assert seg.stat().st_size > 10_000, (
        f"music segment suspiciously small: {seg.stat().st_size}B"
    )

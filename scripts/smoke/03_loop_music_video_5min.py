#!/usr/bin/env python3
"""Live 5-minute end-to-end smoke test for loop_music_video.

Runs the full orchestrator with --duration 300, exercising every step.
Cost: ~$0.40, wall time ~10 min.

Usage:
  RUNPOD_API_KEY=... OPENROUTER_API_KEY=... REPLICATE_API_TOKEN=... \\
    python3 scripts/smoke/03_loop_music_video_5min.py
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_ID = "smoke-5min"


def main() -> int:
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts/loop_music_video.py"),
        "--genre", "ambient",
        "--mood", "smoke test",
        "--duration", "300",
        "--run-id", RUN_ID,
        "--yes",
    ]
    print(f"running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return result.returncode

    final = REPO_ROOT / "out" / "loop_video" / RUN_ID / "final.mp4"
    if not final.exists():
        print(f"FAIL: {final} not produced")
        return 1

    probe = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height : format=duration",
        "-of", "csv=p=0", str(final),
    ], capture_output=True, text=True, check=True)
    print(f"ffprobe: {probe.stdout}")
    if "1280,704" not in probe.stdout:
        print("FAIL: dimensions != 1280x704")
        return 1
    print(f"PASS: {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

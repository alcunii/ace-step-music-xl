import subprocess
from pathlib import Path

import pytest

from scripts.loopvid.loop_build import (
    concat_clips_with_xfades,
    add_loop_seam_fade,
)


def make_color_clip(path: Path, color: str, duration: float = 7.0):
    """Synthesize a solid-color H.264 video clip at 1280x704 24fps for tests."""
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c={color}:s=1280x704:r=24:d={duration}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-tune", "stillimage",
        "-an", str(path),
    ], capture_output=True, check=True)


def probe_duration(path: Path) -> float:
    r = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def probe_dimensions(path: Path) -> tuple[int, int]:
    r = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path),
    ], capture_output=True, text=True, check=True)
    w, h = r.stdout.strip().split(",")
    return int(w), int(h)


def test_concat_clips_with_xfades_produces_expected_duration(tmp_path):
    clips = []
    for i, c in enumerate(["red", "green", "blue", "yellow", "cyan", "magenta"], start=1):
        p = tmp_path / f"clip_{i:02d}.mp4"
        make_color_clip(p, c, duration=7.0)
        clips.append(p)
    out = tmp_path / "concat.mp4"
    concat_clips_with_xfades(clips, out, xfade_sec=0.25)
    d = probe_duration(out)
    # 6 × 7 − 5 × 0.25 = 40.75s
    assert 40.5 < d < 41.0, f"got duration {d}"


def test_concat_preserves_dimensions(tmp_path):
    clips = []
    for i in range(2):
        p = tmp_path / f"c{i}.mp4"
        make_color_clip(p, "red", duration=3)
        clips.append(p)
    out = tmp_path / "concat.mp4"
    concat_clips_with_xfades(clips, out, xfade_sec=0.25)
    assert probe_dimensions(out) == (1280, 704)


def test_concat_atomic_write(tmp_path):
    clips = []
    for i in range(2):
        p = tmp_path / f"c{i}.mp4"
        make_color_clip(p, "red", duration=3)
        clips.append(p)
    out = tmp_path / "concat.mp4"
    concat_clips_with_xfades(clips, out, xfade_sec=0.25)
    assert not (tmp_path / "concat.mp4.tmp").exists()


def test_add_loop_seam_fade_shortens_by_fade_duration(tmp_path):
    base = tmp_path / "base.mp4"
    make_color_clip(base, "red", duration=10.0)
    out = tmp_path / "seamed.mp4"
    add_loop_seam_fade(base, out, fade_sec=0.5)
    d = probe_duration(out)
    # 10s minus 0.5s seam fade ≈ 9.5s
    assert 9.3 < d < 9.7

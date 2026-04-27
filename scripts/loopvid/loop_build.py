"""ffmpeg pipeline: concat 6 clips with xfades, then add loop-seam fade."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from scripts.loopvid.constants import INTER_CLIP_XFADE_SEC, LOOP_SEAM_XFADE_SEC


def _probe_duration(path: Path) -> float:
    """Get duration of video file in seconds."""
    r = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def concat_clips_with_xfades(
    clips: list[Path], out_path: Path,
    *, xfade_sec: float = INTER_CLIP_XFADE_SEC,
) -> None:
    """Concatenate clips with xfade transitions between adjacent pairs.

    Builds a filter_complex chain where each xfade overlaps the tail of clip N
    with the head of clip N+1, shortening the total duration by (N-1)*xfade_sec.

    Args:
        clips: List of video file paths in order.
        out_path: Output file path.
        xfade_sec: Fade duration in seconds. Defaults to INTER_CLIP_XFADE_SEC.

    Raises:
        ValueError: If fewer than 2 clips provided.
        RuntimeError: If ffmpeg fails.
    """
    if len(clips) < 2:
        raise ValueError(f"need at least 2 clips, got {len(clips)}")
    out_path = Path(out_path)
    tmp = out_path.parent / (out_path.name + ".tmp")

    inputs = []
    for c in clips:
        inputs += ["-i", str(c)]

    durations = [_probe_duration(c) for c in clips]
    filters = []
    prev_label = "0:v"
    running_offset = 0.0
    for i in range(1, len(clips)):
        next_label = f"v{i:02d}"
        offset = running_offset + durations[i - 1] - xfade_sec
        running_offset = offset
        filters.append(
            f"[{prev_label}][{i}:v]xfade=transition=fade:duration={xfade_sec}:"
            f"offset={offset}[{next_label}]"
        )
        prev_label = next_label

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", ";".join(filters),
        "-map", f"[{prev_label}]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an",
        "-f", "mp4",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat failed (exit {result.returncode}):\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    os.replace(tmp, out_path)


def add_loop_seam_fade(
    base: Path, out_path: Path,
    *, fade_sec: float = LOOP_SEAM_XFADE_SEC,
) -> None:
    """Make the file loop-seamlessly by xfading the last fade_sec into the first
    fade_sec. Final duration = original − fade_sec.

    Splits video into front (0 to duration-fade_sec) and tail (duration-fade_sec
    to duration), then xfades them together with overlap.

    Args:
        base: Input video file path.
        out_path: Output file path.
        fade_sec: Fade duration in seconds. Defaults to LOOP_SEAM_XFADE_SEC.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    base = Path(base)
    out_path = Path(out_path)
    tmp = out_path.parent / (out_path.name + ".tmp")
    duration = _probe_duration(base)
    tail_offset = duration - fade_sec

    cmd = [
        "ffmpeg", "-y", "-i", str(base),
        "-filter_complex",
        f"[0:v]split=2[front][tail];"
        f"[front]trim=0:{tail_offset},setpts=PTS-STARTPTS[a];"
        f"[tail]trim={tail_offset}:{duration},setpts=PTS-STARTPTS[b];"
        f"[a][b]xfade=transition=fade:duration={fade_sec}:offset={tail_offset - fade_sec}[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
        "-f", "mp4",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg seam fade failed (exit {result.returncode}):\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    os.replace(tmp, out_path)

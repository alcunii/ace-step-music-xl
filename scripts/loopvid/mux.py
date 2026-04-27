"""Final assembly — stream-loop video to target, trim audio, mux."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _atomic_run(cmd: list[str], target: Path) -> None:
    """Replace the last arg of cmd with a .tmp path, run, atomic-rename to target."""
    tmp = target.with_stem(target.stem + ".tmp")
    cmd = list(cmd)
    cmd[-1] = str(tmp)
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    os.replace(tmp, target)


def stream_loop_video(base: Path, out: Path, *, target_sec: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(base),
        "-t", str(target_sec), "-c:v", "copy", "-an", str(out),
    ]
    _atomic_run(cmd, Path(out))


def trim_audio(src: Path, out: Path, *, target_sec: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-t", str(target_sec), "-c:a", "copy", str(out),
    ]
    _atomic_run(cmd, Path(out))


def mux_video_audio(video: Path, audio: Path, out: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video), "-i", str(audio),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(out),
    ]
    _atomic_run(cmd, Path(out))


def final_assembly(
    loop_seamed: Path, music_master: Path, out: Path,
    *, target_sec: int, work_dir: Path,
) -> Path:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    video_full = work_dir / "video_60min.mp4"
    audio_full = work_dir / "music_60min.mp3"
    stream_loop_video(loop_seamed, video_full, target_sec=target_sec)
    trim_audio(music_master, audio_full, target_sec=target_sec)
    mux_video_audio(video_full, audio_full, Path(out))
    return Path(out)

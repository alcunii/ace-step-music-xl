import subprocess
from pathlib import Path

import pytest

from scripts.loopvid.mux import (
    stream_loop_video,
    trim_audio,
    mux_video_audio,
    final_assembly,
)


def make_color_video(path: Path, duration: float):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=red:s=1280x704:r=24:d={duration}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path),
    ], capture_output=True, check=True)


def make_silent_audio(path: Path, duration: float):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "anullsrc=r=44100:cl=stereo", "-t", str(duration),
        "-c:a", "libmp3lame", "-b:a", "128k", str(path),
    ], capture_output=True, check=True)


def probe_duration(path):
    r = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def probe_codec(path, stream_type):
    r = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", f"{stream_type}:0",
        "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path),
    ], capture_output=True, text=True, check=True)
    return r.stdout.strip()


def test_stream_loop_video_fills_target_duration(tmp_path):
    base = tmp_path / "base.mp4"
    make_color_video(base, 5.0)
    out = tmp_path / "looped.mp4"
    stream_loop_video(base, out, target_sec=20)
    d = probe_duration(out)
    assert 19.5 < d < 20.5


def test_stream_loop_video_uses_copy_codec(tmp_path):
    base = tmp_path / "base.mp4"
    make_color_video(base, 3.0)
    out = tmp_path / "looped.mp4"
    stream_loop_video(base, out, target_sec=10)
    assert probe_codec(out, "v") == "h264"


def test_trim_audio_to_exact_duration(tmp_path):
    src = tmp_path / "in.mp3"
    make_silent_audio(src, 30)
    out = tmp_path / "out.mp3"
    trim_audio(src, out, target_sec=20)
    d = probe_duration(out)
    assert 19.5 < d < 20.5


def test_mux_combines_video_and_audio(tmp_path):
    v = tmp_path / "v.mp4"
    a = tmp_path / "a.mp3"
    make_color_video(v, 10)
    make_silent_audio(a, 10)
    out = tmp_path / "final.mp4"
    mux_video_audio(v, a, out)
    assert probe_codec(out, "v") == "h264"
    assert probe_codec(out, "a") == "aac"


def test_final_assembly_end_to_end(tmp_path):
    seamed = tmp_path / "loop_seamed.mp4"
    master = tmp_path / "master.mp3"
    make_color_video(seamed, 5.0)
    make_silent_audio(master, 30.0)
    out = tmp_path / "final.mp4"
    final_assembly(seamed, master, out, target_sec=15, work_dir=tmp_path / "work")
    d = probe_duration(out)
    assert 14.5 < d < 15.5
    assert probe_codec(out, "v") == "h264"
    assert probe_codec(out, "a") == "aac"

import base64
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.loopvid.video_pipeline import (
    slice_audio_chunks,
    build_clip_payload,
    stable_clip_seed,
)
from scripts.loopvid.constants import (
    CLIP_COUNT, CLIP_NUM_FRAMES, CLIP_FPS, CLIP_WIDTH, CLIP_HEIGHT,
    LTX_NEGATIVE_PROMPT,
)


def make_silent_mp3(path: Path, duration_sec: float):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r=44100:cl=stereo", "-t", str(duration_sec),
        "-c:a", "libmp3lame", "-b:a", "128k", str(path),
    ], capture_output=True, check=True)


def test_slice_audio_chunks_yields_six_files(tmp_path):
    master = tmp_path / "master.mp3"
    make_silent_mp3(master, 60.0)
    out_dir = tmp_path / "chunks"
    chunks = slice_audio_chunks(master, out_dir, count=CLIP_COUNT,
                                clip_duration_sec=CLIP_NUM_FRAMES / CLIP_FPS)
    assert len(chunks) == CLIP_COUNT
    for p in chunks:
        assert p.exists()
        assert p.stat().st_size > 0


def test_slice_audio_chunk_durations_match_clip_duration(tmp_path):
    master = tmp_path / "master.mp3"
    make_silent_mp3(master, 60.0)
    chunks = slice_audio_chunks(master, tmp_path / "chunks", count=CLIP_COUNT,
                                clip_duration_sec=7.0417)
    for p in chunks:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(p)],
            capture_output=True, text=True, check=True,
        )
        d = float(probe.stdout.strip())
        assert 6.9 < d < 7.2, f"chunk duration {d} not near 7s"


def test_slice_audio_atomic_write(tmp_path):
    master = tmp_path / "master.mp3"
    make_silent_mp3(master, 60.0)
    out = tmp_path / "chunks"
    slice_audio_chunks(master, out, count=CLIP_COUNT, clip_duration_sec=7.0417)
    assert not list(out.glob("*.tmp"))


def test_build_clip_payload_uses_constants():
    p = build_clip_payload(
        image_b64="img", audio_b64="aud", motion_prompt="rain falls",
        seed=42,
    )
    assert p["input"]["image_base64"] == "img"
    assert p["input"]["audio_base64"] == "aud"
    assert p["input"]["prompt"] == "rain falls"
    assert p["input"]["negative_prompt"] == LTX_NEGATIVE_PROMPT
    assert p["input"]["num_frames"] == CLIP_NUM_FRAMES
    assert p["input"]["fps"] == CLIP_FPS
    assert p["input"]["width"] == CLIP_WIDTH
    assert p["input"]["height"] == CLIP_HEIGHT
    assert p["input"]["seed"] == 42


def test_stable_clip_seed_deterministic():
    s1 = stable_clip_seed("run-1", 1)
    s2 = stable_clip_seed("run-1", 1)
    assert s1 == s2


def test_stable_clip_seed_differs_between_clips():
    assert stable_clip_seed("run-1", 1) != stable_clip_seed("run-1", 2)


def test_stable_clip_seed_differs_between_runs():
    assert stable_clip_seed("run-1", 1) != stable_clip_seed("run-2", 1)


def test_build_clip_payload_uses_custom_negative_when_given():
    payload = build_clip_payload(
        image_b64="img", audio_b64="aud", motion_prompt="p", seed=1,
        negative_prompt="custom negative",
    )
    assert payload["input"]["negative_prompt"] == "custom negative"


def test_build_clip_payload_default_negative_unchanged():
    payload = build_clip_payload(
        image_b64="img", audio_b64="aud", motion_prompt="p", seed=1,
    )
    assert payload["input"]["negative_prompt"] == LTX_NEGATIVE_PROMPT

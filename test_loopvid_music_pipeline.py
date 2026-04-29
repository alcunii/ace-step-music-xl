import base64
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.loopvid.music_pipeline import (
    build_segment_payload,
    run_music_pipeline,
    stitch_segments,
)
from scripts.loopvid.constants import ACE_STEP_PRESET, ACE_STEP_TURBO_PRESET


def test_build_segment_payload_uses_official_preset():
    p = build_segment_payload(prompt="x", duration=360, seed=42)
    for k, v in ACE_STEP_PRESET.items():
        assert p["input"][k] == v


def test_build_segment_payload_text2music_mp3():
    p = build_segment_payload(prompt="x", duration=360, seed=42)
    assert p["input"]["task_type"] == "text2music"
    assert p["input"]["audio_format"] == "mp3"
    assert p["input"]["instrumental"] is True
    assert p["input"]["thinking"] is False
    assert p["input"]["batch_size"] == 1
    assert p["input"]["duration"] == 360


def test_build_segment_payload_includes_prompt_and_seed():
    p = build_segment_payload(prompt="lofi piano", duration=360, seed=7)
    assert p["input"]["prompt"] == "lofi piano"
    assert p["input"]["seed"] == 7


def test_stitch_segments_produces_one_output(tmp_path):
    seg_paths = []
    for i in range(1, 4):
        p = tmp_path / f"seg_{i:02d}.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi",
             "-i", f"anullsrc=r=44100:cl=stereo", "-t", "5",
             "-c:a", "libmp3lame", "-b:a", "128k", str(p)],
            capture_output=True, check=True,
        )
        seg_paths.append(p)
    out = tmp_path / "master.mp3"
    stitch_segments(seg_paths, out, crossfade_sec=1)
    assert out.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    )
    d = float(probe.stdout.strip())
    # 3 × 5s − 2 × 1s xfade = 13s
    assert 12 < d < 14


def test_run_music_pipeline_skips_existing_segments(tmp_path):
    """If seg_03.mp3 already exists, the pipeline should not re-call the API."""
    out_dir = tmp_path / "music"
    out_dir.mkdir()
    existing = out_dir / "seg_03.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"anullsrc=r=44100:cl=stereo", "-t", "1",
         "-c:a", "libmp3lame", "-b:a", "128k", str(existing)],
        capture_output=True, check=True,
    )

    call_count = {"n": 0}

    def fake_run_segment(*, payload, **_):
        call_count["n"] += 1
        with subprocess.Popen(
            ["ffmpeg", "-y", "-f", "lavfi",
             "-i", "anullsrc=r=44100:cl=stereo", "-t", "1",
             "-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
            data, _ = p.communicate()
        return {"output": {"audio_base64": base64.b64encode(data).decode()}}

    with patch("scripts.loopvid.music_pipeline.run_segment", side_effect=fake_run_segment):
        run_music_pipeline(
            prompts=[f"prompt-{i}" for i in range(1, 4)],
            duration_sec=1, seeds=[1, 2, 3],
            out_dir=out_dir, endpoint_id="e", api_key="k",
        )
    assert call_count["n"] == 2
    assert (out_dir / "seg_01.mp3").exists()
    assert (out_dir / "seg_02.mp3").exists()


def test_build_segment_payload_default_preset_is_base():
    """No preset arg → falls back to ACE_STEP_PRESET (base)."""
    p = build_segment_payload(prompt="x", duration=360, seed=42, preset=None)
    for k, v in ACE_STEP_PRESET.items():
        assert p["input"][k] == v


def test_build_segment_payload_honours_explicit_preset():
    """preset=ACE_STEP_TURBO_PRESET → 8 steps, cfg=1.0, use_adg=False."""
    p = build_segment_payload(
        prompt="x", duration=360, seed=42,
        preset=ACE_STEP_TURBO_PRESET,
    )
    assert p["input"]["inference_steps"] == 8
    assert p["input"]["guidance_scale"] == 1.0
    assert p["input"]["use_adg"] is False
    assert p["input"]["shift"] == 3.0
    assert p["input"]["infer_method"] == "ode"


def test_build_segment_payload_preset_does_not_leak():
    """Passing turbo preset must NOT mutate the global ACE_STEP_TURBO_PRESET."""
    snapshot = dict(ACE_STEP_TURBO_PRESET)
    build_segment_payload(
        prompt="x", duration=360, seed=42,
        preset=ACE_STEP_TURBO_PRESET,
    )
    assert ACE_STEP_TURBO_PRESET == snapshot

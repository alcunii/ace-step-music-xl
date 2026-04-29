"""ACE-Step XL 11-segment music pipeline + ffmpeg stitch."""
from __future__ import annotations

import base64
import os
import subprocess
from pathlib import Path
from typing import Callable, Optional

from scripts.loopvid.constants import ACE_STEP_PRESET, CROSSFADE_SEC
from scripts.loopvid.runpod_client import run_segment


def build_segment_payload(
    *, prompt: str, duration: int, seed: int,
    preset: dict | None = None,
) -> dict:
    """Build a text2music payload. Default preset is ACE_STEP_PRESET (base);
    pass ACE_STEP_TURBO_PRESET (or any other dict) to override."""
    chosen = preset if preset is not None else ACE_STEP_PRESET
    return {
        "input": {
            "task_type": "text2music",
            "prompt": prompt,
            "lyrics": "",
            "instrumental": True,
            "duration": duration,
            "seed": seed,
            "batch_size": 1,
            "audio_format": "mp3",
            "thinking": False,
            **chosen,
        }
    }


def _save_segment(output: dict, target: Path) -> None:
    b64 = output.get("audio_base64", "")
    if not b64:
        raise RuntimeError(
            f"ACE-Step response missing 'audio_base64': keys={sorted(output.keys())}"
        )
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(base64.b64decode(b64, validate=True))
    os.replace(tmp, target)


def run_music_pipeline(
    *, prompts: list[str], duration_sec: int, seeds: list[int],
    out_dir: Path, endpoint_id: str, api_key: str,
    preset: dict | None = None,
    on_segment_done: Optional[Callable[[int, Path], None]] = None,
) -> list[Path]:
    """Submit N segments sequentially. Skips canonical files that already exist.

    preset: optional override for the ACE-Step inference preset.
            Default = ACE_STEP_PRESET (base XL high-quality).
            Pass ACE_STEP_TURBO_PRESET for the distilled turbo variant.
    """
    if len(prompts) != len(seeds):
        raise ValueError(f"prompts ({len(prompts)}) and seeds ({len(seeds)}) must align")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (prompt, seed) in enumerate(zip(prompts, seeds), start=1):
        target = out_dir / f"seg_{i:02d}.mp3"
        if target.exists():
            paths.append(target)
            continue
        payload = build_segment_payload(prompt=prompt, duration=duration_sec, seed=seed, preset=preset)
        body = run_segment(
            endpoint_id=endpoint_id, api_key=api_key, payload=payload,
            label=f"music seg {i}",
        )
        output = body.get("output", {})
        if isinstance(output, dict) and "error" in output:
            raise RuntimeError(f"ACE-Step seg {i} error: {output['error']}")
        _save_segment(output, target)
        paths.append(target)
        if on_segment_done:
            on_segment_done(i, target)
    return paths


def stitch_segments(
    segment_paths: list[Path], out_path: Path,
    *, crossfade_sec: int = CROSSFADE_SEC,
) -> None:
    """Chained acrossfade with equal-power qsin curves."""
    n = len(segment_paths)
    if n < 2:
        raise ValueError(f"need at least 2 segments to crossfade, got {n}")
    cmd: list[str] = ["ffmpeg", "-y"]
    for p in segment_paths:
        cmd += ["-i", str(p)]
    filters: list[str] = []
    prev = "0"
    for i in range(1, n):
        nxt = f"a{i:02d}" if i < n - 1 else "out"
        filters.append(
            f"[{prev}][{i}]acrossfade=d={crossfade_sec}:c1=qsin:c2=qsin[{nxt}]"
        )
        prev = nxt
    cmd += [
        "-filter_complex", "; ".join(filters),
        "-map", "[out]",
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg stitch failed (exit {result.returncode}):\n"
            f"{result.stderr.decode(errors='replace')}"
        )

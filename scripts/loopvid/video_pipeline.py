"""LTX-2.3 video pipeline: 6 sequential clips, audio-conditioned by music slices."""
from __future__ import annotations

import base64
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Callable, Optional

from scripts.loopvid.constants import (
    CLIP_COUNT, CLIP_NUM_FRAMES, CLIP_FPS, CLIP_WIDTH, CLIP_HEIGHT,
    CLIP_DURATION_SEC, LTX_NEGATIVE_PROMPT,
)
from scripts.loopvid.runpod_client import run_segment


def slice_audio_chunks(
    master_path: Path, out_dir: Path,
    *, count: int = CLIP_COUNT,
    clip_duration_sec: float = CLIP_DURATION_SEC,
) -> list[Path]:
    """Slice the first count*clip_duration_sec seconds of master_path into
    `count` equal MP3 chunks. Atomic per-chunk write."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for i in range(1, count + 1):
        target = out_dir / f"clip_{i:02d}.mp3"
        tmp = out_dir / f"clip_{i:02d}.tmp.mp3"
        start = (i - 1) * clip_duration_sec
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-t", str(clip_duration_sec),
            "-i", str(master_path),
            "-c:a", "libmp3lame", "-b:a", "128k", str(tmp),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg slice failed for chunk {i}: {result.stderr.decode(errors='replace')}"
            )
        os.replace(tmp, target)
        chunks.append(target)
    return chunks


def build_clip_payload(
    *, image_b64: str, audio_b64: str, motion_prompt: str, seed: int,
    negative_prompt: str = LTX_NEGATIVE_PROMPT,
) -> dict:
    return {
        "input": {
            "image_base64": image_b64,
            "audio_base64": audio_b64,
            "prompt": motion_prompt,
            "negative_prompt": negative_prompt,
            "num_frames": CLIP_NUM_FRAMES,
            "fps": CLIP_FPS,
            "seed": seed,
            "width": CLIP_WIDTH,
            "height": CLIP_HEIGHT,
        }
    }


def stable_clip_seed(run_id: str, clip_num: int) -> int:
    """Deterministic seed per (run_id, clip_num) — survives resume."""
    h = hashlib.sha256(f"{run_id}:{clip_num}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def _save_clip_video(output: dict, out_path: Path) -> None:
    b64 = output.get("video", "")
    if not b64:
        raise RuntimeError(f"LTX response missing 'video' field: {sorted(output.keys())}")
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(base64.b64decode(b64, validate=True))
    os.replace(tmp, out_path)


def run_video_pipeline(
    *, run_id: str,
    still_path: Path,
    audio_chunks: list[Path],
    motion_prompts: list[str],
    out_dir: Path,
    endpoint_id: str,
    api_key: str,
    on_clip_done: Optional[Callable[[int, Path], None]] = None,
    negative_prompt: str = LTX_NEGATIVE_PROMPT,
) -> list[Path]:
    """Submit one LTX call per clip, sequentially. Skips clips with existing
    canonical output. Returns list of clip_NN.mp4 paths in order."""
    if len(audio_chunks) != CLIP_COUNT or len(motion_prompts) != CLIP_COUNT:
        raise ValueError(f"need exactly {CLIP_COUNT} chunks and prompts")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_b64 = base64.b64encode(Path(still_path).read_bytes()).decode()

    clips = []
    for i in range(1, CLIP_COUNT + 1):
        target = out_dir / f"clip_{i:02d}.mp4"
        if target.exists():
            clips.append(target)
            continue
        audio_b64 = base64.b64encode(audio_chunks[i - 1].read_bytes()).decode()
        payload = build_clip_payload(
            image_b64=image_b64,
            audio_b64=audio_b64,
            motion_prompt=motion_prompts[i - 1],
            seed=stable_clip_seed(run_id, i),
            negative_prompt=negative_prompt,
        )
        body = run_segment(
            endpoint_id=endpoint_id, api_key=api_key, payload=payload,
            label=f"video clip {i}",
            max_retries=1,
        )
        output = body.get("output", {})
        if "error" in output:
            raise RuntimeError(f"LTX clip {i} returned error: {output['error']}")
        _save_clip_video(output, target)
        clips.append(target)
        if on_clip_done:
            on_clip_done(i, target)
    return clips

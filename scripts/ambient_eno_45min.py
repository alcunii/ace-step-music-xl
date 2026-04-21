#!/usr/bin/env python3
"""Generate a ~46-minute Eno-style tonal ambient piece via the ACE-Step 1.5 XL
serverless endpoint. Submits 7 text2music segments sequentially, saves each as
FLAC with a JSON sidecar, and stitches them locally with ffmpeg acrossfade.

Design: docs/superpowers/specs/2026-04-21-ambient-eno-45min-design.md

Usage:
  export RUNPOD_API_KEY=<your-key>
  export RUNPOD_ENDPOINT_ID=nwqnd0duxc6o38
  python3 scripts/ambient_eno_45min.py [--run-id ID] [--force] [--segment N]
                                       [--stitch-only] [--dry-run]
                                       [--pin-seeds-from RUN-ID]
                                       [--duration SEC]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Constants — locked sonic palette, per-segment evolution, and the official
# ACE-Step 1.5 XL-base "High-Quality Generation" preset.
# Source: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md
# ---------------------------------------------------------------------------
LOCKED_PALETTE = (
    "Eno-inspired tonal ambient, slowly evolving pads, soft felted piano "
    "notes scattered over sustained synthesizer bed, warm analog tape "
    "bloom, long plate-reverb tails, harmonic major key, 50 BPM in 4/4 "
    "(barely perceptible pulse), no percussion, no vocals, spacious "
    "stereo field, gentle overtone shimmer, meditative calm atmosphere, "
    "pristine recording"
)

SEGMENT_DESCRIPTORS = [
    {"phase": "Inhale-1", "descriptors": "sparse piano notes, soft entry, first unfolding"},
    {"phase": "Inhale-2", "descriptors": "settling pads, slightly lower register"},
    {"phase": "Inhale-3", "descriptors": "deepest stillness, fewest events, suspended"},
    {"phase": "Turn",     "descriptors": "widest reverb, slowest harmonic change, held breath"},
    {"phase": "Exhale-1", "descriptors": "overtones emerging, air widening"},
    {"phase": "Exhale-2", "descriptors": "sparser piano, more air, upper register glow"},
    {"phase": "Dissolve", "descriptors": "dissolving pads, long diminuendo, fade into silence"},
]

PRESET = {
    "inference_steps": 64,
    "guidance_scale": 8.0,
    "shift": 3.0,
    "use_adg": True,
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
    "infer_method": "ode",
}

SEGMENT_COUNT = 7
SEGMENT_DURATION_SEC = 420
CROSSFADE_SEC = 30
DEFAULT_ENDPOINT_ID = "nwqnd0duxc6o38"

POLL_INTERVAL_SEC = 5
REQUEST_TIMEOUT_SEC = 1800
MAX_TRANSIENT_404 = 6
MAX_SEGMENT_RETRIES = 3


# ---------------------------------------------------------------------------
# Prompt + payload builders (pure functions)
# ---------------------------------------------------------------------------
def build_segment_prompt(segment_num: int) -> str:
    """Return the full text2music prompt for segment `segment_num` (1..7).

    Locked palette + the segment's unique descriptors, comma-joined.
    """
    if not 1 <= segment_num <= SEGMENT_COUNT:
        raise ValueError(
            f"segment_num must be in 1..{SEGMENT_COUNT}, got {segment_num}"
        )
    extra = SEGMENT_DESCRIPTORS[segment_num - 1]["descriptors"]
    return f"{LOCKED_PALETTE}, {extra}"


def build_payload(segment_num: int, duration: int, seed: int) -> dict:
    """Build the RunPod /runsync `input` payload for one segment."""
    return {
        "task_type": "text2music",
        "prompt": build_segment_prompt(segment_num),
        "lyrics": "",
        "instrumental": True,
        "duration": duration,
        "seed": seed,
        "batch_size": 1,
        "audio_format": "flac",
        "thinking": False,
        **PRESET,
    }


# ---------------------------------------------------------------------------
# Sidecar + manifest I/O (pure file-system helpers)
# ---------------------------------------------------------------------------
def write_sidecar(path: Path, data: dict) -> None:
    """Write one segment's metadata as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def read_sidecar(path: Path) -> dict:
    """Read a sidecar JSON file previously written by `write_sidecar`."""
    return json.loads(Path(path).read_text())


def write_manifest(
    path: Path,
    run_id: str,
    endpoint_id: str,
    seeds: list[int],
    segment_duration: int,
    crossfade_sec: int,
    locked_palette: str,
) -> None:
    """Write the run-level manifest covering all 7 segments."""
    import datetime as _dt
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "endpoint_id": endpoint_id,
        "segment_count": SEGMENT_COUNT,
        "segment_duration_sec": segment_duration,
        "crossfade_sec": crossfade_sec,
        "seeds": seeds,
        "locked_palette": locked_palette,
        "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# RunPod client — submit a job, poll to completion with 404 tolerance.
# Mirrors the pattern in scripts/bruno_mars_style_midnight_gold.py.
# ---------------------------------------------------------------------------
def submit_job(endpoint_id: str, api_key: str, payload: dict) -> dict:
    """POST the payload to /v2/{ep}/runsync. Returns the full response body.

    /runsync can return synchronously with status=COMPLETED and the full
    output already included — the caller should check body["status"] before
    deciding whether to poll.
    """
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    resp = requests.post(
        url,
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=REQUEST_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    return resp.json()


def poll_job(
    endpoint_id: str,
    api_key: str,
    job_id: str,
    *,
    poll_interval: int = POLL_INTERVAL_SEC,
) -> dict:
    """Poll /v2/{ep}/status/{job_id} until COMPLETED, FAILED, CANCELLED, or
    TIMED_OUT. Tolerates up to MAX_TRANSIENT_404 consecutive 404s (the RunPod
    status endpoint is briefly eventually-consistent after submission).

    Returns the full status JSON on COMPLETED. Raises RuntimeError on terminal
    failure or if transient 404 count is exceeded.
    """
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    consecutive_404s = 0
    while True:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 404:
            consecutive_404s += 1
            if consecutive_404s > MAX_TRANSIENT_404:
                raise RuntimeError(
                    f"Too many consecutive 404s polling job {job_id}"
                )
            if poll_interval > 0:
                time.sleep(poll_interval)
            continue
        resp.raise_for_status()
        consecutive_404s = 0
        body = resp.json()
        status = body.get("status", "")
        if status == "COMPLETED":
            return body
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(
                f"Job {job_id} terminal status {status}: "
                f"{body.get('error', body)}"
            )
        # IN_QUEUE / IN_PROGRESS — keep polling
        if poll_interval > 0:
            time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Whole-segment retry wrapper
# ---------------------------------------------------------------------------
def run_segment(
    *,
    endpoint_id: str,
    api_key: str,
    segment_num: int,
    duration: int,
    seed: int,
    poll_interval: int = POLL_INTERVAL_SEC,
    retry_sleep: int = 5,
) -> dict:
    """Submit one segment and poll to completion, with up to
    MAX_SEGMENT_RETRIES whole-segment retries on transient failure.

    Returns the full RunPod status JSON on success. Raises RuntimeError after
    all retries exhausted.
    """
    payload = {"input": build_payload(segment_num, duration, seed)}
    last_err: Optional[BaseException] = None
    for attempt in range(1, MAX_SEGMENT_RETRIES + 1):
        try:
            body = submit_job(endpoint_id, api_key, payload)
            status = body.get("status", "")
            if status == "COMPLETED":
                # /runsync already finished synchronously — body has output.
                return body
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(
                    f"segment {segment_num} submit returned status={status}: "
                    f"{body.get('error', body)}"
                )
            job_id = body.get("id", "")
            if not job_id:
                raise RuntimeError(
                    f"segment {segment_num} submit response missing id: {body}"
                )
            return poll_job(endpoint_id, api_key, job_id,
                            poll_interval=poll_interval)
        except (requests.RequestException, RuntimeError) as e:
            last_err = e
            if attempt < MAX_SEGMENT_RETRIES:
                if retry_sleep > 0:
                    time.sleep(retry_sleep)
                continue
    raise RuntimeError(
        f"segment {segment_num} failed after {MAX_SEGMENT_RETRIES} attempts: "
        f"{last_err}"
    )


# ---------------------------------------------------------------------------
# FLAC save from RunPod output
# ---------------------------------------------------------------------------
def save_flac_from_output(output: dict, path: Path) -> None:
    """Decode the response's audio_base64 and write it to `path` as raw bytes.

    Does NOT transcode — the endpoint was asked for audio_format="flac", so
    the bytes are already FLAC.
    """
    b64 = output.get("audio_base64", "")
    if not b64:
        if "audio_base64" not in output:
            raise ValueError("response missing 'audio_base64' field")
        raise ValueError("response 'audio_base64' is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(b64, validate=True))


# ---------------------------------------------------------------------------
# ffmpeg crossfade stitcher
# ---------------------------------------------------------------------------
def build_ffmpeg_command(
    segment_paths: list[Path],
    out_path: Path,
    crossfade_sec: int,
) -> list[str]:
    """Build the ffmpeg argv for chained acrossfade with equal-power qsin
    curves.

    For N input files, produces (N-1) chained acrossfade filters; the output
    of each feeds the next, so no audio is ever truncated more than once.
    """
    n = len(segment_paths)
    if n < 2:
        raise ValueError(f"need at least 2 segments to crossfade, got {n}")

    cmd: list[str] = ["ffmpeg", "-y"]  # -y: overwrite out_path without prompt
    for p in segment_paths:
        cmd += ["-i", str(p)]

    # Chain: [0][1]acrossfade=...[a01]; [a01][2]acrossfade=...[a02]; ...
    filters: list[str] = []
    prev_label = "0"
    for i in range(1, n):
        next_label = f"a{i:02d}" if i < n - 1 else "out"
        filters.append(
            f"[{prev_label}][{i}]"
            f"acrossfade=d={crossfade_sec}:c1=qsin:c2=qsin"
            f"[{next_label}]"
        )
        prev_label = next_label

    cmd += [
        "-filter_complex", "; ".join(filters),
        "-map", "[out]",
        "-c:a", "flac",
        str(out_path),
    ]
    return cmd


def stitch_segments(
    segment_paths: list[Path],
    out_path: Path,
    crossfade_sec: int = CROSSFADE_SEC,
) -> None:
    """Run ffmpeg to stitch segments with equal-power crossfades. Raises
    RuntimeError with ffmpeg's stderr on non-zero exit."""
    for p in segment_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"segment not found: {p}")
    cmd = build_ffmpeg_command(segment_paths, out_path, crossfade_sec)
    result = subprocess.run(cmd, check=False, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg stitching failed (exit {result.returncode}):\n"
            f"{result.stderr.decode(errors='replace')}"
        )


# ---------------------------------------------------------------------------
# Pre-flight checks — fail fast before any GPU spend
# ---------------------------------------------------------------------------
def preflight_checks(api_key: str, endpoint_id: str, out_dir: Path) -> None:
    """Raise RuntimeError on any missing precondition. Called before the
    orchestrator submits its first segment."""
    if not api_key:
        raise RuntimeError(
            "RUNPOD_API_KEY is not set — export it before running"
        )
    if not endpoint_id:
        raise RuntimeError("endpoint id is not set")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on $PATH — install ffmpeg before running"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(out_dir, os.W_OK):
        raise RuntimeError(f"output directory not writable: {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit("main() not yet implemented (Task 10)")

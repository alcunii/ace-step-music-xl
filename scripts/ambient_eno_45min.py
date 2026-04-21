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
    "Ambient drone music for meditation, 60 BPM pulseless pacing, "
    "no percussion, no vocals, slowly evolving warm synth pads with "
    "wavetable textures, distant felted piano notes, long plate "
    "reverb with 5-8 second decay, subtle field recording of gentle "
    "rain and distant wind, major pentatonic tonality, low dynamic "
    "range, seamless instrument blending, wide stereo field, "
    "weightless meditative atmosphere"
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
AMBIENT_OUT_DIR = Path(__file__).resolve().parent.parent / "out" / "ambient"

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


# ---------------------------------------------------------------------------
# CLI parsing + run-dir helpers
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate ~46 min of Eno-style ambient via ACE-Step XL.",
    )
    p.add_argument("--run-id", default=None,
                   help="Reuse an existing run directory. Default: new YYYY-MM-DD-HHMM.")
    p.add_argument("--force", action="store_true",
                   help="Regenerate all segments, overwriting existing.")
    p.add_argument("--segment", type=int, default=None,
                   choices=list(range(1, SEGMENT_COUNT + 1)),
                   help=f"Regenerate only segment N (1..{SEGMENT_COUNT}).")
    p.add_argument("--stitch-only", action="store_true",
                   help="Skip all API calls; just run ffmpeg on existing segments.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the 7 /runsync envelopes without submitting. "
                        "Offline; does NOT require RUNPOD_API_KEY or ffmpeg.")
    p.add_argument("--pin-seeds-from", default=None, metavar="RUN-ID",
                   help="Reuse exact seeds from a prior run's sidecars.")
    p.add_argument("--duration", type=int, default=SEGMENT_DURATION_SEC,
                   help=f"Per-segment duration in seconds (default {SEGMENT_DURATION_SEC}).")
    return p.parse_args(argv)


def resolve_run_dir(base: Path, run_id: Optional[str]) -> Path:
    """Return the run directory path. If run_id is None, generate a
    YYYY-MM-DD-HHMM id. Does NOT create the directory — caller does that."""
    import datetime as _dt
    if run_id is None:
        run_id = _dt.datetime.now().strftime("%Y-%m-%d-%H%M")
    return Path(base) / run_id


def segment_paths_for(run_dir: Path) -> list[Path]:
    """Return the 7 FLAC segment paths inside run_dir (numbered 01..07)."""
    return [Path(run_dir) / f"segment_{i:02d}.flac"
            for i in range(1, SEGMENT_COUNT + 1)]


def load_pinned_seeds(run_dir: Path) -> list[int]:
    """Read seeds from sidecars in a prior run directory. Raises
    FileNotFoundError if any of the 7 sidecars are missing."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    seeds: list[int] = []
    for i in range(1, SEGMENT_COUNT + 1):
        sc = run_dir / f"segment_{i:02d}.json"
        if not sc.exists():
            raise FileNotFoundError(sc)
        seeds.append(int(read_sidecar(sc)["seed"]))
    return seeds


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def _print(msg: str) -> None:
    print(msg, flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    endpoint_id = os.environ.get(
        "RUNPOD_ENDPOINT_ID", DEFAULT_ENDPOINT_ID,
    ).strip()

    run_dir = resolve_run_dir(AMBIENT_OUT_DIR, args.run_id)
    _print(f"Run directory: {run_dir}")

    pinned_seeds: Optional[list[int]] = None
    if args.pin_seeds_from:
        prior = resolve_run_dir(AMBIENT_OUT_DIR, args.pin_seeds_from)
        pinned_seeds = load_pinned_seeds(prior)
        _print(f"Pinning seeds from {prior}: {pinned_seeds}")

    # Dry-run: print full /runsync envelopes, exit. Offline — does NOT
    # require RUNPOD_API_KEY or ffmpeg to be present.
    if args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, SEGMENT_COUNT + 1):
            seed = pinned_seeds[i - 1] if pinned_seeds else -1
            envelope = {"input": build_payload(i, args.duration, seed)}
            _print(
                f"\n--- segment {i}/{SEGMENT_COUNT} "
                f"({SEGMENT_DESCRIPTORS[i-1]['phase']}) ---"
            )
            _print(json.dumps(envelope, indent=2))
        return 0

    preflight_checks(api_key, endpoint_id, AMBIENT_OUT_DIR)
    run_dir.mkdir(parents=True, exist_ok=True)
    seg_paths = segment_paths_for(run_dir)

    # Stitch-only: skip API calls, run ffmpeg only.
    if args.stitch_only:
        missing = [p for p in seg_paths if not p.exists()]
        if missing:
            _print(f"Cannot stitch — missing segments: {missing}")
            return 2
        final = run_dir / "eno_45min_final.flac"
        _print(f"Stitching {len(seg_paths)} segments → {final}")
        stitch_segments(seg_paths, final, CROSSFADE_SEC)
        _print(f"Done: {final}")
        return 0

    # Full run (or targeted --segment) — generate what's missing.
    seeds_actual: list[int] = [0] * SEGMENT_COUNT
    is_targeted = args.segment is not None

    for i in range(1, SEGMENT_COUNT + 1):
        seg_path = seg_paths[i - 1]
        sidecar_path = run_dir / f"segment_{i:02d}.json"
        # Skip if files exist and we're not forced and either:
        # - Not in targeted mode (normal resume), OR
        # - It's not the targeted segment
        if not args.force and seg_path.exists() and sidecar_path.exists():
            if not is_targeted or i != args.segment:
                seeds_actual[i - 1] = int(read_sidecar(sidecar_path)["seed"])
                _print(f"[{i}/{SEGMENT_COUNT}] skip — already exists")
                continue

        # Orphaned FLAC (crash between FLAC-write and sidecar-write) — warn
        # user that a previously-generated segment is about to be overwritten.
        if seg_path.exists() and not sidecar_path.exists():
            _print(
                f"[{i}/{SEGMENT_COUNT}] WARN: orphaned FLAC at {seg_path.name} "
                f"(missing sidecar); regenerating — prior segment will be replaced"
            )

        seed = pinned_seeds[i - 1] if pinned_seeds else -1
        _print(
            f"[{i}/{SEGMENT_COUNT}] submitting "
            f"{SEGMENT_DESCRIPTORS[i-1]['phase']} "
            f"(duration={args.duration}s, seed={seed})..."
        )
        t0 = time.time()
        result = run_segment(
            endpoint_id=endpoint_id, api_key=api_key,
            segment_num=i, duration=args.duration, seed=seed,
        )
        elapsed = time.time() - t0
        output = result.get("output", {})
        actual_seed = int(output.get("seed", seed))
        seeds_actual[i - 1] = actual_seed
        save_flac_from_output(output, seg_path)
        write_sidecar(sidecar_path, {
            "segment_num": i,
            "phase": SEGMENT_DESCRIPTORS[i - 1]["phase"],
            "seed": actual_seed,
            "prompt": build_segment_prompt(i),
            "duration_requested": args.duration,
            "duration_actual": float(output.get("duration", args.duration)),
            "sample_rate": int(output.get("sample_rate", 48000)),
            "endpoint_id": endpoint_id,
            "run_id": run_dir.name,
        })
        _print(
            f"[{i}/{SEGMENT_COUNT}] done in {elapsed:.1f}s — "
            f"seed={actual_seed} → {seg_path.name}"
        )

    # Fill in seeds for any skipped segments (already on disk).
    for i in range(1, SEGMENT_COUNT + 1):
        if seeds_actual[i - 1] == 0:
            sc = run_dir / f"segment_{i:02d}.json"
            if sc.exists():
                seeds_actual[i - 1] = int(read_sidecar(sc)["seed"])

    write_manifest(
        path=run_dir / "manifest.json",
        run_id=run_dir.name,
        endpoint_id=endpoint_id,
        seeds=seeds_actual,
        segment_duration=args.duration,
        crossfade_sec=CROSSFADE_SEC,
        locked_palette=LOCKED_PALETTE,
    )

    # Stitch if we have all 7 segments on disk.
    missing = [p for p in seg_paths if not p.exists()]
    if missing:
        _print(f"Skipping stitch — {len(missing)} segments still missing.")
        return 0
    final = run_dir / "eno_45min_final.flac"
    _print(f"Stitching {len(seg_paths)} segments → {final}")
    stitch_segments(seg_paths, final, CROSSFADE_SEC)
    _print(f"\nDONE: {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

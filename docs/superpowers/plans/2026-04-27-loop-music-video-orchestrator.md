# Loop Music Video Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI orchestrator at `scripts/loop_music_video.py` that produces a 1-hour 1280×704 looping music video by composing OpenRouter (Gemini 3 Flash), ACE-Step XL serverless, and LTX-2.3 ComfyUI serverless — with atomic-write resume, three-level rollback, and a cost guard.

**Architecture:** A single CLI entry orchestrates 6 sequential pipeline steps (plan → music → image → video → loop_build → mux). Each step writes atomic artifacts and updates a manifest state machine. Refactor the existing `ambient_eno_45min.py` to share runpod-client and PRESET constants via new `loopvid/` package — both scripts then import from there.

**Tech Stack:** Python 3.10+, pytest, requests, ffmpeg, RunPod /v2 API, OpenRouter API, Replicate API.

**Prerequisite:** Plan A (`2026-04-27-ltx-handler-resolution-params.md`) must be deployed and verified — production endpoint `1g0pvlx8ar6qns` must accept `width=1280 height=704` and pass landscape smoke test.

**Spec:** `docs/superpowers/specs/2026-04-27-loop-music-video-design.md`.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `scripts/loopvid/__init__.py` | Create | Package marker |
| `scripts/loopvid/runpod_client.py` | Create | submit_job, poll_job, run_segment — extracted from ambient_eno |
| `scripts/loopvid/constants.py` | Create | PRESET (extracted), SEEDREAM_HARD_CONSTRAINTS, LTX_NEGATIVE_PROMPT, GENRE_ARCHETYPES |
| `scripts/loopvid/manifest.py` | Create | RunManifest dataclass + atomic JSON I/O + step-status helpers |
| `scripts/loopvid/cost.py` | Create | Per-endpoint pricing constants + cost estimator + budget guard |
| `scripts/loopvid/plan_schema.py` | Create | Plan dataclass + JSON-schema validator |
| `scripts/loopvid/llm_planner.py` | Create | OpenRouter call + structured-output retry |
| `scripts/loopvid/image_pipeline.py` | Create | Replicate Seedream 4.5 call + atomic still.png write |
| `scripts/loopvid/video_pipeline.py` | Create | Audio chunk slicing + 6 sequential LTX calls |
| `scripts/loopvid/music_pipeline.py` | Create | 11-segment ACE-Step runner + stitch (uses runpod_client) |
| `scripts/loopvid/loop_build.py` | Create | ffmpeg concat + xfade + loop seam fade |
| `scripts/loopvid/mux.py` | Create | ffmpeg stream-loop + final mux |
| `scripts/loopvid/preflight.py` | Create | env vars + endpoint workersMax checks |
| `scripts/loopvid/orchestrator.py` | Create | Top-level state machine: walk steps, skip done, run pending |
| `scripts/loopvid/rollback.py` | Create | Three-level rollback (forensic / --keep / --hard) |
| `scripts/loop_music_video.py` | Create | CLI entry — argparse + dispatch to orchestrator/rollback |
| `scripts/smoke/03_loop_music_video_5min.py` | Create | Live 5-min end-to-end smoke test |
| `test_loopvid_runpod_client.py` | Create | Tests for shared runpod client |
| `test_loopvid_constants.py` | Create | Snapshot tests for prompt constants |
| `test_loopvid_manifest.py` | Create | Atomic write, resume-state tests |
| `test_loopvid_cost.py` | Create | Cost estimator tests |
| `test_loopvid_plan_schema.py` | Create | Plan validation tests |
| `test_loopvid_llm_planner.py` | Create | OpenRouter mocked tests |
| `test_loopvid_image_pipeline.py` | Create | Replicate mocked tests |
| `test_loopvid_video_pipeline.py` | Create | Audio chunking + LTX mocked tests |
| `test_loopvid_music_pipeline.py` | Create | ACE-Step mocked tests |
| `test_loopvid_loop_build.py` | Create | ffmpeg loop-build tests (real ffmpeg) |
| `test_loopvid_mux.py` | Create | ffmpeg mux tests (real ffmpeg) |
| `test_loopvid_preflight.py` | Create | Preflight check tests |
| `test_loopvid_orchestrator.py` | Create | E2E with all HTTP mocked |
| `test_loopvid_rollback.py` | Create | Rollback level tests |
| `scripts/ambient_eno_45min.py` | Modify | Import shared modules from loopvid (no behavior change) |
| `requirements-test.txt` | Modify | Add `responses` if not present |
| `README.md` | Modify | Add "Loop music video" section |

---

## Phase 0 — Prerequisite verification

### Task 0: Confirm Plan A is deployed

- [ ] **Step 1: Run landscape smoke against production**

```bash
cd /root/ltx23-pro6000
LTX_ENDPOINT_ID=1g0pvlx8ar6qns python3 scripts/smoke_landscape.py
```

Expected: `[landscape smoke] PASS` with dimensions `1280x704`. If FAIL, stop — execute Plan A first.

- [ ] **Step 2: Confirm ACE-Step endpoint workersMax ≥ 1**

```
mcp__runpod__get-endpoint(endpointId="nwqnd0duxc6o38")
```

If `workersMax == 0`, scale up:

```
mcp__runpod__update-endpoint(endpointId="nwqnd0duxc6o38", workersMax=1)
```

- [ ] **Step 3: Confirm env vars are set**

```bash
for v in RUNPOD_API_KEY OPENROUTER_API_KEY REPLICATE_API_TOKEN; do
  test -n "${!v}" && echo "$v: set" || echo "$v: MISSING"
done
```

Expected: all three "set". If any missing, source from `/root/avatar-video/.env` or set explicitly.

---

## Phase 1 — Extract shared modules from ambient_eno_45min.py

### Task 1: Create loopvid package + extract runpod_client

**Files:**
- Create: `scripts/loopvid/__init__.py`
- Create: `scripts/loopvid/runpod_client.py`
- Create: `test_loopvid_runpod_client.py`
- Modify: `scripts/ambient_eno_45min.py` (replace inline functions with imports)

- [ ] **Step 1: Create package skeleton**

```bash
mkdir -p /root/ace-step-music-xl/scripts/loopvid
touch /root/ace-step-music-xl/scripts/loopvid/__init__.py
```

- [ ] **Step 2: Write tests for runpod_client (mocked HTTP via `responses`)**

Create `/root/ace-step-music-xl/test_loopvid_runpod_client.py`:

```python
import json
import pytest
import responses
from scripts.loopvid.runpod_client import submit_job, poll_job, run_segment


EP = "test-endpoint"
KEY = "test-key"


@responses.activate
def test_submit_job_posts_to_run_endpoint():
    responses.add(
        responses.POST,
        f"https://api.runpod.ai/v2/{EP}/run",
        json={"id": "job-1", "status": "IN_QUEUE"},
        status=200,
    )
    body = submit_job(EP, KEY, {"input": {"x": 1}})
    assert body["id"] == "job-1"
    assert body["status"] == "IN_QUEUE"
    assert responses.calls[0].request.headers["Authorization"] == f"Bearer {KEY}"


@responses.activate
def test_poll_job_returns_completed_body():
    responses.add(
        responses.GET,
        f"https://api.runpod.ai/v2/{EP}/status/job-1",
        json={"status": "COMPLETED", "output": {"audio_base64": "abc"}},
        status=200,
    )
    body = poll_job(EP, KEY, "job-1", poll_interval=0)
    assert body["output"]["audio_base64"] == "abc"


@responses.activate
def test_poll_job_tolerates_transient_404s():
    for _ in range(3):
        responses.add(
            responses.GET,
            f"https://api.runpod.ai/v2/{EP}/status/job-1",
            status=404,
        )
    responses.add(
        responses.GET,
        f"https://api.runpod.ai/v2/{EP}/status/job-1",
        json={"status": "COMPLETED", "output": {}},
        status=200,
    )
    body = poll_job(EP, KEY, "job-1", poll_interval=0)
    assert body["status"] == "COMPLETED"


@responses.activate
def test_poll_job_raises_after_too_many_404s():
    for _ in range(10):
        responses.add(
            responses.GET,
            f"https://api.runpod.ai/v2/{EP}/status/job-1",
            status=404,
        )
    with pytest.raises(RuntimeError, match="404"):
        poll_job(EP, KEY, "job-1", poll_interval=0)


@responses.activate
def test_poll_job_raises_on_terminal_failure():
    responses.add(
        responses.GET,
        f"https://api.runpod.ai/v2/{EP}/status/job-1",
        json={"status": "FAILED", "error": "OOM"},
        status=200,
    )
    with pytest.raises(RuntimeError, match="OOM"):
        poll_job(EP, KEY, "job-1", poll_interval=0)
```

- [ ] **Step 3: Run tests — must FAIL (module doesn't exist)**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_runpod_client.py -v 2>&1 | tail -10`
Expected: ImportError or collection error.

- [ ] **Step 4: Implement runpod_client.py — copy from ambient_eno_45min.py**

Create `/root/ace-step-music-xl/scripts/loopvid/runpod_client.py`:

```python
"""Shared RunPod /v2 async-job client.

Extracted from scripts/ambient_eno_45min.py — see that script's docstrings
for the rationale on /run vs /runsync, transient-404 tolerance, and the
whole-segment retry wrapper.
"""
from __future__ import annotations

import time
from typing import Optional

import requests

REQUEST_TIMEOUT_SEC = 1800
POLL_INTERVAL_SEC = 5
MAX_TRANSIENT_404 = 6
MAX_SEGMENT_RETRIES = 3


def submit_job(endpoint_id: str, api_key: str, payload: dict) -> dict:
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    resp = requests.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
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
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    consecutive_404s = 0
    while True:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 404:
            consecutive_404s += 1
            if consecutive_404s > MAX_TRANSIENT_404:
                raise RuntimeError(f"Too many consecutive 404s polling job {job_id}")
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
                f"Job {job_id} terminal status {status}: {body.get('error', body)}"
            )
        if poll_interval > 0:
            time.sleep(poll_interval)


def run_segment(
    *,
    endpoint_id: str,
    api_key: str,
    payload: dict,
    label: str,
    poll_interval: int = POLL_INTERVAL_SEC,
    retry_sleep: int = 5,
    max_retries: int = MAX_SEGMENT_RETRIES,
) -> dict:
    """Submit one job and poll to completion, with up to max_retries retries
    on transient failure. `label` is for error messages only (e.g., 'music seg 3')."""
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            body = submit_job(endpoint_id, api_key, payload)
            status = body.get("status", "")
            if status == "COMPLETED":
                return body
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(
                    f"{label} submit returned status={status}: {body.get('error', body)}"
                )
            job_id = body.get("id", "")
            if not job_id:
                raise RuntimeError(f"{label} submit response missing id: {body}")
            return poll_job(endpoint_id, api_key, job_id, poll_interval=poll_interval)
        except (requests.RequestException, RuntimeError) as e:
            last_err = e
            if attempt < max_retries:
                if retry_sleep > 0:
                    time.sleep(retry_sleep)
                continue
    raise RuntimeError(f"{label} failed after {max_retries} attempts: {last_err}")
```

- [ ] **Step 5: Run tests — must PASS**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_runpod_client.py -v`
Expected: 5 passed.

- [ ] **Step 6: Refactor ambient_eno_45min.py to use shared module**

Edit `scripts/ambient_eno_45min.py`. Replace the inline function definitions of `submit_job`, `poll_job`, and `run_segment` (and their associated constants `REQUEST_TIMEOUT_SEC`, `MAX_TRANSIENT_404`, `MAX_SEGMENT_RETRIES`, `POLL_INTERVAL_SEC`) with an import:

```python
# Near the top, after `import requests`:
from loopvid.runpod_client import (
    submit_job,
    poll_job,
    run_segment as _run_segment_shared,
    REQUEST_TIMEOUT_SEC,
    POLL_INTERVAL_SEC,
    MAX_TRANSIENT_404,
    MAX_SEGMENT_RETRIES,
)
```

Then replace the existing `def run_segment(...)` (the one taking `segment_num`/`duration`/`seed`) with a thin wrapper that builds the payload and delegates:

```python
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
    """Backwards-compatible wrapper — builds the segment payload, then delegates
    to the shared runpod_client.run_segment."""
    payload = {"input": build_payload(segment_num, duration, seed)}
    return _run_segment_shared(
        endpoint_id=endpoint_id,
        api_key=api_key,
        payload=payload,
        label=f"segment {segment_num}",
        poll_interval=poll_interval,
        retry_sleep=retry_sleep,
    )
```

Delete the now-duplicated `def submit_job`, `def poll_job` definitions and the constants block that's now imported.

- [ ] **Step 7: Make `scripts/` importable from repo root for tests**

Create `/root/ace-step-music-xl/conftest.py` (only if not present):

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
```

- [ ] **Step 8: Run existing ambient_eno tests — must still PASS**

Run: `cd /root/ace-step-music-xl && pytest test_ambient_eno_45min.py -v 2>&1 | tail -15`
Expected: 55 passed (or whatever the existing count is). If any fail, the refactor changed behavior — fix before committing.

- [ ] **Step 9: Commit**

```bash
cd /root/ace-step-music-xl
git add scripts/loopvid/__init__.py scripts/loopvid/runpod_client.py \
        test_loopvid_runpod_client.py scripts/ambient_eno_45min.py conftest.py
git commit -m "refactor: extract runpod_client into loopvid package

Shared by scripts/ambient_eno_45min.py (existing) and the upcoming
scripts/loop_music_video.py orchestrator. ambient_eno tests still pass."
```

---

## Phase 2 — Constants module

### Task 2: Create loopvid/constants.py with PRESET, SEEDREAM, LTX, ARCHETYPES

**Files:**
- Create: `scripts/loopvid/constants.py`
- Create: `test_loopvid_constants.py`
- Modify: `scripts/ambient_eno_45min.py` (import PRESET from shared)

- [ ] **Step 1: Write snapshot tests for the constants**

Create `/root/ace-step-music-xl/test_loopvid_constants.py`:

```python
"""Snapshot tests for prompt constants — guards against accidental drift.

Update SHA1 values ONLY after a deliberate, audited change to a constant."""
import hashlib

from scripts.loopvid.constants import (
    ACE_STEP_PRESET,
    SEEDREAM_HARD_CONSTRAINTS,
    LTX_NEGATIVE_PROMPT,
    GENRE_ARCHETYPES,
    SEGMENT_DURATION_SEC,
    SEGMENT_COUNT_60MIN,
    CROSSFADE_SEC,
    CLIP_COUNT,
    CLIP_NUM_FRAMES,
    CLIP_FPS,
    CLIP_WIDTH,
    CLIP_HEIGHT,
    INTER_CLIP_XFADE_SEC,
    LOOP_SEAM_XFADE_SEC,
)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def test_ace_step_preset_matches_official():
    assert ACE_STEP_PRESET == {
        "inference_steps": 64,
        "guidance_scale": 8.0,
        "shift": 3.0,
        "use_adg": True,
        "cfg_interval_start": 0.0,
        "cfg_interval_end": 1.0,
        "infer_method": "ode",
    }


def test_seedream_hard_constraints_text_starts_with_clean_composition():
    assert SEEDREAM_HARD_CONSTRAINTS.startswith("Clean composition with absolutely no text")
    assert "no people" in SEEDREAM_HARD_CONSTRAINTS.lower()
    assert "16:9" in SEEDREAM_HARD_CONSTRAINTS


def test_ltx_negative_prompt_inherits_handler_default():
    assert "blurry" in LTX_NEGATIVE_PROMPT
    assert "watermark" in LTX_NEGATIVE_PROMPT
    assert "scene change" in LTX_NEGATIVE_PROMPT
    assert "morphing" in LTX_NEGATIVE_PROMPT


def test_genre_archetypes_complete_set():
    keys = set(GENRE_ARCHETYPES.keys())
    assert keys == {
        "rainy_window_desk",
        "mountain_ridge_dusk",
        "candle_dark_wood",
        "dim_bar_booth",
        "study_window_book",
        "observatory_dome",
    }
    for key, val in GENRE_ARCHETYPES.items():
        assert "visual" in val
        assert "anchored" in val
        assert "ambient_motion" in val


def test_video_clip_constants():
    assert CLIP_COUNT == 6
    assert CLIP_NUM_FRAMES == 169
    assert (CLIP_NUM_FRAMES - 1) % 8 == 0
    assert CLIP_FPS == 24
    assert CLIP_WIDTH == 1280
    assert CLIP_HEIGHT == 704
    assert CLIP_WIDTH % 32 == 0
    assert CLIP_HEIGHT % 32 == 0


def test_music_segment_constants_yield_at_least_60_minutes():
    total = SEGMENT_COUNT_60MIN * SEGMENT_DURATION_SEC - (SEGMENT_COUNT_60MIN - 1) * CROSSFADE_SEC
    assert total >= 3600, f"music plan only yields {total}s, need >=3600"


def test_loop_seam_constants_are_short():
    assert 0 < INTER_CLIP_XFADE_SEC <= 1.0
    assert 0 < LOOP_SEAM_XFADE_SEC <= 1.0
```

- [ ] **Step 2: Run tests — must FAIL (module doesn't exist)**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_constants.py -v 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement constants.py**

Create `/root/ace-step-music-xl/scripts/loopvid/constants.py`:

```python
"""Shared constants — both ambient_eno_45min.py and loop_music_video.py
import from here. Snapshot-tested in test_loopvid_constants.py."""
from __future__ import annotations

# ── ACE-Step XL-base official "High-Quality Generation" preset ──
# Source: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md
ACE_STEP_PRESET = {
    "inference_steps": 64,
    "guidance_scale": 8.0,
    "shift": 3.0,
    "use_adg": True,
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
    "infer_method": "ode",
}

# ── Music pipeline (60 min target) ──
# 11 × 360 − 10 × 30 = 3660s ≈ 61 min. Final mux trims to exactly 3600s.
SEGMENT_COUNT_60MIN = 11
SEGMENT_DURATION_SEC = 360
CROSSFADE_SEC = 30

# ── Video pipeline ──
CLIP_COUNT = 6
CLIP_NUM_FRAMES = 169                   # 168+1 → (n-1) % 8 == 0 for LTX-2.3
CLIP_FPS = 24
CLIP_DURATION_SEC = CLIP_NUM_FRAMES / CLIP_FPS   # = 7.0417s exact
CLIP_WIDTH = 1280
CLIP_HEIGHT = 704
INTER_CLIP_XFADE_SEC = 0.25
LOOP_SEAM_XFADE_SEC = 0.5

# ── Seedream still — hard constraints appended to every prompt ──
# Phrased as positive-form requirements because Seedream 4.5 has no
# negative_prompt field on the Replicate API.
SEEDREAM_HARD_CONSTRAINTS = (
    "Clean composition with absolutely no text, letters, numbers, words, "
    "captions, watermarks, signage, signs, neon signs, screen text, book "
    "text, handwriting, signatures, logos, brand marks, or any visible "
    "writing of any kind. No people, no human figures, no faces, no hands, "
    "no fingers. Single focal point, uncluttered background, no mirrors, "
    "no reflective glass surfaces, no transparent objects. Photographic "
    "realism, no AI-style artifacts, no oversaturation, no fake bokeh, "
    "no HDR look. Static medium-wide shot, fixed camera, no pan, no zoom, "
    "no parallax. 16:9 cinematic widescreen."
)

# ── LTX video — full negative prompt (handler default + known weaknesses) ──
LTX_NEGATIVE_PROMPT = (
    "blurry, low quality, still frame, frames, watermark, overlay, "
    "titles, has blurbox, has subtitles, "
    "text, letters, numbers, words, captions, logo, "
    "signage, signs, neon text, screen text, on-screen text, "
    "hands, fingers, face, faces, eyes, mouth, lips, "
    "fast motion, sudden movement, rapid motion, camera shake, "
    "camera pan, camera zoom, dolly, tracking shot, handheld, "
    "hard cut, scene change, scene transition, jump cut, "
    "reflections, mirror reflections, glass refraction, "
    "crowd, multiple subjects, multiple animated objects, "
    "splashing water, pouring liquid, explosion, fireworks, sparks, "
    "animal legs, paws, wings, fur detail, "
    "frame stutter, ghosting, motion smear, double exposure, "
    "morphing, warping, melting, glitch"
)

# ── Genre archetypes — closed set; LLM customizes within these ──
GENRE_ARCHETYPES = {
    "rainy_window_desk": {
        "visual": "Desk-by-rainy-window, dusk",
        "anchored": "Notebook + lamp + plant, fixed",
        "ambient_motion": "Rain on window, lamp-light flicker, steam from cup",
    },
    "mountain_ridge_dusk": {
        "visual": "Mountain ridge at golden hour",
        "anchored": "Stone cairn or lone tree mid-frame",
        "ambient_motion": "Mist drift, distant cloud cycle, grass sway",
    },
    "candle_dark_wood": {
        "visual": "Single candle on dark wood surface",
        "anchored": "Candle + brass holder, centered",
        "ambient_motion": "Flame flicker, smoke wisp, dust motes in beam",
    },
    "dim_bar_booth": {
        "visual": "Dim bar booth, bokeh windows in background",
        "anchored": "Whiskey glass + brass lamp",
        "ambient_motion": "Cigar smoke curl, bokeh shimmer, slow ceiling-fan glint",
    },
    "study_window_book": {
        "visual": "Window-lit study with sheet music",
        "anchored": "Open book + quill + window frame",
        "ambient_motion": "Curtain breath, page edge, sun-shaft dust motes",
    },
    "observatory_dome": {
        "visual": "Observatory dome interior at night",
        "anchored": "Telescope silhouette, fixed",
        "ambient_motion": "Star drift through aperture, monitor glow pulse, dust",
    },
}
```

- [ ] **Step 4: Run tests — must PASS**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_constants.py -v`
Expected: 7 passed.

- [ ] **Step 5: Refactor ambient_eno_45min.py to import PRESET**

In `scripts/ambient_eno_45min.py`, replace the inline `PRESET = {...}` block with:

```python
from loopvid.constants import ACE_STEP_PRESET as PRESET
```

- [ ] **Step 6: Verify ambient_eno tests still pass**

Run: `cd /root/ace-step-music-xl && pytest test_ambient_eno_45min.py -v 2>&1 | tail -5`
Expected: still all passing.

- [ ] **Step 7: Commit**

```bash
git add scripts/loopvid/constants.py test_loopvid_constants.py scripts/ambient_eno_45min.py
git commit -m "feat(loopvid): add shared constants module

PRESET extracted from ambient_eno_45min.py. Adds new constants for the
loop_music_video orchestrator: SEEDREAM_HARD_CONSTRAINTS, LTX_NEGATIVE_PROMPT,
GENRE_ARCHETYPES, and clip/segment dimensions. Snapshot-tested."
```

---

## Phase 3 — Manifest module

### Task 3: Create manifest.py with atomic write + step-state helpers

**Files:**
- Create: `scripts/loopvid/manifest.py`
- Create: `test_loopvid_manifest.py`

- [ ] **Step 1: Write tests**

Create `/root/ace-step-music-xl/test_loopvid_manifest.py`:

```python
import json
from pathlib import Path

import pytest

from scripts.loopvid.manifest import (
    RunManifest,
    StepStatus,
    load_manifest,
    save_manifest,
    new_manifest,
    mark_step_done,
    mark_step_failed,
)


def test_new_manifest_has_all_steps_pending():
    m = new_manifest("run-1", {"genre": "ambient", "duration_sec": 3600})
    for step in ("plan", "music", "image", "video", "loop_build", "mux"):
        assert m.steps[step]["status"] == "pending"


def test_save_manifest_atomic_write_no_partial(tmp_path):
    m = new_manifest("run-1", {"genre": "ambient"})
    save_manifest(tmp_path, m)
    assert (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "manifest.json.tmp").exists()


def test_save_then_load_round_trip(tmp_path):
    m1 = new_manifest("run-1", {"genre": "ambient", "duration_sec": 3600})
    save_manifest(tmp_path, m1)
    m2 = load_manifest(tmp_path)
    assert m2.run_id == "run-1"
    assert m2.args["genre"] == "ambient"


def test_mark_step_done_updates_status_and_timestamp(tmp_path):
    m = new_manifest("run-1", {})
    save_manifest(tmp_path, m)
    mark_step_done(tmp_path, "plan", extra={"prediction_id": None})
    m2 = load_manifest(tmp_path)
    assert m2.steps["plan"]["status"] == "done"
    assert "committed_at" in m2.steps["plan"]


def test_mark_step_failed_appends_to_failures_log(tmp_path):
    m = new_manifest("run-1", {})
    save_manifest(tmp_path, m)
    mark_step_failed(tmp_path, "music", "OOM error", attempts=2)
    m2 = load_manifest(tmp_path)
    assert m2.steps["music"]["status"] == "failed"
    assert m2.failures[-1]["step"] == "music"
    assert m2.failures[-1]["error"] == "OOM error"
    assert m2.failures[-1]["attempts"] == 2


def test_load_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path)


def test_corrupt_manifest_raises_clear_error(tmp_path):
    (tmp_path / "manifest.json").write_text("not json")
    with pytest.raises(ValueError, match="manifest"):
        load_manifest(tmp_path)
```

- [ ] **Step 2: Run tests — must FAIL**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_manifest.py -v 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement manifest.py**

Create `/root/ace-step-music-xl/scripts/loopvid/manifest.py`:

```python
"""Run manifest — single source of truth for step state.

Atomic-written via .tmp + os.replace. If the orchestrator crashes between
two steps, the manifest accurately reflects the world."""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
STEP_NAMES = ("plan", "music", "image", "video", "loop_build", "mux")


@dataclass
class StepStatus:
    status: str = "pending"     # pending | in_progress | done | failed
    attempts: int = 0


@dataclass
class RunManifest:
    run_id: str
    schema_version: int
    created_at: str
    last_updated: str
    args: dict
    endpoints: dict
    steps: dict
    cost_estimate_usd: float = 0.0
    cost_actual_usd: float = 0.0
    failures: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "args": self.args,
            "endpoints": self.endpoints,
            "steps": self.steps,
            "cost_estimate_usd": self.cost_estimate_usd,
            "cost_actual_usd": self.cost_actual_usd,
            "failures": self.failures,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunManifest":
        return cls(**d)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_manifest(run_id: str, args: dict, endpoints: Optional[dict] = None) -> RunManifest:
    now = _now_iso()
    return RunManifest(
        run_id=run_id,
        schema_version=SCHEMA_VERSION,
        created_at=now,
        last_updated=now,
        args=dict(args),
        endpoints=endpoints or {},
        steps={name: {"status": "pending", "attempts": 0} for name in STEP_NAMES},
    )


def _path(run_dir: Path) -> Path:
    return Path(run_dir) / MANIFEST_FILENAME


def save_manifest(run_dir: Path, m: RunManifest) -> None:
    """Atomic write: .tmp → os.replace."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    m.last_updated = _now_iso()
    target = _path(run_dir)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(m.to_dict(), indent=2, sort_keys=True))
    os.replace(tmp, target)


def load_manifest(run_dir: Path) -> RunManifest:
    p = _path(run_dir)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found at {p}")
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"corrupt manifest at {p}: {e}") from e
    return RunManifest.from_dict(d)


def mark_step_done(run_dir: Path, step: str, extra: Optional[dict] = None) -> None:
    m = load_manifest(run_dir)
    m.steps[step] = {
        **m.steps.get(step, {}),
        "status": "done",
        "committed_at": _now_iso(),
        **(extra or {}),
    }
    save_manifest(run_dir, m)


def mark_step_in_progress(run_dir: Path, step: str, extra: Optional[dict] = None) -> None:
    m = load_manifest(run_dir)
    cur = m.steps.get(step, {})
    m.steps[step] = {
        **cur,
        "status": "in_progress",
        "attempts": cur.get("attempts", 0) + 1,
        **(extra or {}),
    }
    save_manifest(run_dir, m)


def mark_step_failed(run_dir: Path, step: str, error: str, attempts: int) -> None:
    m = load_manifest(run_dir)
    m.steps[step] = {
        **m.steps.get(step, {}),
        "status": "failed",
        "attempts": attempts,
    }
    m.failures.append({
        "step": step, "error": error,
        "ts": _now_iso(), "attempts": attempts,
    })
    save_manifest(run_dir, m)
```

- [ ] **Step 4: Run tests — must PASS**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_manifest.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/loopvid/manifest.py test_loopvid_manifest.py
git commit -m "feat(loopvid): manifest state machine with atomic writes"
```

---

## Phase 4 — Cost module

### Task 4: Create cost.py with per-endpoint pricing + budget guard

**Files:**
- Create: `scripts/loopvid/cost.py`
- Create: `test_loopvid_cost.py`

- [ ] **Step 1: Write tests**

Create `/root/ace-step-music-xl/test_loopvid_cost.py`:

```python
from scripts.loopvid.cost import (
    estimate_run_cost,
    cost_breakdown_lines,
    BudgetExceededError,
    enforce_budget,
)


def test_estimate_60min_run_is_in_expected_range():
    cost = estimate_run_cost(duration_sec=3600)
    assert 1.5 < cost < 6.0   # generous range — adjust as pricing changes


def test_estimate_only_remaining_steps():
    full = estimate_run_cost(duration_sec=3600)
    only_video_mux = estimate_run_cost(
        duration_sec=3600,
        skip=("plan", "music", "image"),
    )
    assert only_video_mux < full
    assert only_video_mux < 1.0


def test_breakdown_lines_includes_all_paid_steps():
    lines = cost_breakdown_lines(duration_sec=3600)
    body = "\n".join(lines)
    for step in ("LLM", "Image", "Music", "Video"):
        assert step in body


def test_enforce_budget_under_max_passes():
    enforce_budget(estimated=2.50, max_cost=5.00)


def test_enforce_budget_over_max_raises():
    import pytest
    with pytest.raises(BudgetExceededError, match="\\$5"):
        enforce_budget(estimated=6.00, max_cost=5.00)
```

- [ ] **Step 2: Run tests — must FAIL**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_cost.py -v 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement cost.py**

Create `/root/ace-step-music-xl/scripts/loopvid/cost.py`:

```python
"""Cost estimator for the loop_music_video pipeline.

Per-second pricing constants are best-effort; actual cost is recorded into
manifest.json as call durations come back."""
from __future__ import annotations

from typing import Iterable

from scripts.loopvid.constants import (
    SEGMENT_COUNT_60MIN,
    SEGMENT_DURATION_SEC,
    CLIP_COUNT,
    CLIP_DURATION_SEC,
)

# Per-second pricing (USD)
RTX_4090_PER_SEC = 0.00031     # ACE-Step worker
RTX_PRO_6000_PER_SEC = 0.00076  # LTX worker

# Per-call pricing (USD)
SEEDREAM_PER_IMAGE = 0.03
GEMINI_3_FLASH_PER_RUN = 0.001  # ~700-1500 tokens output, generous estimate

# Heuristics for inference time per second of output
ACE_STEP_INFERENCE_SEC_PER_OUTPUT_SEC = 1.6   # 360s output ≈ 575s on RTX 4090
LTX_INFERENCE_SEC_PER_OUTPUT_SEC = 5.0        # 7s output ≈ 35s on RTX Pro 6000


class BudgetExceededError(RuntimeError):
    pass


def _step_costs(duration_sec: int) -> dict:
    music_segments = max(1, SEGMENT_COUNT_60MIN if duration_sec >= 3000 else
                          (duration_sec + SEGMENT_DURATION_SEC - 1) // SEGMENT_DURATION_SEC)
    music_inference_sec = music_segments * SEGMENT_DURATION_SEC * ACE_STEP_INFERENCE_SEC_PER_OUTPUT_SEC
    music_cost = music_inference_sec * RTX_4090_PER_SEC

    video_inference_sec = CLIP_COUNT * CLIP_DURATION_SEC * LTX_INFERENCE_SEC_PER_OUTPUT_SEC
    video_cost = video_inference_sec * RTX_PRO_6000_PER_SEC

    return {
        "plan":  GEMINI_3_FLASH_PER_RUN,
        "image": SEEDREAM_PER_IMAGE,
        "music": music_cost,
        "video": video_cost,
        # ffmpeg steps run locally — no API cost
        "loop_build": 0.0,
        "mux": 0.0,
    }


def estimate_run_cost(duration_sec: int = 3600, skip: Iterable[str] = ()) -> float:
    skipset = set(skip)
    costs = _step_costs(duration_sec)
    return sum(c for step, c in costs.items() if step not in skipset)


def cost_breakdown_lines(duration_sec: int = 3600, skip: Iterable[str] = ()) -> list[str]:
    skipset = set(skip)
    costs = _step_costs(duration_sec)
    labels = {
        "plan":  "LLM (Gemini 3 Flash)",
        "image": "Image (Seedream 4.5)",
        "music": f"Music (ACE-Step, {SEGMENT_COUNT_60MIN} segments)",
        "video": f"Video (LTX, {CLIP_COUNT} clips)",
    }
    out = []
    for step, label in labels.items():
        if step in skipset:
            continue
        out.append(f"  - {label:<35} ${costs[step]:.3f}")
    out.append(f"  TOTAL: ${estimate_run_cost(duration_sec, skip):.2f}")
    return out


def enforce_budget(estimated: float, max_cost: float) -> None:
    if estimated > max_cost:
        raise BudgetExceededError(
            f"Estimated ${estimated:.2f} exceeds --max-cost ${max_cost:.2f}"
        )
```

- [ ] **Step 4: Run tests — must PASS**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_cost.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/loopvid/cost.py test_loopvid_cost.py
git commit -m "feat(loopvid): cost estimator + budget guard"
```

---

## Phase 5 — Plan schema

### Task 5: Create plan_schema.py with Plan dataclass + validator

**Files:**
- Create: `scripts/loopvid/plan_schema.py`
- Create: `test_loopvid_plan_schema.py`

- [ ] **Step 1: Write tests**

Create `/root/ace-step-music-xl/test_loopvid_plan_schema.py`:

```python
import pytest

from scripts.loopvid.plan_schema import Plan, validate_plan_dict, PlanSchemaError


VALID = {
    "genre": "lofi",
    "mood": "rainy night",
    "music_palette": "Lofi instrumental, 70 BPM, vinyl crackle, jazz piano, "
                     "soft rim shot drums, warm tape saturation",
    "music_segment_descriptors": [
        {"phase": f"phase-{i}", "descriptors": f"desc-{i}"} for i in range(1, 12)
    ],
    "music_bpm": 70,
    "seedream_scene": "A wooden desk by a rainy window at dusk, a notebook and "
                      "a small lamp glowing softly",
    "seedream_style": "Shot on 35mm film, Kodak Portra 400, soft warm rim lighting",
    "motion_prompts": [
        "rain begins gently on the window glass, soft drops",
        "rain continues, slight intensification",
        "lamp flicker grows warmer",
        "rain peaks, steady rhythm",
        "rain softens",
        "rain returns to gentle drops, lamp settles",
    ],
    "motion_archetype": "rain",
    "image_archetype_key": "rainy_window_desk",
}


def test_validate_passes_on_valid():
    plan = validate_plan_dict(VALID)
    assert isinstance(plan, Plan)
    assert plan.genre == "lofi"
    assert len(plan.motion_prompts) == 6


def test_motion_prompts_must_be_exactly_6():
    bad = {**VALID, "motion_prompts": VALID["motion_prompts"][:5]}
    with pytest.raises(PlanSchemaError, match="motion_prompts"):
        validate_plan_dict(bad)


def test_music_segment_descriptors_must_be_exactly_11():
    bad = {**VALID, "music_segment_descriptors": VALID["music_segment_descriptors"][:10]}
    with pytest.raises(PlanSchemaError, match="11"):
        validate_plan_dict(bad)


def test_image_archetype_must_be_from_allowed_set():
    bad = {**VALID, "image_archetype_key": "made_up_archetype"}
    with pytest.raises(PlanSchemaError, match="archetype"):
        validate_plan_dict(bad)


def test_missing_required_field_raises():
    bad = {k: v for k, v in VALID.items() if k != "music_palette"}
    with pytest.raises(PlanSchemaError, match="music_palette"):
        validate_plan_dict(bad)


def test_motion_archetype_must_be_from_allowed_set():
    bad = {**VALID, "motion_archetype": "unicorn"}
    with pytest.raises(PlanSchemaError, match="motion_archetype"):
        validate_plan_dict(bad)
```

- [ ] **Step 2: Run tests — must FAIL**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_plan_schema.py -v 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement plan_schema.py**

Create `/root/ace-step-music-xl/scripts/loopvid/plan_schema.py`:

```python
"""Plan dataclass + validator for the LLM Planner output."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scripts.loopvid.constants import GENRE_ARCHETYPES, SEGMENT_COUNT_60MIN, CLIP_COUNT

ALLOWED_MOTION_ARCHETYPES = {"rain", "candle", "mist", "smoke", "dust", "snow"}


class PlanSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Plan:
    genre: str
    mood: str
    music_palette: str
    music_segment_descriptors: list
    music_bpm: int
    seedream_scene: str
    seedream_style: str
    motion_prompts: list
    motion_archetype: str
    image_archetype_key: str


REQUIRED_FIELDS = {
    "genre": str,
    "mood": str,
    "music_palette": str,
    "music_segment_descriptors": list,
    "music_bpm": int,
    "seedream_scene": str,
    "seedream_style": str,
    "motion_prompts": list,
    "motion_archetype": str,
    "image_archetype_key": str,
}


def validate_plan_dict(d: dict) -> Plan:
    for name, expected_type in REQUIRED_FIELDS.items():
        if name not in d:
            raise PlanSchemaError(f"missing required field: {name}")
        if not isinstance(d[name], expected_type):
            raise PlanSchemaError(
                f"field {name} must be {expected_type.__name__}, got {type(d[name]).__name__}"
            )

    if len(d["music_segment_descriptors"]) != SEGMENT_COUNT_60MIN:
        raise PlanSchemaError(
            f"music_segment_descriptors must have exactly {SEGMENT_COUNT_60MIN} entries, "
            f"got {len(d['music_segment_descriptors'])}"
        )
    for i, seg in enumerate(d["music_segment_descriptors"]):
        if not isinstance(seg, dict) or "phase" not in seg or "descriptors" not in seg:
            raise PlanSchemaError(
                f"music_segment_descriptors[{i}] must have 'phase' and 'descriptors' keys"
            )

    if len(d["motion_prompts"]) != CLIP_COUNT:
        raise PlanSchemaError(
            f"motion_prompts must have exactly {CLIP_COUNT} entries, "
            f"got {len(d['motion_prompts'])}"
        )

    if d["image_archetype_key"] not in GENRE_ARCHETYPES:
        raise PlanSchemaError(
            f"image_archetype_key '{d['image_archetype_key']}' not in allowed set "
            f"{sorted(GENRE_ARCHETYPES.keys())}"
        )

    if d["motion_archetype"] not in ALLOWED_MOTION_ARCHETYPES:
        raise PlanSchemaError(
            f"motion_archetype '{d['motion_archetype']}' not in allowed set "
            f"{sorted(ALLOWED_MOTION_ARCHETYPES)}"
        )

    return Plan(
        genre=d["genre"],
        mood=d["mood"],
        music_palette=d["music_palette"],
        music_segment_descriptors=list(d["music_segment_descriptors"]),
        music_bpm=int(d["music_bpm"]),
        seedream_scene=d["seedream_scene"],
        seedream_style=d["seedream_style"],
        motion_prompts=list(d["motion_prompts"]),
        motion_archetype=d["motion_archetype"],
        image_archetype_key=d["image_archetype_key"],
    )
```

- [ ] **Step 4: Run tests — must PASS**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_plan_schema.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/loopvid/plan_schema.py test_loopvid_plan_schema.py
git commit -m "feat(loopvid): Plan dataclass + JSON schema validator"
```

---

## Phase 6 — LLM Planner (continues in next sub-document)

> **Note for engineer:** The remaining phases (LLM Planner, Image/Video/Music pipelines, Loop Build, Mux, Preflight, Orchestrator, Rollback CLI, E2E test) follow the identical pattern: TDD test file → run/fail → implement → run/pass → commit. See the next plan document `2026-04-27-loop-music-video-orchestrator-part2.md` for tasks 6-17.

This document was split at Phase 6 to keep each plan file under ~1500 lines. Tasks completed in this document:

- ✅ Phase 1: runpod_client extracted (Task 1)
- ✅ Phase 2: constants module (Task 2)
- ✅ Phase 3: manifest module (Task 3)
- ✅ Phase 4: cost module (Task 4)
- ✅ Phase 5: plan_schema module (Task 5)

After completing tasks 1-5 above, proceed to part 2.

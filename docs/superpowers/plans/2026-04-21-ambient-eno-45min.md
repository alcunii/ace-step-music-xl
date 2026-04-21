# 45-minute Eno-style ambient orchestrator — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/ambient_eno_45min.py`, a self-contained local orchestrator that submits 7 `text2music` segments to the existing RunPod endpoint `nwqnd0duxc6o38`, saves each as FLAC with a JSON sidecar, and stitches them into one ~46-minute FLAC via ffmpeg `acrossfade`. Resumable, pinnable, idempotent per-segment.

**Architecture:** Single Python orchestrator script + single pytest test file at repo root, matching the existing `scripts/*.py` + `test_*.py` convention. Sequential RunPod submission (one warm worker), per-segment cache on disk (so failures resume cleanly), ffmpeg `acrossfade` chain with equal-power qsin curves for the final stitch. No changes to `handler.py`, Dockerfile, or `.github/workflows/*`.

**Tech Stack:** Python 3 stdlib + `requests` (already a handler dep) for RunPod HTTP + `subprocess` → `ffmpeg` 6.x (verified installed at `/usr/bin/ffmpeg`). Tests use `pytest` + `responses` (both already in `requirements-test.txt`).

**Spec:** `docs/superpowers/specs/2026-04-21-ambient-eno-45min-design.md` (commit `fcb3c9e`).

---

## File structure

New files:
- `scripts/ambient_eno_45min.py` — orchestrator. One file, ~500 lines. Mimics layout of `scripts/bruno_mars_style_midnight_gold.py`: constants at top, main() at bottom, helpers in between.
- `test_ambient_eno_45min.py` — pytest test file at repo root (matches `test_handler.py` / `test_workflow.py`).

Modified files:
- `README.md` — one-paragraph entry pointing to the new script (last task).

No modifications to `handler.py`, `Dockerfile`, `.github/workflows/deploy.yml`, or any existing test.

**Why one file, not a sub-package:** The project's existing scripts (`scripts/bruno_mars_style_midnight_gold.py` 266 lines; `scripts/cover_afterlife_armstrong.py` ~325 lines) are monolithic; `handler.py` itself is 544 lines. One orchestrator at ~500 lines matches the house style. Splitting into `scripts/ambient/*.py` sub-modules would require `__init__.py` / `sys.path` gymnastics with no clarity win.

**Why `requests` not stdlib `urllib`:** `requests` is already imported by `handler.py:22` (a project dep), and `responses` in `requirements-test.txt` mocks `requests` directly — making the retry logic clean to unit-test. The Bruno Mars script uses `urllib` because it was zero-dep, but our orchestrator has meaningful retry behaviour that we want to cover with tests.

---

## Task 1: Scaffold + constants

**Files:**
- Create: `scripts/ambient_eno_45min.py`
- Test: `test_ambient_eno_45min.py`

- [ ] **Step 1.1: Write the failing test**

Create `test_ambient_eno_45min.py`:

```python
"""Unit tests for scripts/ambient_eno_45min.py orchestrator."""
import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent / "scripts" / "ambient_eno_45min.py"


def _load():
    """Import the script as a module. Does NOT execute main()."""
    spec = importlib.util.spec_from_file_location("ambient_eno_45min", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ambient_eno_45min"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestConstants:
    def test_locked_palette_is_non_empty_string(self):
        m = _load()
        assert isinstance(m.LOCKED_PALETTE, str)
        assert "Eno" in m.LOCKED_PALETTE or "tonal ambient" in m.LOCKED_PALETTE
        assert len(m.LOCKED_PALETTE) > 200

    def test_segment_descriptors_has_seven_entries(self):
        m = _load()
        assert len(m.SEGMENT_DESCRIPTORS) == 7
        phases = [s["phase"] for s in m.SEGMENT_DESCRIPTORS]
        assert phases == [
            "Inhale-1", "Inhale-2", "Inhale-3", "Turn",
            "Exhale-1", "Exhale-2", "Dissolve",
        ]
        for s in m.SEGMENT_DESCRIPTORS:
            assert isinstance(s["descriptors"], str) and s["descriptors"]

    def test_preset_matches_official_xl_base_values(self):
        m = _load()
        # Verbatim from docs/en/INFERENCE.md Example 9.
        assert m.PRESET["inference_steps"] == 64
        assert m.PRESET["guidance_scale"] == 8.0
        assert m.PRESET["shift"] == 3.0
        assert m.PRESET["use_adg"] is True
        assert m.PRESET["cfg_interval_start"] == 0.0
        assert m.PRESET["cfg_interval_end"] == 1.0
        assert m.PRESET["infer_method"] == "ode"

    def test_per_segment_duration_is_420(self):
        m = _load()
        assert m.SEGMENT_DURATION_SEC == 420

    def test_crossfade_is_30s(self):
        m = _load()
        assert m.CROSSFADE_SEC == 30

    def test_segment_count_is_seven(self):
        m = _load()
        assert m.SEGMENT_COUNT == 7
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
cd /root/ace-step-music-xl
pytest test_ambient_eno_45min.py -v
```
Expected: all tests FAIL with "FileNotFoundError" (script doesn't exist yet).

- [ ] **Step 1.3: Create the script scaffold**

Create `scripts/ambient_eno_45min.py`:

```python
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


if __name__ == "__main__":  # pragma: no cover
    sys.exit("main() not yet implemented (Task 10)")
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
pytest test_ambient_eno_45min.py::TestConstants -v
```
Expected: 6 PASS.

- [ ] **Step 1.5: Commit**

```bash
git add scripts/ambient_eno_45min.py test_ambient_eno_45min.py
git commit -m "feat(scripts): scaffold ambient orchestrator constants + preset"
```

---

## Task 2: Prompt and payload builders

**Files:**
- Modify: `scripts/ambient_eno_45min.py` (add `build_segment_prompt` and `build_payload`)
- Modify: `test_ambient_eno_45min.py`

- [ ] **Step 2.1: Write the failing tests**

Append to `test_ambient_eno_45min.py`:

```python
class TestPromptBuilder:
    def test_prompt_contains_locked_palette(self):
        m = _load()
        for n in range(1, 8):
            p = m.build_segment_prompt(n)
            assert m.LOCKED_PALETTE in p

    def test_prompt_contains_correct_descriptor(self):
        m = _load()
        for n in range(1, 8):
            p = m.build_segment_prompt(n)
            expected = m.SEGMENT_DESCRIPTORS[n - 1]["descriptors"]
            assert expected in p

    def test_prompt_rejects_out_of_range_segment(self):
        m = _load()
        import pytest
        with pytest.raises(ValueError):
            m.build_segment_prompt(0)
        with pytest.raises(ValueError):
            m.build_segment_prompt(8)


class TestPayloadBuilder:
    def test_payload_has_all_preset_keys(self):
        m = _load()
        payload = m.build_payload(segment_num=1, duration=420, seed=-1)
        for k, v in m.PRESET.items():
            assert payload[k] == v, f"{k}: expected {v}, got {payload[k]}"

    def test_payload_task_type_is_text2music(self):
        m = _load()
        assert m.build_payload(1, 420, -1)["task_type"] == "text2music"

    def test_payload_is_instrumental(self):
        m = _load()
        assert m.build_payload(1, 420, -1)["instrumental"] is True

    def test_payload_thinking_is_false(self):
        m = _load()
        assert m.build_payload(1, 420, -1)["thinking"] is False

    def test_payload_audio_format_is_flac(self):
        m = _load()
        assert m.build_payload(1, 420, -1)["audio_format"] == "flac"

    def test_payload_batch_size_is_one(self):
        m = _load()
        assert m.build_payload(1, 420, -1)["batch_size"] == 1

    def test_payload_duration_and_seed_respected(self):
        m = _load()
        p = m.build_payload(3, duration=300, seed=12345)
        assert p["duration"] == 300
        assert p["seed"] == 12345

    def test_payload_prompt_matches_builder(self):
        m = _load()
        p = m.build_payload(4, 420, -1)
        assert p["prompt"] == m.build_segment_prompt(4)
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
pytest test_ambient_eno_45min.py::TestPromptBuilder test_ambient_eno_45min.py::TestPayloadBuilder -v
```
Expected: FAIL with "AttributeError: module 'ambient_eno_45min' has no attribute 'build_segment_prompt'".

- [ ] **Step 2.3: Implement**

Insert after the `PRESET` constant block in `scripts/ambient_eno_45min.py`:

```python
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
```

- [ ] **Step 2.4: Run tests to verify they pass**

```bash
pytest test_ambient_eno_45min.py -v
```
Expected: all TestConstants + TestPromptBuilder + TestPayloadBuilder PASS (16+ tests).

- [ ] **Step 2.5: Commit**

```bash
git add scripts/ambient_eno_45min.py test_ambient_eno_45min.py
git commit -m "feat(scripts): prompt and payload builders for ambient orchestrator"
```

---

## Task 3: JSON sidecar + manifest I/O

**Files:**
- Modify: `scripts/ambient_eno_45min.py` (add `write_sidecar`, `read_sidecar`, `write_manifest`)
- Modify: `test_ambient_eno_45min.py`

- [ ] **Step 3.1: Write the failing tests**

Append to `test_ambient_eno_45min.py`:

```python
class TestSidecar:
    def test_sidecar_roundtrip(self, tmp_path):
        m = _load()
        path = tmp_path / "segment_03.json"
        data = {
            "segment_num": 3,
            "phase": "Inhale-3",
            "seed": 42,
            "prompt": "test prompt",
            "duration_requested": 420,
            "duration_actual": 420.5,
            "sample_rate": 48000,
            "endpoint_id": "nwqnd0duxc6o38",
            "run_id": "2026-04-21-1234",
        }
        m.write_sidecar(path, data)
        got = m.read_sidecar(path)
        assert got == data

    def test_sidecar_is_human_readable(self, tmp_path):
        m = _load()
        path = tmp_path / "s.json"
        m.write_sidecar(path, {"seed": 7, "phase": "Turn"})
        text = path.read_text()
        # pretty-printed (indented) for diff-friendliness
        assert "\n" in text
        assert '"seed": 7' in text


class TestManifest:
    def test_manifest_contains_required_fields(self, tmp_path):
        m = _load()
        manifest_path = tmp_path / "manifest.json"
        m.write_manifest(
            path=manifest_path,
            run_id="2026-04-21-1234",
            endpoint_id="nwqnd0duxc6o38",
            seeds=[1, 2, 3, 4, 5, 6, 7],
            segment_duration=420,
            crossfade_sec=30,
            locked_palette="x",
        )
        got = json.loads(manifest_path.read_text())
        assert got["run_id"] == "2026-04-21-1234"
        assert got["endpoint_id"] == "nwqnd0duxc6o38"
        assert got["seeds"] == [1, 2, 3, 4, 5, 6, 7]
        assert got["segment_count"] == 7
        assert got["segment_duration_sec"] == 420
        assert got["crossfade_sec"] == 30
        assert got["locked_palette"] == "x"
        assert "written_at" in got  # ISO8601 timestamp
```

Also add `import json` at the top of the test file if not already present.

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
pytest test_ambient_eno_45min.py::TestSidecar test_ambient_eno_45min.py::TestManifest -v
```
Expected: FAIL with AttributeError on `write_sidecar`.

- [ ] **Step 3.3: Implement**

Insert into `scripts/ambient_eno_45min.py` after the payload builder:

```python
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
```

- [ ] **Step 3.4: Run tests to verify they pass**

```bash
pytest test_ambient_eno_45min.py -v
```
Expected: all prior PASS + TestSidecar + TestManifest PASS.

- [ ] **Step 3.5: Commit**

```bash
git add scripts/ambient_eno_45min.py test_ambient_eno_45min.py
git commit -m "feat(scripts): JSON sidecar and manifest I/O for ambient orchestrator"
```

---

## Task 4: RunPod submit + poll (with 404 tolerance)

**Files:**
- Modify: `scripts/ambient_eno_45min.py` (add `submit_job`, `poll_job`)
- Modify: `test_ambient_eno_45min.py`

- [ ] **Step 4.1: Write the failing tests**

Append to `test_ambient_eno_45min.py`:

```python
class TestRunPodClient:
    def test_submit_job_posts_to_runsync(self):
        m = _load()
        import responses
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "https://api.runpod.ai/v2/EP/runsync",
                json={"id": "abc123", "status": "IN_QUEUE"},
                status=200,
            )
            body = m.submit_job("EP", "key", {"input": {}})
            assert body["id"] == "abc123"
            assert body["status"] == "IN_QUEUE"
            assert len(rsps.calls) == 1
            assert "Bearer key" in rsps.calls[0].request.headers["Authorization"]

    def test_submit_job_returns_output_on_sync_complete(self):
        m = _load()
        import responses
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "https://api.runpod.ai/v2/EP/runsync",
                json={"id": "j1", "status": "COMPLETED",
                      "output": {"audio_base64": "AAA"}},
                status=200,
            )
            body = m.submit_job("EP", "key", {"input": {}})
            assert body["status"] == "COMPLETED"
            assert body["output"]["audio_base64"] == "AAA"

    def test_poll_job_returns_completed_output(self):
        m = _load()
        import responses
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "https://api.runpod.ai/v2/EP/status/job1",
                json={"status": "COMPLETED", "output": {"audio_base64": "AAA"}},
                status=200,
            )
            result = m.poll_job("EP", "key", "job1", poll_interval=0)
            assert result["status"] == "COMPLETED"
            assert result["output"]["audio_base64"] == "AAA"

    def test_poll_job_tolerates_transient_404s(self):
        m = _load()
        import responses
        with responses.RequestsMock() as rsps:
            rsps.add(responses.GET,
                     "https://api.runpod.ai/v2/EP/status/j", status=404)
            rsps.add(responses.GET,
                     "https://api.runpod.ai/v2/EP/status/j", status=404)
            rsps.add(responses.GET,
                     "https://api.runpod.ai/v2/EP/status/j",
                     json={"status": "COMPLETED", "output": {"x": 1}},
                     status=200)
            result = m.poll_job("EP", "key", "j", poll_interval=0)
            assert result["status"] == "COMPLETED"
            assert len(rsps.calls) == 3

    def test_poll_job_gives_up_after_too_many_404s(self):
        m = _load()
        import responses
        import pytest
        with responses.RequestsMock() as rsps:
            # MAX_TRANSIENT_404 + 1 consecutive 404s
            for _ in range(m.MAX_TRANSIENT_404 + 1):
                rsps.add(responses.GET,
                         "https://api.runpod.ai/v2/EP/status/j",
                         status=404)
            with pytest.raises(RuntimeError, match="404"):
                m.poll_job("EP", "key", "j", poll_interval=0)

    def test_poll_job_raises_on_terminal_failure(self):
        m = _load()
        import responses
        import pytest
        with responses.RequestsMock() as rsps:
            rsps.add(responses.GET,
                     "https://api.runpod.ai/v2/EP/status/j",
                     json={"status": "FAILED", "error": "oom"},
                     status=200)
            with pytest.raises(RuntimeError, match="FAILED"):
                m.poll_job("EP", "key", "j", poll_interval=0)
```

- [ ] **Step 4.2: Run tests to verify they fail**

```bash
pytest test_ambient_eno_45min.py::TestRunPodClient -v
```
Expected: FAIL — `submit_job` / `poll_job` don't exist.

- [ ] **Step 4.3: Implement**

Insert into `scripts/ambient_eno_45min.py` after the sidecar helpers:

```python
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
```

- [ ] **Step 4.4: Run tests to verify they pass**

```bash
pytest test_ambient_eno_45min.py -v
```
Expected: all prior PASS + TestRunPodClient (5 tests) PASS.

- [ ] **Step 4.5: Commit**

```bash
git add scripts/ambient_eno_45min.py test_ambient_eno_45min.py
git commit -m "feat(scripts): RunPod submit/poll client with transient 404 tolerance"
```

---

## Task 5: Whole-segment retry wrapper

**Files:**
- Modify: `scripts/ambient_eno_45min.py` (add `run_segment`)
- Modify: `test_ambient_eno_45min.py`

- [ ] **Step 5.1: Write the failing tests**

Append to `test_ambient_eno_45min.py`:

```python
class TestRunSegment:
    def _mock_runsync_completed(self, rsps, endpoint_id, audio_b64):
        rsps.add(
            responses.POST,
            f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
            json={"id": "j1", "status": "COMPLETED",
                  "output": {"audio_base64": audio_b64, "duration": 420.0,
                             "sample_rate": 48000, "seed": 777}},
            status=200,
        )

    def test_run_segment_success_first_try(self):
        m = _load()
        import responses
        b64 = "ZmxhYw=="  # "flac"
        with responses.RequestsMock() as rsps:
            self._mock_runsync_completed(rsps, "EP", b64)
            result = m.run_segment(
                endpoint_id="EP", api_key="key", segment_num=2,
                duration=420, seed=-1, poll_interval=0,
            )
        assert result["output"]["audio_base64"] == b64
        assert result["output"]["seed"] == 777

    def test_run_segment_retries_on_transient_failure(self, monkeypatch):
        m = _load()
        import responses
        # First 2 runsync calls raise a ConnectionError, 3rd succeeds.
        with responses.RequestsMock() as rsps:
            rsps.add(responses.POST,
                     "https://api.runpod.ai/v2/EP/runsync",
                     body=requests.exceptions.ConnectionError("boom"))
            rsps.add(responses.POST,
                     "https://api.runpod.ai/v2/EP/runsync",
                     body=requests.exceptions.ConnectionError("boom"))
            rsps.add(responses.POST,
                     "https://api.runpod.ai/v2/EP/runsync",
                     json={"id": "j1", "status": "COMPLETED",
                           "output": {"audio_base64": "AAA", "duration": 420}},
                     status=200)
            result = m.run_segment(
                endpoint_id="EP", api_key="key", segment_num=1,
                duration=420, seed=-1, poll_interval=0, retry_sleep=0,
            )
        assert result["output"]["audio_base64"] == "AAA"
        assert len(rsps.calls) == 3

    def test_run_segment_gives_up_after_max_retries(self):
        m = _load()
        import responses
        import pytest
        with responses.RequestsMock() as rsps:
            for _ in range(m.MAX_SEGMENT_RETRIES):
                rsps.add(responses.POST,
                         "https://api.runpod.ai/v2/EP/runsync",
                         body=requests.exceptions.ConnectionError("boom"))
            with pytest.raises(RuntimeError, match="segment 4"):
                m.run_segment(
                    endpoint_id="EP", api_key="key", segment_num=4,
                    duration=420, seed=-1, poll_interval=0, retry_sleep=0,
                )
```

Also add `import responses` and `import requests` at the top of the test file if missing.

- [ ] **Step 5.2: Run tests to verify they fail**

```bash
pytest test_ambient_eno_45min.py::TestRunSegment -v
```
Expected: FAIL — `run_segment` not defined.

- [ ] **Step 5.3: Implement**

Insert into `scripts/ambient_eno_45min.py` after `poll_job`:

```python
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
```

- [ ] **Step 5.4: Run tests to verify they pass**

```bash
pytest test_ambient_eno_45min.py -v
```
Expected: all prior PASS + TestRunSegment (3 tests) PASS.

- [ ] **Step 5.5: Commit**

```bash
git add scripts/ambient_eno_45min.py test_ambient_eno_45min.py
git commit -m "feat(scripts): whole-segment retry wrapper with bounded attempts"
```

---

## Task 6: FLAC save from response

**Files:**
- Modify: `scripts/ambient_eno_45min.py` (add `save_flac_from_output`)
- Modify: `test_ambient_eno_45min.py`

- [ ] **Step 6.1: Write the failing tests**

Append to `test_ambient_eno_45min.py`:

```python
class TestSaveFlac:
    def test_save_writes_decoded_bytes(self, tmp_path):
        m = _load()
        raw = b"notactualflacbutfine"
        output = {"audio_base64": base64.b64encode(raw).decode()}
        path = tmp_path / "s01.flac"
        m.save_flac_from_output(output, path)
        assert path.read_bytes() == raw

    def test_save_creates_parent_dir(self, tmp_path):
        m = _load()
        output = {"audio_base64": base64.b64encode(b"x").decode()}
        path = tmp_path / "deep" / "nested" / "s.flac"
        m.save_flac_from_output(output, path)
        assert path.exists()

    def test_save_raises_on_missing_audio(self, tmp_path):
        m = _load()
        import pytest
        with pytest.raises(ValueError, match="audio_base64"):
            m.save_flac_from_output({}, tmp_path / "x.flac")

    def test_save_raises_on_empty_audio(self, tmp_path):
        m = _load()
        import pytest
        with pytest.raises(ValueError, match="empty"):
            m.save_flac_from_output({"audio_base64": ""}, tmp_path / "x.flac")
```

Also add `import base64` at top of test file.

- [ ] **Step 6.2: Run tests to verify they fail**

```bash
pytest test_ambient_eno_45min.py::TestSaveFlac -v
```
Expected: FAIL — `save_flac_from_output` undefined.

- [ ] **Step 6.3: Implement**

Insert into `scripts/ambient_eno_45min.py` after `run_segment`:

```python
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
```

- [ ] **Step 6.4: Run tests to verify they pass**

```bash
pytest test_ambient_eno_45min.py -v
```
Expected: all PASS, including TestSaveFlac (4 tests).

- [ ] **Step 6.5: Commit**

```bash
git add scripts/ambient_eno_45min.py test_ambient_eno_45min.py
git commit -m "feat(scripts): save FLAC bytes from RunPod output"
```

---

## Task 7: ffmpeg crossfade command builder

**Files:**
- Modify: `scripts/ambient_eno_45min.py` (add `build_ffmpeg_command`, `stitch_segments`)
- Modify: `test_ambient_eno_45min.py`

- [ ] **Step 7.1: Write the failing tests**

Append to `test_ambient_eno_45min.py`:

```python
class TestFFmpegCommand:
    def test_command_lists_all_inputs(self, tmp_path):
        m = _load()
        paths = [tmp_path / f"s{i:02d}.flac" for i in range(1, 8)]
        cmd = m.build_ffmpeg_command(paths, tmp_path / "out.flac",
                                      crossfade_sec=30)
        # -i appears once per input
        assert cmd.count("-i") == 7
        for p in paths:
            assert str(p) in cmd

    def test_command_uses_qsin_equal_power_curves(self, tmp_path):
        m = _load()
        paths = [tmp_path / f"s{i:02d}.flac" for i in range(1, 8)]
        cmd = m.build_ffmpeg_command(paths, tmp_path / "out.flac", 30)
        filter_str = cmd[cmd.index("-filter_complex") + 1]
        assert "c1=qsin" in filter_str
        assert "c2=qsin" in filter_str
        assert "d=30" in filter_str

    def test_command_chains_six_crossfades_for_seven_inputs(self, tmp_path):
        m = _load()
        paths = [tmp_path / f"s{i:02d}.flac" for i in range(1, 8)]
        cmd = m.build_ffmpeg_command(paths, tmp_path / "out.flac", 30)
        filter_str = cmd[cmd.index("-filter_complex") + 1]
        assert filter_str.count("acrossfade=") == 6

    def test_command_outputs_flac_codec(self, tmp_path):
        m = _load()
        paths = [tmp_path / f"s{i:02d}.flac" for i in range(1, 8)]
        cmd = m.build_ffmpeg_command(paths, tmp_path / "out.flac", 30)
        # -c:a flac must appear
        i = cmd.index("-c:a")
        assert cmd[i + 1] == "flac"

    def test_command_rejects_fewer_than_two_inputs(self, tmp_path):
        m = _load()
        import pytest
        with pytest.raises(ValueError):
            m.build_ffmpeg_command([tmp_path / "s.flac"], tmp_path / "o.flac", 30)

    def test_stitch_segments_invokes_ffmpeg(self, tmp_path, monkeypatch):
        m = _load()
        calls = []
        def fake_run(cmd, check, capture_output):
            calls.append(cmd)
            # Create an empty output file so post-checks succeed.
            Path(cmd[cmd.index("-map") + 2 if False else -1]).write_bytes(b"")
            class R:
                returncode = 0
                stdout = b""
                stderr = b""
            return R()
        monkeypatch.setattr(m.subprocess, "run", fake_run)
        paths = [tmp_path / f"s{i:02d}.flac" for i in range(1, 8)]
        for p in paths:
            p.write_bytes(b"x")
        out = tmp_path / "final.flac"
        m.stitch_segments(paths, out, crossfade_sec=30)
        assert len(calls) == 1
        assert "ffmpeg" in calls[0][0]
```

- [ ] **Step 7.2: Run tests to verify they fail**

```bash
pytest test_ambient_eno_45min.py::TestFFmpegCommand -v
```
Expected: FAIL — `build_ffmpeg_command` undefined.

- [ ] **Step 7.3: Implement**

Insert into `scripts/ambient_eno_45min.py` after `save_flac_from_output`:

```python
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
```

- [ ] **Step 7.4: Run tests to verify they pass**

```bash
pytest test_ambient_eno_45min.py -v
```
Expected: all PASS including TestFFmpegCommand (6 tests).

- [ ] **Step 7.5: Commit**

```bash
git add scripts/ambient_eno_45min.py test_ambient_eno_45min.py
git commit -m "feat(scripts): ffmpeg crossfade stitcher with qsin equal-power curves"
```

---

## Task 8: Pre-flight checks

**Files:**
- Modify: `scripts/ambient_eno_45min.py` (add `preflight_checks`)
- Modify: `test_ambient_eno_45min.py`

- [ ] **Step 8.1: Write the failing tests**

Append to `test_ambient_eno_45min.py`:

```python
class TestPreflight:
    def test_passes_when_all_deps_present(self, tmp_path, monkeypatch):
        m = _load()
        monkeypatch.setattr(m.shutil, "which",
                            lambda x: "/usr/bin/ffmpeg" if x == "ffmpeg" else None)
        # should not raise
        m.preflight_checks(api_key="k", endpoint_id="EP", out_dir=tmp_path)

    def test_fails_when_api_key_missing(self, tmp_path):
        m = _load()
        import pytest
        with pytest.raises(RuntimeError, match="RUNPOD_API_KEY"):
            m.preflight_checks(api_key="", endpoint_id="EP", out_dir=tmp_path)

    def test_fails_when_endpoint_missing(self, tmp_path):
        m = _load()
        import pytest
        with pytest.raises(RuntimeError, match="endpoint"):
            m.preflight_checks(api_key="k", endpoint_id="", out_dir=tmp_path)

    def test_fails_when_ffmpeg_missing(self, tmp_path, monkeypatch):
        m = _load()
        import pytest
        monkeypatch.setattr(m.shutil, "which", lambda x: None)
        with pytest.raises(RuntimeError, match="ffmpeg"):
            m.preflight_checks(api_key="k", endpoint_id="EP", out_dir=tmp_path)
```

- [ ] **Step 8.2: Run tests to verify they fail**

```bash
pytest test_ambient_eno_45min.py::TestPreflight -v
```
Expected: FAIL — `preflight_checks` undefined.

- [ ] **Step 8.3: Implement**

Insert into `scripts/ambient_eno_45min.py` after the ffmpeg stitcher:

```python
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
```

- [ ] **Step 8.4: Run tests to verify they pass**

```bash
pytest test_ambient_eno_45min.py -v
```
Expected: all PASS including TestPreflight (4 tests).

- [ ] **Step 8.5: Commit**

```bash
git add scripts/ambient_eno_45min.py test_ambient_eno_45min.py
git commit -m "feat(scripts): preflight checks for API key, endpoint, ffmpeg, out dir"
```

---

## Task 9: CLI args + run-dir resolution

**Files:**
- Modify: `scripts/ambient_eno_45min.py` (add `parse_args`, `resolve_run_dir`, `segment_paths_for`, `load_pinned_seeds`)
- Modify: `test_ambient_eno_45min.py`

- [ ] **Step 9.1: Write the failing tests**

Append to `test_ambient_eno_45min.py`:

```python
class TestCLI:
    def test_defaults(self):
        m = _load()
        args = m.parse_args([])
        assert args.run_id is None  # resolved to YYYY-MM-DD-HHMM later
        assert args.force is False
        assert args.segment is None
        assert args.stitch_only is False
        assert args.dry_run is False
        assert args.pin_seeds_from is None
        assert args.duration == m.SEGMENT_DURATION_SEC

    def test_all_flags(self):
        m = _load()
        args = m.parse_args([
            "--run-id", "run1", "--force", "--segment", "3",
            "--stitch-only", "--dry-run",
            "--pin-seeds-from", "run0", "--duration", "60",
        ])
        assert args.run_id == "run1"
        assert args.force is True
        assert args.segment == 3
        assert args.stitch_only is True
        assert args.dry_run is True
        assert args.pin_seeds_from == "run0"
        assert args.duration == 60

    def test_segment_flag_rejects_out_of_range(self):
        m = _load()
        import pytest
        with pytest.raises(SystemExit):
            m.parse_args(["--segment", "8"])
        with pytest.raises(SystemExit):
            m.parse_args(["--segment", "0"])


class TestRunDir:
    def test_resolve_run_dir_new_format(self, tmp_path):
        m = _load()
        rd = m.resolve_run_dir(base=tmp_path, run_id=None)
        assert rd.parent == tmp_path
        # Format: YYYY-MM-DD-HHMM (16 chars)
        assert len(rd.name) == 16
        assert rd.name[4] == "-" and rd.name[7] == "-" and rd.name[10] == "-"

    def test_resolve_run_dir_reuses_supplied_id(self, tmp_path):
        m = _load()
        rd = m.resolve_run_dir(base=tmp_path, run_id="my-run-42")
        assert rd == tmp_path / "my-run-42"

    def test_segment_paths_for_returns_seven(self, tmp_path):
        m = _load()
        paths = m.segment_paths_for(tmp_path)
        assert len(paths) == 7
        assert paths[0].name == "segment_01.flac"
        assert paths[6].name == "segment_07.flac"

    def test_load_pinned_seeds_from_prior_run(self, tmp_path):
        m = _load()
        prior = tmp_path / "old"
        prior.mkdir()
        for i, seed in enumerate([1, 2, 3, 4, 5, 6, 7], start=1):
            m.write_sidecar(prior / f"segment_{i:02d}.json", {"seed": seed})
        seeds = m.load_pinned_seeds(prior)
        assert seeds == [1, 2, 3, 4, 5, 6, 7]

    def test_load_pinned_seeds_missing_file_errors(self, tmp_path):
        m = _load()
        import pytest
        with pytest.raises(FileNotFoundError):
            m.load_pinned_seeds(tmp_path / "does-not-exist")
```

- [ ] **Step 9.2: Run tests to verify they fail**

```bash
pytest test_ambient_eno_45min.py::TestCLI test_ambient_eno_45min.py::TestRunDir -v
```
Expected: FAIL — `parse_args` undefined.

- [ ] **Step 9.3: Implement**

Insert into `scripts/ambient_eno_45min.py` after `preflight_checks`:

```python
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
                   help="Print the 7 payloads without submitting.")
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
```

- [ ] **Step 9.4: Run tests to verify they pass**

```bash
pytest test_ambient_eno_45min.py -v
```
Expected: all PASS including TestCLI + TestRunDir.

- [ ] **Step 9.5: Commit**

```bash
git add scripts/ambient_eno_45min.py test_ambient_eno_45min.py
git commit -m "feat(scripts): CLI arg parsing and run-directory helpers"
```

---

## Task 10: Main orchestration (ties everything together)

**Files:**
- Modify: `scripts/ambient_eno_45min.py` (add `main`; replace the scaffold stub)
- Modify: `test_ambient_eno_45min.py`

- [ ] **Step 10.1: Write the failing tests**

Append to `test_ambient_eno_45min.py`:

```python
class TestMain:
    def _mock_runsync(self, rsps, endpoint_id, seed_base):
        """Add 7 mocked /runsync calls returning distinct seeds.

        /runsync returns status=COMPLETED synchronously with output inline,
        so run_segment never calls /status for these — only POSTs mocked.
        """
        for i in range(1, 8):
            rsps.add(
                responses.POST,
                f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
                json={
                    "id": f"job{i}", "status": "COMPLETED",
                    "output": {
                        "audio_base64": base64.b64encode(f"seg{i}".encode()).decode(),
                        "duration": 420.0, "sample_rate": 48000,
                        "seed": seed_base + i,
                    },
                },
                status=200,
            )

    def test_dry_run_submits_nothing(self, tmp_path, monkeypatch, capsys):
        m = _load()
        monkeypatch.setenv("RUNPOD_API_KEY", "k")
        monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "EP")
        monkeypatch.setattr(m, "AMBIENT_OUT_DIR", tmp_path)
        monkeypatch.setattr(m.shutil, "which", lambda x: "/usr/bin/ffmpeg")
        exit_code = m.main(["--dry-run", "--run-id", "dry1"])
        assert exit_code == 0
        out = capsys.readouterr().out
        # 7 payload dumps must appear
        assert out.count('"task_type": "text2music"') == 7
        # No FLAC should have been written
        assert not any((tmp_path / "dry1").glob("segment_*.flac"))

    def test_stitch_only_runs_ffmpeg(self, tmp_path, monkeypatch):
        m = _load()
        monkeypatch.setenv("RUNPOD_API_KEY", "k")
        monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "EP")
        monkeypatch.setattr(m, "AMBIENT_OUT_DIR", tmp_path)
        monkeypatch.setattr(m.shutil, "which", lambda x: "/usr/bin/ffmpeg")
        run_dir = tmp_path / "rs1"
        run_dir.mkdir()
        paths = m.segment_paths_for(run_dir)
        for p in paths:
            p.write_bytes(b"x")  # 7 fake FLACs
        calls = []
        def fake_run(cmd, check, capture_output):
            calls.append(cmd)
            Path(cmd[-1]).write_bytes(b"finalbytes")
            class R:
                returncode = 0; stdout = b""; stderr = b""
            return R()
        monkeypatch.setattr(m.subprocess, "run", fake_run)
        exit_code = m.main(["--stitch-only", "--run-id", "rs1"])
        assert exit_code == 0
        assert len(calls) == 1
        final = run_dir / "eno_45min_final.flac"
        assert final.exists()

    def test_full_run_writes_7_segments_and_manifest(self, tmp_path,
                                                     monkeypatch):
        m = _load()
        monkeypatch.setenv("RUNPOD_API_KEY", "k")
        monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "EP")
        monkeypatch.setattr(m, "AMBIENT_OUT_DIR", tmp_path)
        monkeypatch.setattr(m.shutil, "which", lambda x: "/usr/bin/ffmpeg")
        def fake_run(cmd, check, capture_output):
            Path(cmd[-1]).write_bytes(b"final")
            class R:
                returncode = 0; stdout = b""; stderr = b""
            return R()
        monkeypatch.setattr(m.subprocess, "run", fake_run)
        with responses.RequestsMock() as rsps:
            self._mock_runsync(rsps, "EP", seed_base=1000)
            exit_code = m.main(
                ["--run-id", "full1", "--duration", "10"],
            )
        assert exit_code == 0
        run_dir = tmp_path / "full1"
        for i in range(1, 8):
            assert (run_dir / f"segment_{i:02d}.flac").exists()
            assert (run_dir / f"segment_{i:02d}.json").exists()
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "eno_45min_final.flac").exists()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["seeds"] == [1001, 1002, 1003, 1004, 1005, 1006, 1007]

    def test_resume_skips_existing_segments(self, tmp_path, monkeypatch):
        m = _load()
        monkeypatch.setenv("RUNPOD_API_KEY", "k")
        monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "EP")
        monkeypatch.setattr(m, "AMBIENT_OUT_DIR", tmp_path)
        monkeypatch.setattr(m.shutil, "which", lambda x: "/usr/bin/ffmpeg")
        run_dir = tmp_path / "res1"
        run_dir.mkdir()
        # Pre-create segments 1..5 so only 6 + 7 need generating.
        for i in range(1, 6):
            (run_dir / f"segment_{i:02d}.flac").write_bytes(b"pre")
            m.write_sidecar(run_dir / f"segment_{i:02d}.json", {"seed": 100 + i})
        def fake_run(cmd, check, capture_output):
            Path(cmd[-1]).write_bytes(b"final")
            class R:
                returncode = 0; stdout = b""; stderr = b""
            return R()
        monkeypatch.setattr(m.subprocess, "run", fake_run)
        with responses.RequestsMock() as rsps:
            # Only 2 segments should be generated (sync COMPLETED returns
            # output inline — no /status GET needed).
            for i in (6, 7):
                rsps.add(responses.POST,
                         "https://api.runpod.ai/v2/EP/runsync",
                         json={"id": f"j{i}", "status": "COMPLETED",
                               "output": {
                                   "audio_base64": base64.b64encode(b"x").decode(),
                                   "duration": 420, "sample_rate": 48000,
                                   "seed": 100 + i,
                               }},
                         status=200)
            exit_code = m.main(["--run-id", "res1"])
            # Exactly 2 runsync POSTs — one per missing segment.
            runsyncs = [c for c in rsps.calls if c.request.method == "POST"]
            assert len(runsyncs) == 2
        assert exit_code == 0
```

- [ ] **Step 10.2: Run tests to verify they fail**

```bash
pytest test_ambient_eno_45min.py::TestMain -v
```
Expected: FAIL — `main` still raises "not yet implemented".

- [ ] **Step 10.3: Implement `main`**

Replace the `if __name__ == "__main__":` block at the bottom of `scripts/ambient_eno_45min.py` with the following. Also add the `AMBIENT_OUT_DIR` constant near the top (in the constants block, just after `DEFAULT_ENDPOINT_ID`):

```python
AMBIENT_OUT_DIR = Path(__file__).resolve().parent.parent / "out" / "ambient"
```

Then append (bottom of file):

```python
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

    # Dry-run: print payloads, exit.
    if args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, SEGMENT_COUNT + 1):
            seed = pinned_seeds[i - 1] if pinned_seeds else -1
            payload = build_payload(i, args.duration, seed)
            _print(
                f"\n--- segment {i}/{SEGMENT_COUNT} "
                f"({SEGMENT_DESCRIPTORS[i-1]['phase']}) ---"
            )
            _print(json.dumps(payload, indent=2))
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
    targets = [args.segment] if args.segment else list(range(1, SEGMENT_COUNT + 1))

    for i in range(1, SEGMENT_COUNT + 1):
        seg_path = seg_paths[i - 1]
        sidecar_path = run_dir / f"segment_{i:02d}.json"
        if i not in targets and not args.force:
            if seg_path.exists() and sidecar_path.exists():
                seeds_actual[i - 1] = int(read_sidecar(sidecar_path)["seed"])
                _print(f"[{i}/{SEGMENT_COUNT}] skip — already exists")
                continue
        if seg_path.exists() and not args.force and i not in targets:
            seeds_actual[i - 1] = int(read_sidecar(sidecar_path)["seed"])
            _print(f"[{i}/{SEGMENT_COUNT}] skip — already exists")
            continue

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
```

- [ ] **Step 10.4: Run tests to verify they pass**

```bash
pytest test_ambient_eno_45min.py -v
```
Expected: all PASS including TestMain (4 tests).

- [ ] **Step 10.5: Commit**

```bash
git add scripts/ambient_eno_45min.py test_ambient_eno_45min.py
git commit -m "feat(scripts): main() orchestration with dry-run, stitch-only, resume"
```

---

## Task 11: Final polish — README + smoke test

**Files:**
- Modify: `README.md`
- Test: run the complete unit test suite one final time.

- [ ] **Step 11.1: Add a README entry**

Edit `README.md`; add a new section just before `## Environment variables`:

```markdown
## Scripts

Standalone client-side orchestrators for generation patterns beyond a single
`/runsync` call.

- `scripts/bruno_mars_style_midnight_gold.py` — 3-minute original funk-pop
  song via `text2music`.
- `scripts/cover_afterlife_armstrong.py` — reinterpret a local MP3 as a
  different style via `task_type=cover`.
- `scripts/ambient_eno_45min.py` — 7-segment × 420 s Eno-style ambient run
  stitched locally with ffmpeg crossfade to a ~46-minute FLAC. Resumable,
  pinnable via `--pin-seeds-from`, idempotent per segment. See
  `docs/superpowers/specs/2026-04-21-ambient-eno-45min-design.md`.
```

- [ ] **Step 11.2: Run the full test file + a dry-run smoke**

```bash
pytest test_ambient_eno_45min.py -v
python3 scripts/ambient_eno_45min.py --dry-run --run-id smoke-dryrun
```

Expected:
- All pytest tests PASS.
- The dry-run prints 7 payload blocks with the 7 phase names (`Inhale-1` … `Dissolve`), then exits 0. `out/ambient/smoke-dryrun/` is created but empty.

- [ ] **Step 11.3: Run the existing handler tests to confirm nothing else regressed**

```bash
pytest test_handler.py test_workflow.py -q
```

Expected: all prior tests still PASS — we haven't touched `handler.py` or the workflow.

- [ ] **Step 11.4: Commit**

```bash
git add README.md
git commit -m "docs(readme): link new ambient orchestrator script"
```

---

## Task 12: Live smoke test — one 60 s segment end-to-end

**Only run this after Tasks 1–11 are green.** Costs ~1 × single-segment RunPod run (<2 minutes, <$0.10 on a 4090).

- [ ] **Step 12.1: Submit one real 60 s segment**

```bash
export RUNPOD_API_KEY=<your-key>
export RUNPOD_ENDPOINT_ID=nwqnd0duxc6o38
python3 scripts/ambient_eno_45min.py \
    --run-id smoke-live \
    --segment 1 \
    --duration 60
```

Expected console output:
```
Run directory: /root/ace-step-music-xl/out/ambient/smoke-live
[1/7] submitting Inhale-1 (duration=60s, seed=-1)...
[1/7] done in XXXs — seed=<int> → segment_01.flac
Skipping stitch — 6 segments still missing.
```

Verify:
- `out/ambient/smoke-live/segment_01.flac` exists and plays (`ffprobe` or local audio player).
- `out/ambient/smoke-live/segment_01.json` captures the seed.
- Exit code 0.

Do NOT run the full 7-segment batch from this plan — that's a user-driven execution after they've reviewed the smoke-test output.

- [ ] **Step 12.2: (Optional) Final commit of run artifacts**

Skip this step unless the user explicitly wants to archive the smoke-test output in git. Run artifacts under `out/` are usually ignored by `.gitignore`.

---

## Self-review checklist (for the implementer to verify before declaring done)

- [ ] Every step contains the actual code/command needed — no "TODO", "TBD", "similar to above", "add validation", etc.
- [ ] Exact file paths given in every task.
- [ ] All 7 segments get their sidecar + FLAC, and a manifest.json is written even on partial completion.
- [ ] `--force`, `--segment N`, `--stitch-only`, `--dry-run`, `--pin-seeds-from`, `--duration` all behave per §8 of the design.
- [ ] FFmpeg invocation uses `c1=qsin:c2=qsin` (equal-power) with `d=30`, and six `acrossfade=` filters for seven inputs.
- [ ] `PRESET` dict matches the official INFERENCE.md Example 9 values verbatim.
- [ ] No changes to `handler.py`, `Dockerfile`, or `.github/workflows/*`.
- [ ] `pytest test_ambient_eno_45min.py` is fully green with **zero** real HTTP calls (`responses` mocks everything).
- [ ] `pytest test_handler.py test_workflow.py` still green (no regression).
- [ ] Live smoke in Task 12 produced a playable FLAC of ~60 s at 48 kHz.

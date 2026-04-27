# LTX Handler Resolution Params Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `width` and `height` input params to the LTX-2.3 ComfyUI serverless handler at `/root/ltx23-pro6000` so it can produce 1280×704 landscape video while preserving its current 480×832 portrait default exactly. Deploy via RunPod template versioning with a one-API-call rollback path.

**Architecture:** Backwards-compatible handler change — width/height default to 480/832 when omitted, so existing callers (avatar-video) see no behavior change. Workflow JSON nodes 33 (`ResizeImageMaskNode`) and 30 (`ResizeImagesByLongerEdge`) get injected from the handler instead of being hardcoded.

**Tech Stack:** Python 3, pytest, runpod SDK, Docker, RunPod CLI/MCP, ComfyUI LTX-2.3 workflow.

**Prerequisite:** None. This plan ships independently of the orchestrator.

**Spec:** `docs/superpowers/specs/2026-04-27-loop-music-video-design.md` §7.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `/root/ltx23-pro6000/handler.py` | Modify (~15 LOC) | Add `width`/`height` parsing + workflow node injection |
| `/root/ltx23-pro6000/test_handler.py` | Create | Unit tests for `inject_inputs` width/height behavior + back-compat sentinel |
| `/root/ltx23-pro6000/requirements-test.txt` | Create | `pytest`, `pytest-cov` |
| `/root/ltx23-pro6000/fixtures/workflow_baseline.json` | Create | Snapshot of current `inject_inputs` output for byte-identical regression |
| `/root/ltx23-pro6000/scripts/smoke_portrait.py` | Create | Live smoke test against deployed endpoint, payload omits width/height, expects 480×832 |
| `/root/ltx23-pro6000/scripts/smoke_landscape.py` | Create | Live smoke test, explicit width=1280 height=704, expects 1280×704 |
| `/root/ltx23-pro6000/docs/ROLLBACK.md` | Create | Operator runbook for template-version rollback |

---

## Task 1: Set up local test environment

**Files:**
- Create: `/root/ltx23-pro6000/requirements-test.txt`

- [ ] **Step 1: Create requirements-test.txt**

```
pytest>=7.0.0
pytest-cov>=4.0.0
```

- [ ] **Step 2: Install test dependencies**

Run: `cd /root/ltx23-pro6000 && pip install -r requirements-test.txt`
Expected: pytest installed (or "Requirement already satisfied").

- [ ] **Step 3: Verify pytest discovery works**

Run: `cd /root/ltx23-pro6000 && pytest --collect-only 2>&1 | tail -5`
Expected: "no tests ran" or empty collection (no test_*.py exists yet).

- [ ] **Step 4: Commit**

```bash
cd /root/ltx23-pro6000
git add requirements-test.txt
git commit -m "test: add pytest + pytest-cov for handler unit tests"
```

---

## Task 2: Capture baseline workflow JSON for back-compat sentinel

**Files:**
- Create: `/root/ltx23-pro6000/fixtures/workflow_baseline.json`

- [ ] **Step 1: Write a baseline-capture script (one-shot, not committed)**

Create `/tmp/capture_baseline.py`:

```python
"""Capture the current inject_inputs() output as a baseline fixture."""
import json
import sys
sys.path.insert(0, "/root/ltx23-pro6000")
from handler import inject_inputs, load_workflow

wf = load_workflow.__wrapped__() if hasattr(load_workflow, '__wrapped__') else None
# load_workflow reads from /workflow_api.json (runtime path); read directly here:
with open("/root/ltx23-pro6000/workflow_api.json") as f:
    wf_template = json.load(f)

# Use the same args avatar-video would pass (no width/height, current handler signature)
result = inject_inputs(
    workflow=wf_template,
    image_filename="input_baseline.png",
    audio_filename="input_baseline.mp3",
    prompt="a person talking to camera",
    negative_prompt="blurry, low quality, still frame, frames, watermark, overlay, titles, has blurbox, has subtitles",
    num_frames=None,
    fps=24,
    seed=42,
    reference_filename=None,
    lora_strength=None,
)

with open("/root/ltx23-pro6000/fixtures/workflow_baseline.json", "w") as f:
    json.dump(result, f, indent=2, sort_keys=True)
print("Baseline captured.")
```

- [ ] **Step 2: Create fixtures dir + run capture**

Run: `mkdir -p /root/ltx23-pro6000/fixtures && python3 /tmp/capture_baseline.py`
Expected: Prints "Baseline captured." File `fixtures/workflow_baseline.json` exists.

- [ ] **Step 3: Verify baseline contains expected nodes**

Run: `cd /root/ltx23-pro6000 && python3 -c "import json; d=json.load(open('fixtures/workflow_baseline.json')); print(d['33']['inputs']['resize_type.width'], d['33']['inputs']['resize_type.height'], d['30']['inputs']['longer_edge'])"`
Expected output: `480 832 832`

- [ ] **Step 4: Commit baseline**

```bash
cd /root/ltx23-pro6000
git add fixtures/workflow_baseline.json
git commit -m "test: snapshot current inject_inputs output as baseline fixture"
```

---

## Task 3: Write the back-compat sentinel test (must PASS against current code)

**Files:**
- Create: `/root/ltx23-pro6000/test_handler.py`

- [ ] **Step 1: Write the sentinel test**

```python
"""Unit tests for handler.inject_inputs — width/height behavior and back-compat sentinel."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from handler import inject_inputs

REPO_ROOT = Path(__file__).parent
BASELINE = json.loads((REPO_ROOT / "fixtures" / "workflow_baseline.json").read_text())


def load_workflow_template():
    return json.loads((REPO_ROOT / "workflow_api.json").read_text())


def call_inject(**overrides):
    """Convenience: call inject_inputs with the same args used to capture the baseline,
    overridden by any kwargs the test cares about."""
    args = dict(
        workflow=load_workflow_template(),
        image_filename="input_baseline.png",
        audio_filename="input_baseline.mp3",
        prompt="a person talking to camera",
        negative_prompt="blurry, low quality, still frame, frames, watermark, overlay, titles, has blurbox, has subtitles",
        num_frames=None,
        fps=24,
        seed=42,
        reference_filename=None,
        lora_strength=None,
    )
    args.update(overrides)
    return inject_inputs(**args)


def test_default_call_matches_baseline_byte_identical():
    """Regression sentinel: a call without width/height must produce the same workflow
    JSON as the v4 deployment. If this test ever fails, avatar-video integration is at risk."""
    result = call_inject()
    assert result == BASELINE, (
        "inject_inputs output drifted from baseline. "
        "Re-capture baseline only after deliberate, audited change."
    )
```

- [ ] **Step 2: Run the test (should PASS — code is unchanged)**

Run: `cd /root/ltx23-pro6000 && pytest test_handler.py::test_default_call_matches_baseline_byte_identical -v`
Expected: 1 passed.

- [ ] **Step 3: Commit the sentinel**

```bash
cd /root/ltx23-pro6000
git add test_handler.py
git commit -m "test: add byte-identical workflow JSON sentinel for inject_inputs back-compat"
```

---

## Task 4: Write failing tests for new width/height behavior

**Files:**
- Modify: `/root/ltx23-pro6000/test_handler.py` (append)

- [ ] **Step 1: Add the landscape injection test (will FAIL — function doesn't accept kwargs yet)**

Append to `test_handler.py`:

```python
def test_landscape_1280x704_injects_into_nodes_30_and_33():
    """When width=1280 height=704 is passed, both resize nodes must be updated."""
    result = call_inject(width=1280, height=704)
    assert result["33"]["inputs"]["resize_type.width"] == 1280
    assert result["33"]["inputs"]["resize_type.height"] == 704
    assert result["30"]["inputs"]["longer_edge"] == 1280


def test_portrait_default_when_width_height_omitted():
    """Default is 480x832 portrait — same as current v4 behavior."""
    result = call_inject()
    assert result["33"]["inputs"]["resize_type.width"] == 480
    assert result["33"]["inputs"]["resize_type.height"] == 832
    assert result["30"]["inputs"]["longer_edge"] == 832


def test_explicit_portrait_480x832_matches_default():
    """Explicit width=480 height=832 produces identical output to no-args call."""
    explicit = call_inject(width=480, height=832)
    default = call_inject()
    assert explicit == default


def test_square_1024x1024_injects_correctly():
    """Square aspect — covers the longer_edge=max(w,h) branch when w==h."""
    result = call_inject(width=1024, height=1024)
    assert result["33"]["inputs"]["resize_type.width"] == 1024
    assert result["33"]["inputs"]["resize_type.height"] == 1024
    assert result["30"]["inputs"]["longer_edge"] == 1024
```

- [ ] **Step 2: Run new tests (must FAIL — inject_inputs has no width/height kwargs)**

Run: `cd /root/ltx23-pro6000 && pytest test_handler.py -v`
Expected: 3 failures with `TypeError: inject_inputs() got an unexpected keyword argument 'width'`. The sentinel test still passes.

---

## Task 5: Implement width/height params in inject_inputs

**Files:**
- Modify: `/root/ltx23-pro6000/handler.py:34-44, 84-133`

- [ ] **Step 1: Add new constants near the existing NODE_* block**

Edit `handler.py`. Find the block starting `NODE_LOAD_IMAGE = "48"` (around line 34) and add at the end of that constants block (before `# ── Defaults ──`):

```python
NODE_RESIZE_IMAGE = "33"             # ResizeImageMaskNode — drives output dimensions
NODE_RESIZE_LONGER_EDGE = "30"       # ResizeImagesByLongerEdge — pre-crop sizing
```

Then add to the `# ── Defaults ──` block:

```python
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 832
```

- [ ] **Step 2: Add width/height params to inject_inputs signature and body**

Modify the `inject_inputs` function signature to add two params at the end:

```python
def inject_inputs(workflow, image_filename, audio_filename, prompt,
                  negative_prompt, num_frames, fps, seed,
                  reference_filename=None, lora_strength=None,
                  width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
```

At the end of the function body, just before the `return wf` line, add:

```python
    if NODE_RESIZE_IMAGE in wf:
        wf[NODE_RESIZE_IMAGE]["inputs"]["resize_type.width"] = width
        wf[NODE_RESIZE_IMAGE]["inputs"]["resize_type.height"] = height
    if NODE_RESIZE_LONGER_EDGE in wf:
        wf[NODE_RESIZE_LONGER_EDGE]["inputs"]["longer_edge"] = max(width, height)
```

- [ ] **Step 3: Run unit tests — all 4 must now PASS**

Run: `cd /root/ltx23-pro6000 && pytest test_handler.py -v`
Expected: 4 passed (sentinel + 3 new). The sentinel still passes because defaults match the baseline.

- [ ] **Step 4: Commit handler change**

```bash
cd /root/ltx23-pro6000
git add handler.py test_handler.py
git commit -m "feat(handler): add width/height params to inject_inputs (default 480x832)

Backwards-compatible: callers omitting width/height get the existing portrait
output. New: pass width=1280 height=704 for 16:9 landscape video.

Workflow nodes 33 (ResizeImageMaskNode) and 30 (ResizeImagesByLongerEdge)
are now injected from the handler instead of being hardcoded in the JSON."
```

---

## Task 6: Add 32-divisibility validation to handler()

**Files:**
- Modify: `/root/ltx23-pro6000/test_handler.py` (append)
- Modify: `/root/ltx23-pro6000/handler.py:224-260` (the `handler` function)

- [ ] **Step 1: Write failing tests for handler-level validation**

Append to `test_handler.py`:

```python
from handler import handler


def make_job(**input_overrides):
    """Build a minimal job dict for handler() — image and audio are required."""
    base = {
        "image_base64": "aGVsbG8=",   # b"hello", just to satisfy the presence check
        "audio_base64": "aGVsbG8=",
    }
    base.update(input_overrides)
    return {"input": base}


def test_handler_rejects_width_not_multiple_of_32():
    result = handler(make_job(width=1280, height=720))
    assert "error" in result
    assert "multiples of 32" in result["error"]
    assert "720" in result["error"]


def test_handler_rejects_height_not_multiple_of_32():
    result = handler(make_job(width=1281, height=704))
    assert "error" in result
    assert "multiples of 32" in result["error"]
    assert "1281" in result["error"]


def test_handler_accepts_valid_landscape():
    """1280x704 must NOT trigger validation — both divisible by 32.
    The test catches the validation error specifically; a downstream ComfyUI
    error from the mocked-absent server is expected and ignored here."""
    result = handler(make_job(width=1280, height=704))
    # Validation passed if error doesn't mention 'multiples of 32'.
    if "error" in result:
        assert "multiples of 32" not in result["error"]


def test_handler_accepts_default_when_width_height_omitted():
    """No width/height → defaults pass validation."""
    result = handler(make_job())
    if "error" in result:
        assert "multiples of 32" not in result["error"]
```

- [ ] **Step 2: Run tests (validation tests must FAIL — no validation exists yet)**

Run: `cd /root/ltx23-pro6000 && pytest test_handler.py -v`
Expected: 4 passed (from previous task) + 2 failures on the rejection tests.

- [ ] **Step 3: Implement validation in handler()**

Edit `handler.py`. Find the `handler(job)` function, locate the input-parsing block (around line 240). After the existing input parsing for `seed`/`lora_strength`, add:

```python
        width = int(job_input.get("width", DEFAULT_WIDTH))
        height = int(job_input.get("height", DEFAULT_HEIGHT))
        if width % 32 != 0 or height % 32 != 0:
            return {"error": f"width and height must be multiples of 32 (got {width}x{height})"}
```

Then find the call to `inject_inputs(...)` (around line 285) and add `width` and `height` to the kwargs:

```python
        workflow = inject_inputs(
            workflow,
            image_filename=f"input_{job_id}.png",
            audio_filename=f"input_{job_id}.mp3",
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            reference_filename=ref_filename,
            lora_strength=lora_strength,
            width=width,
            height=height,
        )
```

- [ ] **Step 4: Run tests — all 8 must now PASS**

Run: `cd /root/ltx23-pro6000 && pytest test_handler.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit validation**

```bash
cd /root/ltx23-pro6000
git add handler.py test_handler.py
git commit -m "feat(handler): validate width/height are multiples of 32

LTX-2.3 video VAE requires spatial dimensions divisible by 32. Rejecting
invalid input early (before ComfyUI submission) saves ~$0.05 of wasted
worker time per bad request."
```

---

## Task 7: Run coverage check

- [ ] **Step 1: Measure handler coverage**

Run: `cd /root/ltx23-pro6000 && pytest test_handler.py --cov=handler --cov-report=term-missing 2>&1 | tail -25`
Expected: `inject_inputs` and `handler` width/height code paths covered. Coverage of `inject_inputs` ≥ 80%.

- [ ] **Step 2: If coverage < 80% on new lines, add tests for missed branches**

Identify missed lines from the coverage report. Common gaps: workflow without nodes 30/33 (defensive `if NODE_X in wf` paths). Add a test if needed:

```python
def test_inject_inputs_skips_when_node_30_absent():
    wf = load_workflow_template()
    del wf["30"]
    # Should not raise
    result = call_inject(width=1280, height=704)  # uses fresh template
    # Re-call with stripped wf
    from handler import inject_inputs
    args = dict(workflow=wf, image_filename="x.png", audio_filename="x.mp3",
                prompt="x", negative_prompt="x", num_frames=None, fps=24,
                seed=42, reference_filename=None, lora_strength=None,
                width=1280, height=704)
    out = inject_inputs(**args)
    assert "30" not in out
    assert out["33"]["inputs"]["resize_type.width"] == 1280
```

Run: `cd /root/ltx23-pro6000 && pytest test_handler.py -v` → 9 passed.

---

## Task 8: Push handler changes — CI builds new Docker image

The repo's GitHub Actions workflow (`.github/workflows/docker-build.yml`) auto-builds and pushes `dmrabh/ltx-23-video-pro6000:latest` and `dmrabh/ltx-23-video-pro6000:<sha>` on push to main.

- [ ] **Step 1: Push to main**

```bash
cd /root/ltx23-pro6000
git push origin main
```

- [ ] **Step 2: Wait for CI build to complete**

Run: `gh run list --workflow=docker-build.yml --limit=1`
Expected: A run for the latest commit. Wait for status `completed` and conclusion `success`.

To watch in real time: `gh run watch`

- [ ] **Step 3: Capture the new image SHA tag**

Run: `cd /root/ltx23-pro6000 && git rev-parse --short=7 HEAD`
Expected: 7-char SHA. Note this — call it `NEW_SHA`.

The new Docker image is at `dmrabh/ltx-23-video-pro6000:<NEW_SHA>`.

---

## Task 9: Create a new RunPod template version pointing at the new image

**Files:**
- None — RunPod-side changes via MCP tool / CLI

- [ ] **Step 1: Read current template (qo92k71b0g) v4 settings**

Use the runpod MCP tool `mcp__runpod__get-template`:

```
mcp__runpod__get-template(templateId="qo92k71b0g")
```

Note the current `imageName` (probably `dmrabh/ltx-23-video-pro6000:latest` or similar) and all env vars.

- [ ] **Step 2: Create a NEW template with the SHA-pinned image**

Use `mcp__runpod__create-template`:

```
mcp__runpod__create-template(
    name="LTX2.3 ComfyUI PRO6000 v5",
    imageName="dmrabh/ltx-23-video-pro6000:<NEW_SHA>",  # SHA-pinned, not :latest
    containerDiskInGb=20,
    isServerless=True,
    env={
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.8"
    },
    ports="8888/http,22/tcp,22/udp"
)
```

Capture the returned template ID — call it `NEW_TEMPLATE_ID`.

**Why SHA-pinned:** `:latest` is a moving target. Template versions must be reproducible — if we revert and `:latest` has moved, the "old" template might pull a different image than what we tested with.

---

## Task 10: Pre-flip smoke test on a one-off pod

**Files:**
- Create: `/root/ltx23-pro6000/scripts/smoke_portrait.py`
- Create: `/root/ltx23-pro6000/scripts/smoke_landscape.py`

- [ ] **Step 1: Create smoke_portrait.py**

```python
"""Live smoke test: payload omits width/height, expects 480x832 portrait video.
This is the back-compat canary. Call it after every redeploy."""
import argparse
import base64
import io
import os
import sys
import time

import requests
from PIL import Image


def gradient_image(w=480, h=832):
    """Cheap synthetic input image."""
    img = Image.new("RGB", (w, h), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def silent_audio(duration_sec=4):
    """Synthesize a silent MP3 in memory using ffmpeg via subprocess."""
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r=44100:cl=stereo",
        "-t", str(duration_sec),
        "-c:a", "libmp3lame", "-b:a", "128k",
        "-f", "mp3", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return base64.b64encode(result.stdout).decode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default=os.environ.get("LTX_ENDPOINT_ID", "1g0pvlx8ar6qns"))
    p.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    args = p.parse_args()
    if not args.api_key:
        sys.exit("RUNPOD_API_KEY not set")

    payload = {
        "input": {
            "image_base64": gradient_image(),
            "audio_base64": silent_audio(),
            "prompt": "smooth gentle ambient motion",
            "fps": 24,
            "num_frames": 97,   # default
            "seed": 42,
            # NOTE: width/height intentionally OMITTED — back-compat path
        }
    }
    url = f"https://api.runpod.ai/v2/{args.endpoint}/run"
    headers = {"Authorization": f"Bearer {args.api_key}"}

    print(f"[portrait smoke] submitting to {args.endpoint}")
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    job_id = r.json()["id"]
    print(f"[portrait smoke] job id {job_id}")

    while True:
        s = requests.get(f"https://api.runpod.ai/v2/{args.endpoint}/status/{job_id}",
                         headers=headers, timeout=30).json()
        status = s.get("status", "")
        print(f"[portrait smoke] status={status}")
        if status == "COMPLETED":
            break
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            sys.exit(f"job terminal: {s}")
        time.sleep(5)

    out = s.get("output", {})
    if "error" in out:
        sys.exit(f"handler error: {out['error']}")
    video_b64 = out.get("video")
    if not video_b64:
        sys.exit(f"no video in output: {list(out.keys())}")

    video_bytes = base64.b64decode(video_b64)
    out_path = "/tmp/smoke_portrait.mp4"
    with open(out_path, "wb") as f:
        f.write(video_bytes)
    print(f"[portrait smoke] wrote {out_path} ({len(video_bytes):,} bytes)")

    # Verify dimensions with ffprobe
    import subprocess
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", out_path],
        capture_output=True, text=True, check=True,
    )
    w, h = probe.stdout.strip().split(",")
    print(f"[portrait smoke] dimensions: {w}x{h}")
    if (w, h) != ("480", "832"):
        sys.exit(f"FAIL: expected 480x832, got {w}x{h}")
    print("[portrait smoke] PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create smoke_landscape.py (analogous, but with width=1280 height=704)**

Same content as `smoke_portrait.py` with these diffs:

- File header comment changed to: `"""Live smoke test: explicit width=1280 height=704, expects landscape video."""`
- All `[portrait smoke]` log prefixes → `[landscape smoke]`
- Add to the `payload["input"]` dict: `"width": 1280, "height": 704`
- Output path: `/tmp/smoke_landscape.mp4`
- Final dimension check: `if (w, h) != ("1280", "704"): sys.exit(...)`

- [ ] **Step 3: Make scripts executable + commit**

```bash
mkdir -p /root/ltx23-pro6000/scripts
chmod +x /root/ltx23-pro6000/scripts/smoke_portrait.py /root/ltx23-pro6000/scripts/smoke_landscape.py
cd /root/ltx23-pro6000
git add scripts/smoke_portrait.py scripts/smoke_landscape.py
git commit -m "test: add live smoke scripts for portrait back-compat + landscape new path"
```

- [ ] **Step 4: Spin up a one-off pod with the NEW image (NOT yet flipped to endpoint)**

Use `mcp__runpod__create-pod`:

```
mcp__runpod__create-pod(
    name="ltx-smoke-v5",
    imageName="dmrabh/ltx-23-video-pro6000:<NEW_SHA>",
    gpuTypeIds=["NVIDIA RTX PRO 6000 Blackwell Server Edition"],
    containerDiskInGb=20,
    networkVolumeId="7oh9kxn660",
    env={
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.8"
    },
    ports=["8888/http", "22/tcp"]
)
```

Note the pod ID and IP from the response. Wait until pod status is `RUNNING`.

- [ ] **Step 5: Run handler.py on the pod via SSH and submit a portrait test job**

This is for a **standalone pod**, not serverless. The handler.py file is run via `python3 /handler.py` which starts the runpod serverless agent locally. Easier alternative for smoke testing: use a temp serverless endpoint pointed at the new template version.

**Simpler approach — create a temp endpoint:**

```
mcp__runpod__create-endpoint(
    name="ltx-smoke-v5-endpoint",
    templateId="<NEW_TEMPLATE_ID>",
    gpuTypeIds=["NVIDIA RTX PRO 6000 Blackwell Server Edition"],
    workersMax=1,
    workersMin=0,
    networkVolumeId="7oh9kxn660"
)
```

Capture the temp endpoint ID — call it `TEMP_EP_ID`.

- [ ] **Step 6: Run portrait smoke against TEMP_EP_ID**

Run: `LTX_ENDPOINT_ID=<TEMP_EP_ID> python3 /root/ltx23-pro6000/scripts/smoke_portrait.py`
Expected: ends with `[portrait smoke] PASS`. Total ~3-5 min including cold start.

- [ ] **Step 7: Run landscape smoke against TEMP_EP_ID**

Run: `LTX_ENDPOINT_ID=<TEMP_EP_ID> python3 /root/ltx23-pro6000/scripts/smoke_landscape.py`
Expected: ends with `[landscape smoke] PASS` and dimensions `1280x704`. Total ~3-5 min (worker is now warm).

- [ ] **Step 8: Tear down temp endpoint**

```
mcp__runpod__delete-endpoint(endpointId="<TEMP_EP_ID>")
```

If you also created a one-off pod in step 4, delete it too:
```
mcp__runpod__delete-pod(podId="<POD_ID>")
```

---

## Task 11: Flip production endpoint to new template version

- [ ] **Step 1: Snapshot the current endpoint template-version for rollback reference**

Use `mcp__runpod__get-endpoint`:

```
mcp__runpod__get-endpoint(endpointId="1g0pvlx8ar6qns", includeTemplate=True)
```

Record the current `templateId` and template version — call it `OLD_TEMPLATE_ID`. (Currently `qo92k71b0g`.)

- [ ] **Step 2: Update endpoint to point at new template**

Use `mcp__runpod__update-endpoint`:

```
mcp__runpod__update-endpoint(
    endpointId="1g0pvlx8ar6qns",
    templateId="<NEW_TEMPLATE_ID>"
)
```

The endpoint workers will drain (finish in-flight jobs) and re-pull with the new image. ETA 30-90 seconds.

- [ ] **Step 3: Wait for new workers to be ready**

Run repeatedly (every 30s) until at least one worker has the new image:

```
mcp__runpod__get-endpoint(endpointId="1g0pvlx8ar6qns", includeWorkers=True)
```

---

## Task 12: Post-flip smoke tests + avatar-video integration

- [ ] **Step 1: Run portrait smoke against the LIVE production endpoint**

Run: `LTX_ENDPOINT_ID=1g0pvlx8ar6qns python3 /root/ltx23-pro6000/scripts/smoke_portrait.py`
Expected: `[portrait smoke] PASS` with dimensions `480x832`. **If this fails, ROLLBACK immediately (Task 13).**

- [ ] **Step 2: Run landscape smoke against the live endpoint**

Run: `LTX_ENDPOINT_ID=1g0pvlx8ar6qns python3 /root/ltx23-pro6000/scripts/smoke_landscape.py`
Expected: `[landscape smoke] PASS` with dimensions `1280x704`. **If this fails, ROLLBACK.**

- [ ] **Step 3: Run avatar-video integration test (existing, unchanged)**

Run:

```bash
cd /root/avatar-video
RUNPOD_ENDPOINT_ID=1g0pvlx8ar6qns npx ts-node src/scripts/test-episode-avatar.ts
```

Expected: existing test passes (it doesn't pass width/height, so it exercises the back-compat path end-to-end with the production endpoint). **If this fails, ROLLBACK.**

If all three steps pass, the deploy is verified.

---

## Task 13: Document rollback procedure

**Files:**
- Create: `/root/ltx23-pro6000/docs/ROLLBACK.md`

- [ ] **Step 1: Write the runbook**

```markdown
# Rollback runbook — LTX-2.3 ComfyUI endpoint

If a recently-deployed handler version causes failures (smoke tests fail,
production 5xx, avatar-video integration breaks), revert via:

## Quick revert (RunPod template version, ~10 sec)

```
mcp__runpod__update-endpoint(
    endpointId="1g0pvlx8ar6qns",
    templateId="<OLD_TEMPLATE_ID>"
)
```

The endpoint workers will drain in-flight jobs and re-pull the previous image.
Recovery time: ~10 sec for the API call, plus ~30-90 sec for workers to be ready.
**No data loss, no in-flight job loss.**

## Verify revert

```bash
LTX_ENDPOINT_ID=1g0pvlx8ar6qns python3 scripts/smoke_portrait.py
```

Expected: `[portrait smoke] PASS`.

## Known-good template versions

| Date | Template ID | Description |
|---|---|---|
| 2026-03-16 | qo92k71b0g v4 | Original — portrait 480x832 only, no width/height params |
| 2026-04-27 | <NEW_TEMPLATE_ID> | Adds width/height params, defaults preserve v4 behavior |

After every successful redeploy, update this table — the template ID is the
**single source of truth** for "what to revert to."

## Post-rollback investigation

1. Check the `gh run list --workflow=docker-build.yml` for build issues
2. Re-run unit tests locally: `cd /root/ltx23-pro6000 && pytest test_handler.py`
3. Check the failing smoke test output for the actual error
4. If logs are needed, check the live endpoint's logs panel in the RunPod UI
   (or via `mcp__runpod__get-endpoint(endpointId=..., includeWorkers=True)`
   then per-worker log queries)
```

- [ ] **Step 2: Commit + push**

```bash
mkdir -p /root/ltx23-pro6000/docs
cd /root/ltx23-pro6000
git add docs/ROLLBACK.md
git commit -m "docs: rollback runbook for LTX endpoint template versioning"
git push origin main
```

---

## Task 14: Tag the release

- [ ] **Step 1: Tag the release commit**

```bash
cd /root/ltx23-pro6000
git tag -a v5-resolution-params -m "Add width/height params to handler. Default 480x832 preserved."
git push origin v5-resolution-params
```

This gives us a named anchor for "the version where width/height was added."

---

## Self-Review

**Spec coverage** (against §7 of the design):
- §7.1 handler diff (~15 LOC) → Task 5 + Task 6 ✓
- §7.2 back-compat byte-identical workflow JSON sentinel → Task 3 ✓
- §7.3 deploy + rollback strategy:
  - Build new image with version tag → Task 8 (CI auto-tags by SHA) ✓
  - Create new template version → Task 9 ✓
  - Pre-flip smoke tests via temp endpoint → Task 10 ✓
  - Flip endpoint → Task 11 ✓
  - Post-flip smoke tests + avatar-video integration → Task 12 ✓
  - Rollback runbook → Task 13 ✓

**Placeholder scan:** No "TBD"/"TODO"/"add error handling later" — every step has actual code or actual commands. ✓

**Type consistency:** `inject_inputs(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)` signature matches the calls in tests (Task 4) and in `handler()` (Task 6). The constants `DEFAULT_WIDTH=480, DEFAULT_HEIGHT=832` defined in Task 5 are used consistently. ✓

**Spec gaps:** None.

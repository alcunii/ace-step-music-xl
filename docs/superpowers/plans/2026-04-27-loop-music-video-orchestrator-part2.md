# Loop Music Video Orchestrator — Part 2 (Tasks 6-17)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** Part 1 (`2026-04-27-loop-music-video-orchestrator.md`) tasks 1–5 must be complete.

**Spec:** `docs/superpowers/specs/2026-04-27-loop-music-video-design.md`.

---

## Phase 6 — LLM Planner

### Task 6: OpenRouter Gemini 3 Flash call with structured output + retry

**Files:**
- Create: `scripts/loopvid/llm_planner.py`
- Create: `test_loopvid_llm_planner.py`

- [ ] **Step 1: Write tests**

Create `/root/ace-step-music-xl/test_loopvid_llm_planner.py`:

```python
import json
from unittest.mock import patch

import pytest
import responses

from scripts.loopvid.llm_planner import plan, OPENROUTER_URL


VALID_RESPONSE = {
    "choices": [{"message": {"content": json.dumps({
        "genre": "lofi",
        "mood": "rainy night",
        "music_palette": "Lofi instrumental, 70 BPM, vinyl crackle, jazz piano, "
                         "soft rim shot drums, warm tape saturation",
        "music_segment_descriptors": [
            {"phase": f"phase-{i}", "descriptors": f"desc-{i}"} for i in range(1, 12)
        ],
        "music_bpm": 70,
        "seedream_scene": "A wooden desk by a rainy window at dusk",
        "seedream_style": "Shot on 35mm film, Kodak Portra 400, soft warm rim",
        "motion_prompts": [f"motion {i}" for i in range(1, 7)],
        "motion_archetype": "rain",
        "image_archetype_key": "rainy_window_desk",
    })}}]
}


@responses.activate
def test_plan_returns_valid_plan_on_success():
    responses.add(responses.POST, OPENROUTER_URL, json=VALID_RESPONSE, status=200)
    p = plan(genre="lofi", mood="rainy night", api_key="k")
    assert p.genre == "lofi"
    assert len(p.motion_prompts) == 6


@responses.activate
def test_plan_uses_gemini_3_flash_preview_model():
    responses.add(responses.POST, OPENROUTER_URL, json=VALID_RESPONSE, status=200)
    plan(genre="lofi", mood="rainy night", api_key="k")
    body = json.loads(responses.calls[0].request.body)
    assert body["model"] == "google/gemini-3-flash-preview"


@responses.activate
def test_plan_retries_on_5xx():
    responses.add(responses.POST, OPENROUTER_URL, status=503)
    responses.add(responses.POST, OPENROUTER_URL, status=503)
    responses.add(responses.POST, OPENROUTER_URL, json=VALID_RESPONSE, status=200)
    p = plan(genre="lofi", mood="x", api_key="k", retry_sleep=0)
    assert p.genre == "lofi"
    assert len(responses.calls) == 3


@responses.activate
def test_plan_aborts_after_max_5xx_retries():
    for _ in range(4):
        responses.add(responses.POST, OPENROUTER_URL, status=503)
    with pytest.raises(RuntimeError, match="503"):
        plan(genre="lofi", mood="x", api_key="k", retry_sleep=0)


@responses.activate
def test_plan_retries_on_schema_validation_failure():
    bad = {
        "choices": [{"message": {"content": json.dumps({
            "genre": "lofi", "mood": "x",
            # MISSING required music_palette
        })}}]
    }
    responses.add(responses.POST, OPENROUTER_URL, json=bad, status=200)
    responses.add(responses.POST, OPENROUTER_URL, json=VALID_RESPONSE, status=200)
    p = plan(genre="lofi", mood="x", api_key="k", retry_sleep=0)
    assert p.genre == "lofi"
    assert len(responses.calls) == 2


def test_plan_raises_clear_error_when_api_key_missing():
    with pytest.raises(ValueError, match="api_key"):
        plan(genre="lofi", mood="x", api_key="")
```

- [ ] **Step 2: Run tests — must FAIL**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_llm_planner.py -v 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement llm_planner.py**

Create `/root/ace-step-music-xl/scripts/loopvid/llm_planner.py`:

```python
"""OpenRouter Gemini 3 Flash planner with structured output + retry."""
from __future__ import annotations

import json
import time
from typing import Optional

import requests

from scripts.loopvid.constants import GENRE_ARCHETYPES, SEGMENT_COUNT_60MIN, CLIP_COUNT
from scripts.loopvid.plan_schema import Plan, PlanSchemaError, validate_plan_dict

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3-flash-preview"
TEMPERATURE = 0.7
MAX_5XX_RETRIES = 3
MAX_SCHEMA_RETRIES = 3
EXP_BACKOFF_BASE_SEC = 1


def _build_system_prompt() -> str:
    archetypes = ", ".join(sorted(GENRE_ARCHETYPES.keys()))
    return (
        "You are a planner for a 1-hour looping ambient/instrumental music-video "
        "generator. Given (genre, mood), produce a plan as JSON matching the schema.\n\n"
        "CONSTRAINTS YOU MUST OBEY:\n"
        f"- Music: {SEGMENT_COUNT_60MIN} segments, each ~360s, sharing a locked sonic "
        "palette (10-15 descriptors, ~400 chars). Each segment adds 2-3 phase descriptors "
        "implementing a breathing arc: 3 settle → 1 hold → 5 deepen-and-release → 2 dissolve.\n"
        f"- Image: pick the closest archetype from {{{archetypes}}}. Customize the specific "
        "objects, lighting, color grade. Camera must be locked.\n"
        f"- Motion: {CLIP_COUNT} prompts, all sharing the same base motion (single ambient "
        "source from the still). Clip 1 and clip 6 must depict the rest-state (so the loop "
        "seam is invisible). Clips 2-5 escalate then descalate amplitude.\n\n"
        "YOU MUST NOT:\n"
        "- Mention text, signs, faces, hands, fingers, mirrors, scene cuts, fast motion in "
        "any prompt (the constants module appends explicit constraints later).\n"
        "- Invent new genre archetypes; pick from the list.\n"
        "- Vary the still across clips (one still, six motion variations)."
    )


def _build_response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "genre": {"type": "string"},
            "mood": {"type": "string"},
            "music_palette": {"type": "string"},
            "music_segment_descriptors": {
                "type": "array", "minItems": SEGMENT_COUNT_60MIN, "maxItems": SEGMENT_COUNT_60MIN,
                "items": {
                    "type": "object",
                    "properties": {"phase": {"type": "string"}, "descriptors": {"type": "string"}},
                    "required": ["phase", "descriptors"],
                },
            },
            "music_bpm": {"type": "integer"},
            "seedream_scene": {"type": "string"},
            "seedream_style": {"type": "string"},
            "motion_prompts": {
                "type": "array", "minItems": CLIP_COUNT, "maxItems": CLIP_COUNT,
                "items": {"type": "string"},
            },
            "motion_archetype": {"type": "string"},
            "image_archetype_key": {"type": "string"},
        },
        "required": list(validate_plan_dict.__defaults__ or []) or [
            "genre", "mood", "music_palette", "music_segment_descriptors", "music_bpm",
            "seedream_scene", "seedream_style", "motion_prompts", "motion_archetype",
            "image_archetype_key",
        ],
    }


def _post(api_key: str, messages: list, retry_sleep: int) -> dict:
    """POST to OpenRouter with 5xx retry."""
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "plan", "schema": _build_response_schema(), "strict": True},
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(1, MAX_5XX_RETRIES + 1):
        resp = requests.post(OPENROUTER_URL, json=body, headers=headers, timeout=120)
        if 500 <= resp.status_code < 600:
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if attempt < MAX_5XX_RETRIES and retry_sleep > 0:
                time.sleep(retry_sleep * (4 ** (attempt - 1)))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"OpenRouter failed after {MAX_5XX_RETRIES} attempts: {last_err}")


def plan(
    *, genre: str, mood: str, api_key: str,
    raw_response_path: Optional[str] = None,
    retry_sleep: int = EXP_BACKOFF_BASE_SEC,
) -> Plan:
    """Call OpenRouter Gemini 3 Flash with structured output. Retries on 5xx
    and on schema-validation failure (with the validation error appended to
    the system prompt for the next attempt)."""
    if not api_key:
        raise ValueError("api_key is required (set OPENROUTER_API_KEY)")

    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": f"genre: {genre}\nmood: {mood}"},
    ]
    last_validation_err = None
    for attempt in range(1, MAX_SCHEMA_RETRIES + 1):
        if last_validation_err:
            messages = messages + [
                {"role": "user", "content": (
                    "Your previous response failed schema validation: "
                    f"{last_validation_err}. Respond again with valid JSON."
                )}
            ]
        body = _post(api_key, messages, retry_sleep=retry_sleep)
        content = body["choices"][0]["message"]["content"]
        if raw_response_path:
            from pathlib import Path
            Path(raw_response_path).write_text(content)
        try:
            d = json.loads(content)
            return validate_plan_dict(d)
        except (json.JSONDecodeError, PlanSchemaError) as e:
            last_validation_err = str(e)
            if attempt >= MAX_SCHEMA_RETRIES:
                raise RuntimeError(
                    f"Plan schema validation failed after {MAX_SCHEMA_RETRIES} attempts: {e}"
                )
    raise RuntimeError("unreachable")
```

- [ ] **Step 4: Run tests — must PASS**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_llm_planner.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/loopvid/llm_planner.py test_loopvid_llm_planner.py
git commit -m "feat(loopvid): OpenRouter Gemini 3 Flash planner with structured output + retry"
```

---

## Phase 7 — Image Pipeline

### Task 7: Replicate Seedream 4.5 caller with atomic write

**Files:**
- Create: `scripts/loopvid/image_pipeline.py`
- Create: `test_loopvid_image_pipeline.py`

- [ ] **Step 1: Write tests**

Create `/root/ace-step-music-xl/test_loopvid_image_pipeline.py`:

```python
import json
from unittest.mock import patch

import pytest
import responses

from scripts.loopvid.image_pipeline import (
    build_seedream_prompt,
    generate_still,
    REPLICATE_PREDICTIONS_URL,
)
from scripts.loopvid.constants import SEEDREAM_HARD_CONSTRAINTS


def test_build_seedream_prompt_appends_constraints():
    p = build_seedream_prompt("a red barn", "shot on 35mm")
    assert "a red barn" in p
    assert "shot on 35mm" in p
    assert SEEDREAM_HARD_CONSTRAINTS in p


def test_build_seedream_prompt_orders_scene_style_constraints():
    p = build_seedream_prompt("scene", "style")
    assert p.index("scene") < p.index("style") < p.index(SEEDREAM_HARD_CONSTRAINTS)


@responses.activate
def test_generate_still_does_not_send_seed_or_negative_or_image_input(tmp_path):
    submit_url = REPLICATE_PREDICTIONS_URL.format(model="bytedance/seedream-4.5")
    responses.add(responses.POST, submit_url,
                  json={"id": "pred-1", "status": "starting"}, status=201)
    responses.add(responses.GET, "https://api.replicate.com/v1/predictions/pred-1",
                  json={"id": "pred-1", "status": "succeeded",
                        "output": "https://example.com/out.png"}, status=200)
    responses.add(responses.GET, "https://example.com/out.png",
                  body=b"\x89PNG\r\n\x1a\n" + b"x" * 100, status=200)

    out = tmp_path / "still.png"
    generate_still(prompt="x", api_token="t", out_path=out, poll_interval=0)

    sent = json.loads(responses.calls[0].request.body)["input"]
    assert "seed" not in sent
    assert "negative_prompt" not in sent
    assert "image_input" not in sent
    assert sent["aspect_ratio"] == "16:9"


@responses.activate
def test_generate_still_writes_atomic(tmp_path):
    submit_url = REPLICATE_PREDICTIONS_URL.format(model="bytedance/seedream-4.5")
    responses.add(responses.POST, submit_url,
                  json={"id": "p", "status": "starting"}, status=201)
    responses.add(responses.GET, "https://api.replicate.com/v1/predictions/p",
                  json={"id": "p", "status": "succeeded",
                        "output": "https://example.com/img.png"}, status=200)
    responses.add(responses.GET, "https://example.com/img.png",
                  body=b"\x89PNG\r\n\x1a\nFAKE", status=200)

    out = tmp_path / "still.png"
    pred_id = generate_still(prompt="x", api_token="t", out_path=out, poll_interval=0)
    assert out.exists()
    assert not (tmp_path / "still.png.tmp").exists()
    assert pred_id == "p"


@responses.activate
def test_generate_still_returns_prediction_id_for_manifest(tmp_path):
    submit_url = REPLICATE_PREDICTIONS_URL.format(model="bytedance/seedream-4.5")
    responses.add(responses.POST, submit_url,
                  json={"id": "abc-123", "status": "starting"}, status=201)
    responses.add(responses.GET, "https://api.replicate.com/v1/predictions/abc-123",
                  json={"status": "succeeded", "output": "https://example.com/out.png"}, status=200)
    responses.add(responses.GET, "https://example.com/out.png",
                  body=b"\x89PNG\r\n", status=200)

    pred_id = generate_still(prompt="x", api_token="t", out_path=tmp_path / "s.png", poll_interval=0)
    assert pred_id == "abc-123"


@responses.activate
def test_generate_still_raises_on_failed(tmp_path):
    submit_url = REPLICATE_PREDICTIONS_URL.format(model="bytedance/seedream-4.5")
    responses.add(responses.POST, submit_url,
                  json={"id": "p", "status": "starting"}, status=201)
    responses.add(responses.GET, "https://api.replicate.com/v1/predictions/p",
                  json={"status": "failed", "error": "OOM"}, status=200)

    with pytest.raises(RuntimeError, match="OOM"):
        generate_still(prompt="x", api_token="t", out_path=tmp_path / "s.png", poll_interval=0)
```

- [ ] **Step 2: Run tests — must FAIL**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_image_pipeline.py -v 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement image_pipeline.py**

Create `/root/ace-step-music-xl/scripts/loopvid/image_pipeline.py`:

```python
"""Replicate Seedream 4.5 caller — text-to-image, atomic write to disk."""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from scripts.loopvid.constants import SEEDREAM_HARD_CONSTRAINTS

REPLICATE_PREDICTIONS_URL = "https://api.replicate.com/v1/models/{model}/predictions"
REPLICATE_PREDICTION_STATUS_URL = "https://api.replicate.com/v1/predictions/{pred_id}"
MODEL = "bytedance/seedream-4.5"
ASPECT_RATIO = "16:9"
POLL_INTERVAL_SEC = 3
POLL_TIMEOUT_SEC = 600
DOWNLOAD_TIMEOUT_SEC = 60


def build_seedream_prompt(scene: str, style: str) -> str:
    return f"{scene}. {style}. {SEEDREAM_HARD_CONSTRAINTS}"


def generate_still(
    *, prompt: str, api_token: str, out_path: Path,
    poll_interval: int = POLL_INTERVAL_SEC,
    timeout_sec: int = POLL_TIMEOUT_SEC,
) -> str:
    """Generate one still image via Replicate. Returns the prediction_id (for
    manifest tracking) and atomically writes the PNG bytes to out_path."""
    if not api_token:
        raise ValueError("api_token is required (set REPLICATE_API_TOKEN)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    submit_url = REPLICATE_PREDICTIONS_URL.format(model=MODEL)
    body = {"input": {"prompt": prompt, "aspect_ratio": ASPECT_RATIO}}
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

    resp = requests.post(submit_url, json=body, headers=headers, timeout=60)
    resp.raise_for_status()
    pred_id = resp.json()["id"]

    status_url = REPLICATE_PREDICTION_STATUS_URL.format(pred_id=pred_id)
    start = time.time()
    while time.time() - start < timeout_sec:
        s = requests.get(status_url, headers=headers, timeout=30).json()
        status = s.get("status", "")
        if status == "succeeded":
            output = s.get("output")
            url = output if isinstance(output, str) else (output[0] if output else None)
            if not url:
                raise RuntimeError(f"Replicate succeeded but no output URL: {s}")
            img = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SEC).content
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp.write_bytes(img)
            os.replace(tmp, out_path)
            return pred_id
        if status in ("failed", "canceled"):
            raise RuntimeError(
                f"Replicate prediction {pred_id} {status}: {s.get('error', s)}"
            )
        if poll_interval > 0:
            time.sleep(poll_interval)
    raise RuntimeError(f"Replicate prediction {pred_id} timed out after {timeout_sec}s")
```

- [ ] **Step 4: Run tests — must PASS**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_image_pipeline.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/loopvid/image_pipeline.py test_loopvid_image_pipeline.py
git commit -m "feat(loopvid): Replicate Seedream 4.5 caller with atomic write

Verified omits seed (would 422), negative_prompt (not in schema), and
image_input (we want text-to-image). Returns prediction_id for manifest."
```

---

## Phase 8 — Video Pipeline

### Task 8: Audio chunk slicing + 6 sequential LTX calls

**Files:**
- Create: `scripts/loopvid/video_pipeline.py`
- Create: `test_loopvid_video_pipeline.py`

- [ ] **Step 1: Write tests**

Create `/root/ace-step-music-xl/test_loopvid_video_pipeline.py`:

```python
import base64
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import responses

from scripts.loopvid.video_pipeline import (
    slice_audio_chunks,
    build_clip_payload,
    stable_clip_seed,
    run_video_pipeline,
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
```

- [ ] **Step 2: Run tests — must FAIL**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_video_pipeline.py -v 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement video_pipeline.py**

Create `/root/ace-step-music-xl/scripts/loopvid/video_pipeline.py`:

```python
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
        tmp = target.with_suffix(target.suffix + ".tmp")
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
) -> dict:
    return {
        "input": {
            "image_base64": image_b64,
            "audio_base64": audio_b64,
            "prompt": motion_prompt,
            "negative_prompt": LTX_NEGATIVE_PROMPT,
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
        )
        body = run_segment(
            endpoint_id=endpoint_id, api_key=api_key, payload=payload,
            label=f"video clip {i}",
            max_retries=1,   # LTX timeout is 1200s; one retry is enough
        )
        output = body.get("output", {})
        if "error" in output:
            raise RuntimeError(f"LTX clip {i} returned error: {output['error']}")
        _save_clip_video(output, target)
        clips.append(target)
        if on_clip_done:
            on_clip_done(i, target)
    return clips
```

- [ ] **Step 4: Run tests — must PASS**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_video_pipeline.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/loopvid/video_pipeline.py test_loopvid_video_pipeline.py
git commit -m "feat(loopvid): video pipeline — audio chunk slicing + 6 sequential LTX clips

Each clip is conditioned on its corresponding 7s music slice for
music-reactive motion. Resume-safe via per-clip canonical files."
```

---

## Phase 9 — Music Pipeline

### Task 9: 11-segment ACE-Step runner + stitch

**Files:**
- Create: `scripts/loopvid/music_pipeline.py`
- Create: `test_loopvid_music_pipeline.py`

- [ ] **Step 1: Write tests**

Create `/root/ace-step-music-xl/test_loopvid_music_pipeline.py`:

```python
import base64
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import responses

from scripts.loopvid.music_pipeline import (
    build_segment_payload,
    run_music_pipeline,
    stitch_segments,
)
from scripts.loopvid.constants import ACE_STEP_PRESET, SEGMENT_DURATION_SEC, CROSSFADE_SEC


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
    # Pre-create one segment
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
        # Return a tiny silent mp3 in audio_base64
        import io
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
    # Should have called for segs 1 and 2 only (not 3 — already present)
    assert call_count["n"] == 2
    assert (out_dir / "seg_01.mp3").exists()
    assert (out_dir / "seg_02.mp3").exists()
```

- [ ] **Step 2: Run tests — must FAIL**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_music_pipeline.py -v 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement music_pipeline.py**

Create `/root/ace-step-music-xl/scripts/loopvid/music_pipeline.py`:

```python
"""ACE-Step XL 11-segment music pipeline + ffmpeg stitch."""
from __future__ import annotations

import base64
import os
import subprocess
from pathlib import Path
from typing import Callable, Optional

from scripts.loopvid.constants import (
    ACE_STEP_PRESET, SEGMENT_DURATION_SEC, CROSSFADE_SEC,
)
from scripts.loopvid.runpod_client import run_segment


def build_segment_payload(*, prompt: str, duration: int, seed: int) -> dict:
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
            **ACE_STEP_PRESET,
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
    on_segment_done: Optional[Callable[[int, Path], None]] = None,
) -> list[Path]:
    """Submit N segments sequentially. Skips canonical files that already exist.
    Returns the segment paths in order."""
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
        payload = build_segment_payload(prompt=prompt, duration=duration_sec, seed=seed)
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
    """Chained acrossfade with equal-power qsin curves. Writes to out_path."""
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
```

- [ ] **Step 4: Run tests — must PASS**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_music_pipeline.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/loopvid/music_pipeline.py test_loopvid_music_pipeline.py
git commit -m "feat(loopvid): 11-segment ACE-Step pipeline + stitch

Resumes by skipping seg_NN.mp3 that already exist on disk."
```

---

## Phase 10 — Loop Build (ffmpeg concat + xfade + seam)

### Task 10: Loop builder

**Files:**
- Create: `scripts/loopvid/loop_build.py`
- Create: `test_loopvid_loop_build.py`

- [ ] **Step 1: Write tests**

Create `/root/ace-step-music-xl/test_loopvid_loop_build.py`:

```python
import subprocess
from pathlib import Path

import pytest

from scripts.loopvid.loop_build import (
    concat_clips_with_xfades,
    add_loop_seam_fade,
)


def make_color_clip(path: Path, color: str, duration: float = 7.0):
    """Synthesize a solid-color H.264 video clip at 1280x704 24fps for tests."""
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c={color}:s=1280x704:r=24:d={duration}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-tune", "stillimage",
        "-an", str(path),
    ], capture_output=True, check=True)


def probe_duration(path: Path) -> float:
    r = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def probe_dimensions(path: Path) -> tuple[int, int]:
    r = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path),
    ], capture_output=True, text=True, check=True)
    w, h = r.stdout.strip().split(",")
    return int(w), int(h)


def test_concat_clips_with_xfades_produces_expected_duration(tmp_path):
    clips = []
    for i, c in enumerate(["red", "green", "blue", "yellow", "cyan", "magenta"], start=1):
        p = tmp_path / f"clip_{i:02d}.mp4"
        make_color_clip(p, c, duration=7.0)
        clips.append(p)
    out = tmp_path / "concat.mp4"
    concat_clips_with_xfades(clips, out, xfade_sec=0.25)
    d = probe_duration(out)
    # 6 × 7 − 5 × 0.25 = 40.75s
    assert 40.5 < d < 41.0, f"got duration {d}"


def test_concat_preserves_dimensions(tmp_path):
    clips = []
    for i in range(2):
        p = tmp_path / f"c{i}.mp4"
        make_color_clip(p, "red", duration=3)
        clips.append(p)
    out = tmp_path / "concat.mp4"
    concat_clips_with_xfades(clips, out, xfade_sec=0.25)
    assert probe_dimensions(out) == (1280, 704)


def test_concat_atomic_write(tmp_path):
    clips = []
    for i in range(2):
        p = tmp_path / f"c{i}.mp4"
        make_color_clip(p, "red", duration=3)
        clips.append(p)
    out = tmp_path / "concat.mp4"
    concat_clips_with_xfades(clips, out, xfade_sec=0.25)
    assert not (tmp_path / "concat.mp4.tmp").exists()


def test_add_loop_seam_fade_shortens_by_fade_duration(tmp_path):
    base = tmp_path / "base.mp4"
    make_color_clip(base, "red", duration=10.0)
    out = tmp_path / "seamed.mp4"
    add_loop_seam_fade(base, out, fade_sec=0.5)
    d = probe_duration(out)
    # 10s minus 0.5s seam fade ≈ 9.5s
    assert 9.3 < d < 9.7
```

- [ ] **Step 2: Run tests — must FAIL**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_loop_build.py -v 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement loop_build.py**

Create `/root/ace-step-music-xl/scripts/loopvid/loop_build.py`:

```python
"""ffmpeg pipeline: concat 6 clips with xfades, then add loop-seam fade."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from scripts.loopvid.constants import INTER_CLIP_XFADE_SEC, LOOP_SEAM_XFADE_SEC


def _probe_duration(path: Path) -> float:
    r = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def concat_clips_with_xfades(
    clips: list[Path], out_path: Path,
    *, xfade_sec: float = INTER_CLIP_XFADE_SEC,
) -> None:
    """Concatenate clips with xfade transitions between adjacent pairs.
    Atomic write."""
    if len(clips) < 2:
        raise ValueError(f"need at least 2 clips, got {len(clips)}")
    out_path = Path(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")

    # Build chained xfade filter. Track running offset for the offset= param.
    inputs = []
    for c in clips:
        inputs += ["-i", str(c)]

    durations = [_probe_duration(c) for c in clips]
    filters = []
    prev_label = "0:v"
    running_offset = 0.0
    for i in range(1, len(clips)):
        next_label = f"v{i:02d}"
        offset = running_offset + durations[i - 1] - xfade_sec
        running_offset = offset
        filters.append(
            f"[{prev_label}][{i}:v]xfade=transition=fade:duration={xfade_sec}:"
            f"offset={offset}[{next_label}]"
        )
        prev_label = next_label

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", ";".join(filters),
        "-map", f"[{prev_label}]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat failed (exit {result.returncode}):\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    os.replace(tmp, out_path)


def add_loop_seam_fade(
    base: Path, out_path: Path,
    *, fade_sec: float = LOOP_SEAM_XFADE_SEC,
) -> None:
    """Make the file loop-seamlessly by xfading the last fade_sec into the first
    fade_sec. Final duration = original − fade_sec."""
    base = Path(base)
    out_path = Path(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    duration = _probe_duration(base)
    head_offset = 0.0
    tail_offset = duration - fade_sec

    # Two split inputs: tail (last fade_sec) and head-to-end (everything except trailing fade)
    # Strategy: trim front [0, dur-fade], trim tail [dur-fade, dur], then xfade the tail's start
    # over the front's end. Simpler approach: re-encode with a self-fade overlap.
    cmd = [
        "ffmpeg", "-y", "-i", str(base),
        "-filter_complex",
        # Split into two streams; trim first to [0, dur-fade], trim second to [dur-fade, dur].
        # Then xfade the tail over the start of the front (which would loop).
        # This produces a clip where the last fade_sec fades into a copy of the first fade_sec.
        f"[0:v]split=2[front][tail];"
        f"[front]trim=0:{tail_offset},setpts=PTS-STARTPTS[a];"
        f"[tail]trim={tail_offset}:{duration},setpts=PTS-STARTPTS[b];"
        f"[a][b]xfade=transition=fade:duration={fade_sec}:offset={tail_offset - fade_sec}[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg seam fade failed (exit {result.returncode}):\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    os.replace(tmp, out_path)
```

- [ ] **Step 4: Run tests — must PASS**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_loop_build.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/loopvid/loop_build.py test_loopvid_loop_build.py
git commit -m "feat(loopvid): ffmpeg loop builder — concat with xfades + seam fade"
```

---

## Phase 11 — Mux

### Task 11: Final assembly — stream-loop video, trim audio, mux

**Files:**
- Create: `scripts/loopvid/mux.py`
- Create: `test_loopvid_mux.py`

- [ ] **Step 1: Write tests**

Create `/root/ace-step-music-xl/test_loopvid_mux.py`:

```python
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
    # Output should still be H.264 (no re-encode) — fast and lossless
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
```

- [ ] **Step 2: Run tests — must FAIL**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_mux.py -v 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement mux.py**

Create `/root/ace-step-music-xl/scripts/loopvid/mux.py`:

```python
"""Final assembly — stream-loop video to target, trim audio, mux."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _atomic_run(cmd: list[str], target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    cmd = list(cmd)
    cmd[-1] = str(tmp)   # caller passes target as last arg
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    os.replace(tmp, target)


def stream_loop_video(base: Path, out: Path, *, target_sec: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(base),
        "-t", str(target_sec), "-c:v", "copy", "-an", str(out),
    ]
    _atomic_run(cmd, Path(out))


def trim_audio(src: Path, out: Path, *, target_sec: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-t", str(target_sec), "-c:a", "copy", str(out),
    ]
    _atomic_run(cmd, Path(out))


def mux_video_audio(video: Path, audio: Path, out: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video), "-i", str(audio),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(out),
    ]
    _atomic_run(cmd, Path(out))


def final_assembly(
    loop_seamed: Path, music_master: Path, out: Path,
    *, target_sec: int, work_dir: Path,
) -> Path:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    video_full = work_dir / "video_60min.mp4"
    audio_full = work_dir / "music_60min.mp3"
    stream_loop_video(loop_seamed, video_full, target_sec=target_sec)
    trim_audio(music_master, audio_full, target_sec=target_sec)
    mux_video_audio(video_full, audio_full, Path(out))
    return Path(out)
```

- [ ] **Step 4: Run tests — must PASS**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_mux.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/loopvid/mux.py test_loopvid_mux.py
git commit -m "feat(loopvid): final mux — stream-loop, trim, combine"
```

---

## Phase 12 — Preflight

### Task 12: Pre-flight checks (env vars, endpoint workersMax, ffmpeg)

**Files:**
- Create: `scripts/loopvid/preflight.py`
- Create: `test_loopvid_preflight.py`

- [ ] **Step 1: Write tests**

Create `/root/ace-step-music-xl/test_loopvid_preflight.py`:

```python
from unittest.mock import patch

import pytest

from scripts.loopvid.preflight import (
    PreflightError,
    check_env_vars,
    check_ffmpeg_available,
    check_endpoint_workers,
    run_preflight,
)


def test_check_env_vars_passes_when_all_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("REPLICATE_API_TOKEN", "x")
    monkeypatch.setenv("RUNPOD_API_KEY", "x")
    check_env_vars(("OPENROUTER_API_KEY", "REPLICATE_API_TOKEN", "RUNPOD_API_KEY"))


def test_check_env_vars_raises_with_missing_var_named(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(PreflightError, match="OPENROUTER_API_KEY"):
        check_env_vars(("OPENROUTER_API_KEY",))


def test_check_ffmpeg_available_passes_when_present():
    check_ffmpeg_available()  # ffmpeg is required for the test suite anyway


def test_check_ffmpeg_available_raises_when_missing():
    with patch("shutil.which", return_value=None):
        with pytest.raises(PreflightError, match="ffmpeg"):
            check_ffmpeg_available()


def test_check_endpoint_workers_passes_when_max_ge_1():
    fake_response = {"workersMax": 1}
    with patch("scripts.loopvid.preflight._get_endpoint", return_value=fake_response):
        check_endpoint_workers("ep-1", "k")


def test_check_endpoint_workers_raises_when_max_zero():
    fake_response = {"workersMax": 0}
    with patch("scripts.loopvid.preflight._get_endpoint", return_value=fake_response):
        with pytest.raises(PreflightError, match="workersMax"):
            check_endpoint_workers("nwqnd0duxc6o38", "k")
```

- [ ] **Step 2: Run tests — must FAIL**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_preflight.py -v 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement preflight.py**

Create `/root/ace-step-music-xl/scripts/loopvid/preflight.py`:

```python
"""Pre-flight checks — fail fast before any paid API call."""
from __future__ import annotations

import os
import shutil

import requests


class PreflightError(RuntimeError):
    pass


def check_env_vars(names: tuple[str, ...]) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise PreflightError(
            f"Missing required env vars: {', '.join(missing)}. "
            f"Source from /root/avatar-video/.env or export explicitly."
        )


def check_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise PreflightError("ffmpeg not found on $PATH — install ffmpeg before running")


def _get_endpoint(endpoint_id: str, api_key: str) -> dict:
    """Fetch endpoint metadata via RunPod GraphQL. Wrapped for test mocking."""
    url = "https://api.runpod.io/graphql"
    query = """query($id: String!) {
        endpoint(id: $id) {
            id
            name
            workersMax
        }
    }"""
    resp = requests.post(
        url,
        json={"query": query, "variables": {"id": endpoint_id}},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json().get("data", {}).get("endpoint")
    if not data:
        raise PreflightError(f"Endpoint {endpoint_id} not found or inaccessible")
    return data


def check_endpoint_workers(endpoint_id: str, api_key: str) -> None:
    info = _get_endpoint(endpoint_id, api_key)
    if info.get("workersMax", 0) < 1:
        raise PreflightError(
            f"Endpoint {endpoint_id} has workersMax={info.get('workersMax')}. "
            f"Run: rp endpoint update {endpoint_id} --workers-max 1"
        )


def run_preflight(
    *, runpod_api_key: str,
    ace_step_endpoint: str,
    ltx_endpoint: str,
    require_ace_step: bool = True,
    require_ltx: bool = True,
) -> None:
    check_env_vars(("OPENROUTER_API_KEY", "REPLICATE_API_TOKEN", "RUNPOD_API_KEY"))
    check_ffmpeg_available()
    if require_ace_step:
        check_endpoint_workers(ace_step_endpoint, runpod_api_key)
    if require_ltx:
        check_endpoint_workers(ltx_endpoint, runpod_api_key)
```

- [ ] **Step 4: Run tests — must PASS**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_preflight.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/loopvid/preflight.py test_loopvid_preflight.py
git commit -m "feat(loopvid): preflight — env vars, ffmpeg, endpoint workersMax"
```

---

## Phase 13 — Orchestrator state machine

### Task 13: Orchestrator — walks steps, skips done, runs pending

**Files:**
- Create: `scripts/loopvid/orchestrator.py`
- Create: `test_loopvid_orchestrator.py`

- [ ] **Step 1: Write tests for the orchestrator**

Create `/root/ace-step-music-xl/test_loopvid_orchestrator.py`:

```python
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scripts.loopvid.orchestrator import run_orchestrator, OrchestratorConfig
from scripts.loopvid.manifest import load_manifest, new_manifest, save_manifest


@pytest.fixture
def cfg(tmp_path):
    return OrchestratorConfig(
        run_id="run-1",
        run_dir=tmp_path / "run-1",
        genre="ambient",
        mood="x",
        duration_sec=3600,
        ace_step_endpoint="ep-music",
        ltx_endpoint="ep-video",
        runpod_api_key="rp",
        openrouter_api_key="or",
        replicate_api_token="rep",
        only=None,
        skip=None,
        force=False,
        dry_run=False,
    )


def fake_plan_module(plan_obj):
    """Builds a mock plan() that returns the given plan object."""
    return MagicMock(return_value=plan_obj)


def test_dry_run_does_not_call_apis(cfg):
    with patch("scripts.loopvid.orchestrator.plan") as mp, \
         patch("scripts.loopvid.orchestrator.run_music_pipeline") as mm, \
         patch("scripts.loopvid.orchestrator.generate_still") as mi, \
         patch("scripts.loopvid.orchestrator.run_video_pipeline") as mv:
        cfg = cfg._replace(dry_run=True) if hasattr(cfg, "_replace") else \
              OrchestratorConfig(**{**cfg.__dict__, "dry_run": True})
        run_orchestrator(cfg)
        mp.assert_not_called()
        mm.assert_not_called()
        mi.assert_not_called()
        mv.assert_not_called()


def test_orchestrator_creates_manifest(cfg):
    with patch("scripts.loopvid.orchestrator.run_preflight"), \
         patch("scripts.loopvid.orchestrator.plan") as mp, \
         patch("scripts.loopvid.orchestrator.run_music_pipeline"), \
         patch("scripts.loopvid.orchestrator.generate_still"), \
         patch("scripts.loopvid.orchestrator.run_video_pipeline"), \
         patch("scripts.loopvid.orchestrator.stitch_segments"), \
         patch("scripts.loopvid.orchestrator.slice_audio_chunks", return_value=[]), \
         patch("scripts.loopvid.orchestrator.concat_clips_with_xfades"), \
         patch("scripts.loopvid.orchestrator.add_loop_seam_fade"), \
         patch("scripts.loopvid.orchestrator.final_assembly"):
        mp.return_value = _fake_plan_obj()
        run_orchestrator(cfg)
        assert (cfg.run_dir / "manifest.json").exists()


def test_resume_skips_done_steps(cfg):
    """Pre-mark plan as done; the orchestrator should not call plan() again."""
    cfg.run_dir.mkdir(parents=True)
    m = new_manifest("run-1", {})
    m.steps["plan"]["status"] = "done"
    save_manifest(cfg.run_dir, m)
    # Pre-create plan.json so downstream steps can read it
    (cfg.run_dir / "plan.json").write_text(json.dumps(_fake_plan_dict()))

    with patch("scripts.loopvid.orchestrator.run_preflight"), \
         patch("scripts.loopvid.orchestrator.plan") as mp, \
         patch("scripts.loopvid.orchestrator.run_music_pipeline"), \
         patch("scripts.loopvid.orchestrator.generate_still"), \
         patch("scripts.loopvid.orchestrator.run_video_pipeline"), \
         patch("scripts.loopvid.orchestrator.stitch_segments"), \
         patch("scripts.loopvid.orchestrator.slice_audio_chunks", return_value=[]), \
         patch("scripts.loopvid.orchestrator.concat_clips_with_xfades"), \
         patch("scripts.loopvid.orchestrator.add_loop_seam_fade"), \
         patch("scripts.loopvid.orchestrator.final_assembly"):
        run_orchestrator(cfg)
        mp.assert_not_called()


def test_only_runs_specified_steps(cfg):
    cfg.run_dir.mkdir(parents=True)
    # Mark all earlier steps as done
    m = new_manifest("run-1", {})
    for s in ("plan", "music", "image", "video", "loop_build"):
        m.steps[s]["status"] = "done"
    save_manifest(cfg.run_dir, m)
    (cfg.run_dir / "plan.json").write_text(json.dumps(_fake_plan_dict()))

    with patch("scripts.loopvid.orchestrator.run_preflight"), \
         patch("scripts.loopvid.orchestrator.final_assembly") as mf:
        new_cfg = OrchestratorConfig(**{**cfg.__dict__, "only": ("mux",)})
        run_orchestrator(new_cfg)
        mf.assert_called_once()


def _fake_plan_dict():
    return {
        "genre": "ambient", "mood": "x",
        "music_palette": "Ambient drone, 60 BPM, no percussion, no vocals, "
                         "warm pads, plate reverb",
        "music_segment_descriptors": [{"phase": f"p{i}", "descriptors": f"d{i}"} for i in range(1, 12)],
        "music_bpm": 60,
        "seedream_scene": "Mountain ridge at golden hour",
        "seedream_style": "35mm film, soft golden light",
        "motion_prompts": [f"m{i}" for i in range(1, 7)],
        "motion_archetype": "mist",
        "image_archetype_key": "mountain_ridge_dusk",
    }


def _fake_plan_obj():
    from scripts.loopvid.plan_schema import validate_plan_dict
    return validate_plan_dict(_fake_plan_dict())
```

- [ ] **Step 2: Run tests — must FAIL**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_orchestrator.py -v 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement orchestrator.py**

Create `/root/ace-step-music-xl/scripts/loopvid/orchestrator.py`:

```python
"""Top-level orchestrator — walks the 6 pipeline steps in order."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from scripts.loopvid.constants import (
    SEGMENT_COUNT_60MIN, SEGMENT_DURATION_SEC,
)
from scripts.loopvid.cost import estimate_run_cost, cost_breakdown_lines, enforce_budget
from scripts.loopvid.image_pipeline import generate_still, build_seedream_prompt
from scripts.loopvid.llm_planner import plan
from scripts.loopvid.loop_build import concat_clips_with_xfades, add_loop_seam_fade
from scripts.loopvid.manifest import (
    load_manifest, new_manifest, save_manifest,
    mark_step_done, mark_step_failed, mark_step_in_progress,
)
from scripts.loopvid.music_pipeline import run_music_pipeline, stitch_segments
from scripts.loopvid.mux import final_assembly
from scripts.loopvid.plan_schema import validate_plan_dict
from scripts.loopvid.preflight import run_preflight
from scripts.loopvid.video_pipeline import (
    run_video_pipeline, slice_audio_chunks, stable_clip_seed,
)


@dataclass
class OrchestratorConfig:
    run_id: str
    run_dir: Path
    genre: str
    mood: str
    duration_sec: int
    ace_step_endpoint: str
    ltx_endpoint: str
    runpod_api_key: str
    openrouter_api_key: str
    replicate_api_token: str
    only: Optional[tuple] = None
    skip: Optional[tuple] = None
    force: bool = False
    dry_run: bool = False
    max_cost: Optional[float] = None


STEP_ORDER = ("plan", "music", "image", "video", "loop_build", "mux")


def _should_run(step: str, cfg: OrchestratorConfig, current_status: str) -> bool:
    if cfg.only and step not in cfg.only:
        return False
    if cfg.skip and step in cfg.skip:
        return False
    if cfg.force:
        return True
    return current_status != "done"


def _print(msg: str) -> None:
    print(msg, flush=True)


def run_orchestrator(cfg: OrchestratorConfig) -> Path:
    """Execute pipeline. Returns path to final.mp4."""
    cfg.run_dir = Path(cfg.run_dir)

    if cfg.dry_run:
        _print(f"[DRY RUN] genre={cfg.genre} mood={cfg.mood} duration={cfg.duration_sec}s")
        for line in cost_breakdown_lines(duration_sec=cfg.duration_sec):
            _print(line)
        return cfg.run_dir / "final.mp4"

    # Cost guard
    estimated = estimate_run_cost(duration_sec=cfg.duration_sec)
    if cfg.max_cost is not None:
        enforce_budget(estimated, cfg.max_cost)

    # Preflight
    run_preflight(
        runpod_api_key=cfg.runpod_api_key,
        ace_step_endpoint=cfg.ace_step_endpoint,
        ltx_endpoint=cfg.ltx_endpoint,
    )

    # Manifest setup
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    try:
        m = load_manifest(cfg.run_dir)
    except FileNotFoundError:
        m = new_manifest(cfg.run_id, {
            "genre": cfg.genre, "mood": cfg.mood, "duration_sec": cfg.duration_sec,
        }, endpoints={"ltx": cfg.ltx_endpoint, "ace_step": cfg.ace_step_endpoint})
        save_manifest(cfg.run_dir, m)

    plan_path = cfg.run_dir / "plan.json"

    # Step 1: plan
    if _should_run("plan", cfg, m.steps["plan"]["status"]):
        mark_step_in_progress(cfg.run_dir, "plan")
        _print("▸ plan (LLM)")
        plan_obj = plan(
            genre=cfg.genre, mood=cfg.mood,
            api_key=cfg.openrouter_api_key,
            raw_response_path=str(cfg.run_dir / "plan_raw.json"),
        )
        plan_path.write_text(json.dumps(plan_obj.__dict__, indent=2, sort_keys=True))
        mark_step_done(cfg.run_dir, "plan")
        _print("✓ plan committed")
    else:
        _print("✓ plan (cached)")

    # Load plan_obj for downstream steps
    plan_dict = json.loads(plan_path.read_text())
    plan_obj = validate_plan_dict(plan_dict)

    # Step 2: music
    music_dir = cfg.run_dir / "music"
    master_path = music_dir / "master.mp3"
    if _should_run("music", cfg, m.steps["music"]["status"]):
        mark_step_in_progress(cfg.run_dir, "music")
        _print(f"▸ music ({SEGMENT_COUNT_60MIN} segments × {SEGMENT_DURATION_SEC}s)")
        prompts = [
            f"{plan_obj.music_palette}, {seg['descriptors']}"
            for seg in plan_obj.music_segment_descriptors
        ]
        seeds = [stable_clip_seed(cfg.run_id, i) for i in range(1, SEGMENT_COUNT_60MIN + 1)]
        seg_paths = run_music_pipeline(
            prompts=prompts, duration_sec=SEGMENT_DURATION_SEC, seeds=seeds,
            out_dir=music_dir,
            endpoint_id=cfg.ace_step_endpoint, api_key=cfg.runpod_api_key,
            on_segment_done=lambda i, p: _print(f"  ✓ seg {i} ({p.stat().st_size:,} B)"),
        )
        if not master_path.exists():
            stitch_segments(seg_paths, master_path)
        mark_step_done(cfg.run_dir, "music", extra={"master_committed": True})
        _print("✓ music master committed")
    else:
        _print("✓ music (cached)")

    # Step 3: image
    still_path = cfg.run_dir / "still.png"
    if _should_run("image", cfg, m.steps["image"]["status"]):
        mark_step_in_progress(cfg.run_dir, "image")
        _print("▸ image (Seedream)")
        prompt_str = build_seedream_prompt(plan_obj.seedream_scene, plan_obj.seedream_style)
        pred_id = generate_still(
            prompt=prompt_str, api_token=cfg.replicate_api_token, out_path=still_path,
        )
        mark_step_done(cfg.run_dir, "image", extra={"prediction_id": pred_id})
        _print(f"✓ image committed (prediction_id={pred_id})")
    else:
        _print("✓ image (cached)")

    # Step 4: video
    video_dir = cfg.run_dir / "video"
    audio_chunks_dir = video_dir / "audio_chunks"
    if _should_run("video", cfg, m.steps["video"]["status"]):
        mark_step_in_progress(cfg.run_dir, "video")
        _print("▸ video (LTX × 6)")
        chunks = slice_audio_chunks(master_path, audio_chunks_dir)
        run_video_pipeline(
            run_id=cfg.run_id,
            still_path=still_path,
            audio_chunks=chunks,
            motion_prompts=plan_obj.motion_prompts,
            out_dir=video_dir,
            endpoint_id=cfg.ltx_endpoint, api_key=cfg.runpod_api_key,
            on_clip_done=lambda i, p: _print(f"  ✓ clip {i}"),
        )
        mark_step_done(cfg.run_dir, "video")
        _print("✓ video clips committed")
    else:
        _print("✓ video (cached)")

    # Step 5: loop_build
    concat_path = video_dir / "concat_42s.mp4"
    seamed_path = video_dir / "loop_seamed.mp4"
    if _should_run("loop_build", cfg, m.steps["loop_build"]["status"]):
        mark_step_in_progress(cfg.run_dir, "loop_build")
        _print("▸ loop_build (ffmpeg)")
        clips = sorted(video_dir.glob("clip_*.mp4"))
        concat_clips_with_xfades(clips, concat_path)
        add_loop_seam_fade(concat_path, seamed_path)
        mark_step_done(cfg.run_dir, "loop_build")
        _print("✓ loop_seamed.mp4 committed")
    else:
        _print("✓ loop_build (cached)")

    # Step 6: mux
    final_path = cfg.run_dir / "final.mp4"
    if _should_run("mux", cfg, m.steps["mux"]["status"]):
        mark_step_in_progress(cfg.run_dir, "mux")
        _print("▸ mux (ffmpeg)")
        final_assembly(
            seamed_path, master_path, final_path,
            target_sec=cfg.duration_sec, work_dir=cfg.run_dir / "_work",
        )
        mark_step_done(cfg.run_dir, "mux")
        _print(f"✓ final.mp4: {final_path}")
    else:
        _print(f"✓ mux (cached): {final_path}")

    return final_path
```

- [ ] **Step 4: Run tests — must PASS**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_orchestrator.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/loopvid/orchestrator.py test_loopvid_orchestrator.py
git commit -m "feat(loopvid): orchestrator state machine with resume + only/skip/force"
```

---

## Phase 14 — Rollback CLI

### Task 14: Three-level rollback (forensic, --keep, --hard)

**Files:**
- Create: `scripts/loopvid/rollback.py`
- Create: `test_loopvid_rollback.py`

- [ ] **Step 1: Write tests**

Create `/root/ace-step-music-xl/test_loopvid_rollback.py`:

```python
import json
from pathlib import Path

import pytest

from scripts.loopvid.rollback import (
    rollback_forensic,
    rollback_with_keep,
    rollback_hard,
    RollbackError,
)
from scripts.loopvid.manifest import new_manifest, save_manifest, load_manifest


def make_run_dir(tmp_path: Path, run_id: str = "run-1") -> Path:
    run_dir = tmp_path / "out" / "loop_video" / run_id
    run_dir.mkdir(parents=True)
    save_manifest(run_dir, new_manifest(run_id, {}))
    (run_dir / "plan.json").write_text("{}")
    (run_dir / "still.png").write_bytes(b"fake png")
    (run_dir / "music").mkdir()
    (run_dir / "music" / "master.mp3").write_bytes(b"fake mp3")
    return run_dir


def test_rollback_forensic_renames_to_failed_timestamp(tmp_path):
    run_dir = make_run_dir(tmp_path)
    rollback_forensic(run_dir)
    assert not run_dir.exists()
    failed = list(run_dir.parent.glob("run-1.failed-*"))
    assert len(failed) == 1
    assert (failed[0] / "manifest.json").exists()


def test_rollback_with_keep_music_preserves_plan_and_master(tmp_path):
    run_dir = make_run_dir(tmp_path)
    rollback_with_keep(run_dir, keep=("music",))
    assert run_dir.exists()
    assert (run_dir / "plan.json").exists()
    assert (run_dir / "music" / "master.mp3").exists()
    # still.png should be in the failed dir, not in clean run_dir
    assert not (run_dir / "still.png").exists()
    failed = list(run_dir.parent.glob("run-1.failed-*"))
    assert len(failed) == 1


def test_rollback_with_keep_music_image_preserves_both(tmp_path):
    run_dir = make_run_dir(tmp_path)
    rollback_with_keep(run_dir, keep=("music", "image"))
    assert run_dir.exists()
    assert (run_dir / "plan.json").exists()
    assert (run_dir / "music" / "master.mp3").exists()
    assert (run_dir / "still.png").exists()


def test_rollback_with_keep_resets_downstream_status_to_pending(tmp_path):
    run_dir = make_run_dir(tmp_path)
    # Mark all steps as done first
    m = load_manifest(run_dir)
    for s in m.steps:
        m.steps[s]["status"] = "done"
    save_manifest(run_dir, m)

    rollback_with_keep(run_dir, keep=("music",))
    m2 = load_manifest(run_dir)
    assert m2.steps["plan"]["status"] == "done"
    assert m2.steps["music"]["status"] == "done"
    assert m2.steps["image"]["status"] == "pending"
    assert m2.steps["video"]["status"] == "pending"
    assert m2.steps["mux"]["status"] == "pending"


def test_rollback_hard_requires_confirm_y(tmp_path):
    run_dir = make_run_dir(tmp_path)
    rollback_hard(run_dir, confirm=True)
    assert not run_dir.exists()


def test_rollback_hard_aborts_without_confirm(tmp_path):
    run_dir = make_run_dir(tmp_path)
    with pytest.raises(RollbackError, match="confirm"):
        rollback_hard(run_dir, confirm=False)
    assert run_dir.exists()
```

- [ ] **Step 2: Run tests — must FAIL**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_rollback.py -v 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement rollback.py**

Create `/root/ace-step-music-xl/scripts/loopvid/rollback.py`:

```python
"""Three-level rollback for failed loop_music_video runs."""
from __future__ import annotations

import datetime as _dt
import shutil
from pathlib import Path

from scripts.loopvid.manifest import load_manifest, save_manifest


class RollbackError(RuntimeError):
    pass


# What each --keep target preserves (always also preserves plan.json).
KEEP_PATHS = {
    "music":   ("music",),                        # whole music/ subdir
    "image":   ("still.png",),
    "video":   ("video",),                        # whole video/ subdir
}

PLAN_FILES = ("plan.json", "plan_raw.json")


def _failed_dir(run_dir: Path) -> Path:
    """Generate timestamped sibling path for forensic preservation."""
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return run_dir.parent / f"{run_dir.name}.failed-{ts}"


def rollback_forensic(run_dir: Path) -> Path:
    """Move the entire run dir to <run_id>.failed-<ts>/ and return new path."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise RollbackError(f"run dir {run_dir} does not exist")
    target = _failed_dir(run_dir)
    run_dir.rename(target)   # atomic on same fs
    return target


def rollback_with_keep(run_dir: Path, *, keep: tuple) -> Path:
    """Preserve plan.json + the items named by `keep` (e.g. ('music',) or
    ('music','image')) in a clean run_dir, move everything else to a
    .failed-<ts>/ sibling. Resets downstream step statuses to pending."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise RollbackError(f"run dir {run_dir} does not exist")

    bad_keys = set(keep) - set(KEEP_PATHS.keys())
    if bad_keys:
        raise RollbackError(
            f"unknown keep targets: {bad_keys}. allowed: {sorted(KEEP_PATHS.keys())}"
        )

    keep_paths = set(PLAN_FILES) | {"manifest.json"}
    for k in keep:
        keep_paths |= set(KEEP_PATHS[k])

    failed = _failed_dir(run_dir)
    failed.mkdir(parents=True)

    # Move everything not in keep_paths to failed/, then keep the rest in run_dir
    for item in list(run_dir.iterdir()):
        if item.name in keep_paths:
            continue
        shutil.move(str(item), str(failed / item.name))

    # Reset manifest: steps after the kept ones become pending
    m = load_manifest(run_dir)
    keep_steps = {"plan"}                         # plan is always kept implicitly
    for k in keep:
        keep_steps.add(k if k != "image" else "image")   # already same name
    for step_name in m.steps:
        if step_name not in keep_steps:
            m.steps[step_name] = {"status": "pending", "attempts": 0}
    save_manifest(run_dir, m)
    return failed


def rollback_hard(run_dir: Path, *, confirm: bool) -> None:
    """Permanently delete the run dir. Requires confirm=True."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise RollbackError(f"run dir {run_dir} does not exist")
    if not confirm:
        raise RollbackError("hard rollback requires confirm=True (--hard --yes from CLI)")
    shutil.rmtree(run_dir)
```

- [ ] **Step 4: Run tests — must PASS**

Run: `cd /root/ace-step-music-xl && pytest test_loopvid_rollback.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/loopvid/rollback.py test_loopvid_rollback.py
git commit -m "feat(loopvid): three-level rollback (forensic / --keep / --hard)"
```

---

## Phase 15 — CLI entry script

### Task 15: scripts/loop_music_video.py — argparse + dispatch

**Files:**
- Create: `scripts/loop_music_video.py`

- [ ] **Step 1: Implement CLI**

Create `/root/ace-step-music-xl/scripts/loop_music_video.py`:

```python
#!/usr/bin/env python3
"""Loop Music Video Generator — CLI entry.

Generates a 1-hour 1280×704 looping music video from a genre+mood input.

Usage:
  python3 scripts/loop_music_video.py --genre ambient
  python3 scripts/loop_music_video.py --genre lofi --mood "rainy autumn evening" \\
      --run-id lofi-rainy-001
  python3 scripts/loop_music_video.py --resume lofi-rainy-001
  python3 scripts/loop_music_video.py --rollback lofi-rainy-001 --keep music
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from pathlib import Path

# Ensure the loopvid package is importable when this script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loopvid.cost import estimate_run_cost, cost_breakdown_lines
from loopvid.orchestrator import OrchestratorConfig, run_orchestrator
from loopvid.rollback import (
    rollback_forensic, rollback_with_keep, rollback_hard, RollbackError,
)


DEFAULT_LTX_ENDPOINT = "1g0pvlx8ar6qns"
DEFAULT_ACE_STEP_ENDPOINT = "nwqnd0duxc6o38"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "out" / "loop_video"


def _autogen_run_id(genre: str) -> str:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{genre}-{ts}"


def _parse_csv(s: str | None) -> tuple | None:
    if s is None:
        return None
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Mode selectors
    p.add_argument("--rollback", help="Rollback the named run instead of running pipeline")
    p.add_argument("--resume", help="Resume the named run from its last completed step")
    p.add_argument("--run-id", help="Explicit run id (default: autogen from genre+timestamp)")

    # Generation params
    p.add_argument("--genre", help="ambient | lofi | jazz | classical | meditation | cinematic | ...")
    p.add_argument("--mood", default="", help="Free-text mood/scene hint")
    p.add_argument("--duration", type=int, default=3600, help="Target seconds (default 3600)")

    # Endpoints
    p.add_argument("--ltx-endpoint", default=os.environ.get("RUNPOD_LTX_ENDPOINT_ID", DEFAULT_LTX_ENDPOINT))
    p.add_argument("--ace-step-endpoint", default=os.environ.get("RUNPOD_ACE_STEP_ENDPOINT_ID", DEFAULT_ACE_STEP_ENDPOINT))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))

    # Step controls
    p.add_argument("--only", help="Comma-separated steps to run only")
    p.add_argument("--skip", help="Comma-separated steps to skip")
    p.add_argument("--force", action="store_true", help="Re-run completed steps (with cost prompt)")
    p.add_argument("--dry-run", action="store_true", help="Show plan + cost, no API calls")

    # Cost guard
    p.add_argument("--max-cost", type=float, help="Abort if estimated cost exceeds this USD value")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompts")

    # Rollback options
    p.add_argument("--keep", help="--rollback: comma-separated keepers (music,image,video)")
    p.add_argument("--hard", action="store_true", help="--rollback: hard delete (with confirm)")

    return p


def cmd_rollback(args, out_dir: Path) -> int:
    run_dir = out_dir / args.rollback
    try:
        if args.hard:
            confirmed = args.yes or _confirm(f"Permanently delete {run_dir}? [y/N] ")
            rollback_hard(run_dir, confirm=confirmed)
            print(f"✓ deleted {run_dir}")
        elif args.keep:
            keep = _parse_csv(args.keep)
            failed = rollback_with_keep(run_dir, keep=keep)
            print(f"✓ kept {keep} in {run_dir}; rest moved to {failed}")
        else:
            failed = rollback_forensic(run_dir)
            print(f"✓ moved {run_dir} to {failed}")
        return 0
    except RollbackError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


def cmd_run(args, out_dir: Path) -> int:
    if not args.genre and not args.resume:
        print("ERROR: --genre is required (or --resume to continue an existing run)", file=sys.stderr)
        return 2
    run_id = args.resume or args.run_id or _autogen_run_id(args.genre)

    # If resuming, infer genre/mood/duration from manifest later (orchestrator
    # already prefers manifest args over CLI for resume).
    cfg = OrchestratorConfig(
        run_id=run_id,
        run_dir=out_dir / run_id,
        genre=args.genre or "ambient",
        mood=args.mood,
        duration_sec=args.duration,
        ace_step_endpoint=args.ace_step_endpoint,
        ltx_endpoint=args.ltx_endpoint,
        runpod_api_key=os.environ.get("RUNPOD_API_KEY", ""),
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        replicate_api_token=os.environ.get("REPLICATE_API_TOKEN", ""),
        only=_parse_csv(args.only),
        skip=_parse_csv(args.skip),
        force=args.force,
        dry_run=args.dry_run,
        max_cost=args.max_cost,
    )

    if not args.dry_run and not args.yes:
        skip = set(cfg.skip or ())
        cost = estimate_run_cost(duration_sec=cfg.duration_sec, skip=skip)
        print(f"Estimated cost: ${cost:.2f}")
        for line in cost_breakdown_lines(duration_sec=cfg.duration_sec, skip=skip):
            print(line)
        if not _confirm("Continue? [y/N] "):
            print("aborted.")
            return 1

    final = run_orchestrator(cfg)
    print(f"Done: {final}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.rollback:
        return cmd_rollback(args, out_dir)
    return cmd_run(args, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test the CLI in dry-run**

Run: `cd /root/ace-step-music-xl && python3 scripts/loop_music_video.py --genre ambient --dry-run --yes`
Expected: prints the cost breakdown lines, exits 0.

- [ ] **Step 3: Commit CLI**

```bash
chmod +x /root/ace-step-music-xl/scripts/loop_music_video.py
git add scripts/loop_music_video.py
git commit -m "feat(loopvid): CLI entry — argparse + dispatch to orchestrator/rollback"
```

---

## Phase 16 — Live smoke test + README

### Task 16: 5-minute end-to-end smoke + README update

**Files:**
- Create: `scripts/smoke/03_loop_music_video_5min.py`
- Modify: `README.md`

- [ ] **Step 1: Create the smoke test**

Create `/root/ace-step-music-xl/scripts/smoke/03_loop_music_video_5min.py`:

```python
#!/usr/bin/env python3
"""Live 5-minute end-to-end smoke test for loop_music_video.

Runs the full orchestrator with --duration 300, exercising every step.
Cost: ~$0.40, wall time ~10 min.

Usage:
  RUNPOD_API_KEY=... OPENROUTER_API_KEY=... REPLICATE_API_TOKEN=... \\
    python3 scripts/smoke/03_loop_music_video_5min.py
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_ID = "smoke-5min"


def main() -> int:
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts/loop_music_video.py"),
        "--genre", "ambient",
        "--mood", "smoke test",
        "--duration", "300",
        "--run-id", RUN_ID,
        "--yes",
    ]
    print(f"running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return result.returncode

    final = REPO_ROOT / "out" / "loop_video" / RUN_ID / "final.mp4"
    if not final.exists():
        print(f"FAIL: {final} not produced")
        return 1

    # Verify duration ≈ 300s, dimensions 1280x704
    probe = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height : format=duration",
        "-of", "csv=p=0", str(final),
    ], capture_output=True, text=True, check=True)
    print(f"ffprobe: {probe.stdout}")
    if "1280,704" not in probe.stdout:
        print("FAIL: dimensions != 1280x704")
        return 1
    print(f"PASS: {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Make executable**

```bash
mkdir -p /root/ace-step-music-xl/scripts/smoke
chmod +x /root/ace-step-music-xl/scripts/smoke/03_loop_music_video_5min.py
```

- [ ] **Step 3: Update README with a "Loop music video" section**

Append to `/root/ace-step-music-xl/README.md`:

```markdown

## Loop music video generator

`scripts/loop_music_video.py` produces a 1-hour 1280×704 looping music video by
composing OpenRouter (Gemini 3 Flash), ACE-Step XL serverless, and LTX-2.3
ComfyUI serverless.

### Quick start

```bash
# Set credentials
export RUNPOD_API_KEY=...
export OPENROUTER_API_KEY=...
export REPLICATE_API_TOKEN=...

# Default: 60-min ambient run
python3 scripts/loop_music_video.py --genre ambient

# Specific genre + mood
python3 scripts/loop_music_video.py --genre lofi --mood "rainy autumn evening, cozy"

# Dry run — show plan + cost, no API calls
python3 scripts/loop_music_video.py --genre jazz --dry-run --yes

# Resume a failed run
python3 scripts/loop_music_video.py --resume <run-id>

# Rollback (forensic — preserves everything in <run-id>.failed-<ts>/)
python3 scripts/loop_music_video.py --rollback <run-id>

# Rollback preserving expensive intermediates
python3 scripts/loop_music_video.py --rollback <run-id> --keep music
python3 scripts/loop_music_video.py --rollback <run-id> --keep music,image
```

### Cost guard

`--max-cost <USD>` aborts the run before any paid call if the estimate exceeds
the budget. Default behavior (no `--max-cost`) prompts for confirmation with
the cost breakdown.

### Live smoke test (paid, ~$0.40, ~10 min wall)

```bash
python3 scripts/smoke/03_loop_music_video_5min.py
```

### Architecture

See `docs/superpowers/specs/2026-04-27-loop-music-video-design.md` for the
full design (LLM Planner contract, music/video pipelines, loop seam math,
manifest state machine, rollback strategy).
```

- [ ] **Step 4: Run the full mocked test suite**

```bash
cd /root/ace-step-music-xl && pytest test_loopvid_*.py -v --cov=scripts/loopvid --cov-report=term-missing 2>&1 | tail -40
```

Expected: all `test_loopvid_*.py` pass. Coverage of `scripts/loopvid/` ≥ 80%.

- [ ] **Step 5: Commit smoke test + README**

```bash
git add scripts/smoke/03_loop_music_video_5min.py README.md
git commit -m "docs+test: add loop music video README section + 5-min smoke test"
```

---

## Phase 17 — Live verification + tag

### Task 17: Run the smoke test and tag the release

- [ ] **Step 1: Confirm endpoints are warm and ready**

```bash
# ACE-Step
mcp__runpod__get-endpoint(endpointId="nwqnd0duxc6o38")
# Expect workersMax >= 1

# LTX (already verified by Plan A)
mcp__runpod__get-endpoint(endpointId="1g0pvlx8ar6qns")
```

- [ ] **Step 2: Run the 5-min smoke**

```bash
cd /root/ace-step-music-xl && python3 scripts/smoke/03_loop_music_video_5min.py
```

Expected: ends with `PASS: out/loop_video/smoke-5min/final.mp4`. If FAIL, investigate via:
- `cat out/loop_video/smoke-5min/manifest.json` — see which step failed
- `out/loop_video/smoke-5min/run.log` — JSONL events
- `out/loop_video/smoke-5min/api/<step>/<call-id>.json` — request/response forensics

- [ ] **Step 3: Verify the artifact plays**

```bash
ffprobe -v quiet -print_format json -show_streams \
    /root/ace-step-music-xl/out/loop_video/smoke-5min/final.mp4 | \
    jq '.streams[] | {codec_type, codec_name, width, height, duration}'
```

Expected: video stream `h264 1280x704 ~300s`, audio stream `aac stereo ~300s`.

- [ ] **Step 4: Tag the release**

```bash
cd /root/ace-step-music-xl
git tag -a loopvid-v1 -m "Loop music video orchestrator v1.0 — 60min 1280x704 looping music video from genre+mood input"
git push origin main loopvid-v1
```

---

## Self-Review

**Spec coverage** (against the design doc sections):

| Spec § | Covered by |
|---|---|
| §1 Overview / §2 Goals | Plan as a whole |
| §4 CLI shape | Task 15 (CLI), Task 16 (README) |
| §5.1 Top-level pipeline | Task 13 (orchestrator) |
| §5.3 File layout | All tasks (paths match) |
| §6 Music-first ordering | Task 13 (sequential step order in orchestrator) |
| §7 LTX handler change | Plan A (separate document) |
| §8.1 LLM Planner contract | Task 6 |
| §8.2 Music pipeline | Task 9 |
| §8.3 Image pipeline | Task 7 |
| §8.4 Video pipeline | Task 8 |
| §8.5 Mux pipeline | Task 11 |
| §9 Loop seam strategy | Task 10 (loop_build) |
| §10 Constants & prompt design | Task 2 (constants module) |
| §11 Manifest schema | Task 3 (manifest module) |
| §12 Error handling, rollback, resume | Tasks 13 (orchestrator) + 14 (rollback) |
| §13 Cost guard | Task 4 (cost) + Task 15 (CLI prompt) |
| §14 Logging contract | Inline in orchestrator (Task 13) — uses _print + atomic manifest writes |
| §15 Testing strategy | Every task has a TDD test cycle |

**Placeholder scan:** Searched for "TBD", "TODO", "implement later", "add error handling later", "fill in details" — none found. Every step has actual code or actual commands. ✓

**Type consistency cross-check:**

- `Plan` dataclass in plan_schema.py — used by llm_planner.py, orchestrator.py, plan_schema.py tests. Field names consistent.
- `RunManifest`, `new_manifest`, `save_manifest`, `load_manifest`, `mark_step_done`, `mark_step_in_progress`, `mark_step_failed` — defined in manifest.py, called from orchestrator.py + tests. Signatures consistent.
- `OrchestratorConfig` dataclass — defined in orchestrator.py, instantiated in CLI (loop_music_video.py) and tests. Field names match.
- `run_segment` — has the shared signature in `loopvid/runpod_client.py` (`payload`, `label`, `max_retries`) and is called that way from music_pipeline.py and video_pipeline.py. The legacy ambient_eno_45min.py wraps it with the older positional-arg signature for back-compat. ✓
- `slice_audio_chunks(master_path, out_dir, count, clip_duration_sec)` — same signature in tests and orchestrator caller. ✓
- `concat_clips_with_xfades(clips, out_path, xfade_sec)` — consistent. ✓
- `add_loop_seam_fade(base, out_path, fade_sec)` — consistent. ✓
- `final_assembly(seamed, master, out, target_sec, work_dir)` — consistent. ✓
- `generate_still(prompt, api_token, out_path, poll_interval, timeout_sec)` — consistent. ✓
- `rollback_*` — consistent. ✓

**Spec gaps fixed inline during writing:**

- Phase 13 orchestrator handles all 6 steps + resume + only/skip/force in a single task to avoid combinatorial explosion.
- Cost-confirmation prompt placed in CLI (Task 15) rather than orchestrator core, so the orchestrator stays unit-testable without stdin mocking.
- Logging is folded into orchestrator (Task 13) rather than a separate module — kept simple per YAGNI.
- `conftest.py` at repo root (Task 1) is the single import-path setup for all tests.

No re-review needed.

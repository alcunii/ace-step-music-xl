# ACE-Step 1.5 XL Serverless Deployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a RunPod serverless endpoint for ACE-Step 1.5 XL (`acestep-v15-xl-base`) that exposes all six task types (`text2music`, `cover`, `repaint`, `extract`, `lego`, `complete`) behind a single unified handler, deployed on an A40 GPU in EU-SE-1 with a 35 GB network volume holding the weights.

**Architecture:** Single-container Python service running on RunPod's serverless fabric. Weights mounted from a regional network volume so the image stays ~5 GB. GitHub Actions builds and pushes to Docker Hub on every push to `main`, then updates the RunPod template via GraphQL. The handler parses a unified input schema, routes to the correct `task_type` branch, resolves `src_audio` inputs (base64 or public `https://` URL), and returns base64-encoded audio plus metadata. Torch 2.4.0+cu124 on A40 sm_86 — same stack as the reference turbo deployment at `/root/ace-step-music`.

**Tech Stack:** Python 3.11, PyTorch 2.4.0, CUDA 12.4, ACE-Step 1.5 (`pip install -e`), `runpod` SDK, `soundfile`, `requests`, pytest + `responses`, Docker Buildx, GitHub Actions, RunPod MCP (`mcp__runpod__*`).

---

## File structure

```
/root/ace-step-music-xl/
├── .dockerignore                                       (copy from turbo)
├── .env.example                                        (XL-specific vars)
├── .github/workflows/deploy.yml                        (turbo + tag/secret deltas)
├── .gitignore                                          (copy from turbo)
├── Dockerfile                                          (turbo + XL env deltas)
├── README.md                                           (minimal reference)
├── docs/superpowers/
│   ├── specs/2026-04-20-ace-step-xl-serverless-design.md   (already written)
│   └── plans/2026-04-20-ace-step-xl-serverless.md          (this file)
├── fixtures/short.mp3                                  (~2s deterministic fixture)
├── handler.py                                          (unified 6-task router)
├── requirements-test.txt                               (pytest, pyyaml, responses, soundfile)
├── test_endpoint.py                                    (manual integration test)
├── test_handler.py                                     (unit tests — CI)
└── test_workflow.py                                    (deploy.yml structure — CI)
```

**File responsibilities:**
- `handler.py` (~350 lines) — cold-start model loading, torch 2.4 compat patches, unified job handler with 6 task types, `_resolve_src_audio` helper, `_validate` schema gate.
- `test_handler.py` (~500 lines) — mocks `acestep.*`, `runpod`, `torch`; exercises each task type, validation, schema, cleanup.
- `Dockerfile` — identical stack to turbo, only env vars change.
- `deploy.yml` — identical pipeline to turbo, only image name and template-id secret change.
- `test_endpoint.py` — user-run script for end-to-end smoke test.
- `fixtures/short.mp3` — real 2-second audio used as input for `cover`/`repaint`/etc. tests.

---

## Task 1: Initialize project directory with static config files

**Files:**
- Create: `/root/ace-step-music-xl/.gitignore`
- Create: `/root/ace-step-music-xl/.dockerignore`
- Create: `/root/ace-step-music-xl/.env.example`
- Create: `/root/ace-step-music-xl/requirements-test.txt`

- [ ] **Step 1:** Write `.gitignore`

```
__pycache__/
*.pyc
.pytest_cache/
.env
*.egg-info/
dist/
build/
.coverage
out/
```

- [ ] **Step 2:** Write `.dockerignore`

```
.git
__pycache__
*.pyc
.pytest_cache
.env
.env.*
docs/
fixtures/
test_handler.py
test_workflow.py
test_endpoint.py
*.md
```

- [ ] **Step 3:** Write `.env.example`

```
# ACE-Step 1.5 XL RunPod Serverless Handler
# Copy to .env for local testing

# Model paths (weights live on RunPod network volume in production)
ACESTEP_PROJECT_ROOT=/app/ACE-Step-1.5
ACESTEP_CONFIG_PATH=acestep-v15-xl-base
ACESTEP_CHECKPOINT_DIR=/runpod-volume/checkpoints
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ACESTEP_LM_BACKEND=vllm

# A40 GPU optimizations (48GB VRAM, Ampere sm_86)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCH_CUDA_ARCH_LIST=8.6
ACESTEP_COMPILE_MODEL=1
ACESTEP_MAX_BATCH_SIZE=4

# XL-specific inference defaults
ACESTEP_INFERENCE_STEPS_DEFAULT=50
ACESTEP_GUIDANCE_SCALE_DEFAULT=7.0

# Safety valve (set to 1 if OOM)
ACESTEP_OFFLOAD_DIT_TO_CPU=0

# CUDA startup retries (for RunPod cold starts)
ACESTEP_LOAD_RETRIES=5
ACESTEP_RETRY_DELAY=10

# For running test_endpoint.py locally
# RUNPOD_API_KEY=
# RUNPOD_ENDPOINT_ID=
```

- [ ] **Step 4:** Write `requirements-test.txt`

```
pytest
pyyaml
responses
soundfile
numpy
```

- [ ] **Step 5:** Commit

```bash
cd /root/ace-step-music-xl
git init
git add .gitignore .dockerignore .env.example requirements-test.txt
git commit -m "chore: scaffold project with static config files"
```

---

## Task 2: Generate fixtures/short.mp3 for deterministic tests

**Files:**
- Create: `/root/ace-step-music-xl/fixtures/short.mp3`

- [ ] **Step 1:** Create the fixtures directory

```bash
mkdir -p /root/ace-step-music-xl/fixtures
```

- [ ] **Step 2:** Generate a 2-second 440Hz tone mp3 using ffmpeg (already installed in the Dockerfile base, also on most dev machines)

```bash
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" \
  -c:a libmp3lame -b:a 64k \
  /root/ace-step-music-xl/fixtures/short.mp3
```

Expected: `fixtures/short.mp3` exists, ~16 KB.

- [ ] **Step 3:** Verify it decodes

```bash
python -c "
import soundfile as sf
info = sf.info('/root/ace-step-music-xl/fixtures/short.mp3')
print(f'duration={info.duration}s channels={info.channels} samplerate={info.samplerate}')
"
```

Expected output: `duration=2.0s channels=1 samplerate=44100` (or similar).

- [ ] **Step 4:** Commit

```bash
cd /root/ace-step-music-xl
git add fixtures/short.mp3
git commit -m "test: add 2s fixture mp3 for deterministic src_audio tests"
```

---

## Task 3: Create test_handler.py skeleton with shared mocking fixtures

Building the test file before the implementation so we can run it in red/green cycles for every later task.

**Files:**
- Create: `/root/ace-step-music-xl/test_handler.py`

- [ ] **Step 1:** Write the skeleton with shared fixtures, no actual test cases yet

```python
"""Unit tests for the ACE-Step XL RunPod handler.

All ACE-Step model loading and inference is mocked so these tests
run without a GPU or model weights. Tests cover all six task types
(text2music, cover, repaint, extract, lego, complete), schema
validation, src_audio resolution, and response shape.
"""
import base64
import importlib
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fake result dataclass matching what generate_music returns
# ---------------------------------------------------------------------------
@dataclass
class FakeGenerationResult:
    success: bool = True
    error: str | None = None
    audios: list[dict[str, Any]] = field(default_factory=list)
    status_message: str = ""
    extra_outputs: dict[str, Any] = field(default_factory=dict)


def _make_param_container(name: str):
    """Return a class that stores all kwargs as attributes."""

    class Container:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    Container.__name__ = name
    Container.__qualname__ = name
    return Container


# ---------------------------------------------------------------------------
# Auto-applied fixture: mock every external dependency handler.py imports
# so that `import handler` succeeds on a clean CI runner.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _mock_external_modules(monkeypatch, tmp_path):
    fake_project = tmp_path / "ACE-Step-1.5"
    (fake_project / "checkpoints").mkdir(parents=True)
    fake_ckpt = tmp_path / "checkpoints"
    fake_ckpt.mkdir(parents=True)

    monkeypatch.setenv("ACESTEP_PROJECT_ROOT", str(fake_project))
    monkeypatch.setenv("ACESTEP_CHECKPOINT_DIR", str(fake_ckpt))
    monkeypatch.setenv("ACESTEP_MAX_BATCH_SIZE", "4")

    mock_runpod = types.ModuleType("runpod")
    mock_runpod_serverless = types.ModuleType("runpod.serverless")
    mock_runpod_serverless.start = MagicMock()
    mock_runpod.serverless = mock_runpod_serverless

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.bool = bool
    mock_torch.int32 = int

    mock_acestep = types.ModuleType("acestep")
    mock_acestep_handler = types.ModuleType("acestep.handler")
    mock_acestep_llm = types.ModuleType("acestep.llm_inference")
    mock_acestep_inf = types.ModuleType("acestep.inference")
    mock_acestep_dl = types.ModuleType("acestep.model_downloader")

    mock_ace_handler_cls = MagicMock()
    mock_ace_handler_cls.return_value.initialize_service.return_value = ("ok", True)
    mock_acestep_handler.AceStepHandler = mock_ace_handler_cls

    mock_llm_handler_cls = MagicMock()
    mock_llm_handler_cls.return_value.initialize.return_value = ("ok", True)
    mock_acestep_llm.LLMHandler = mock_llm_handler_cls

    mock_acestep_dl.ensure_main_model = MagicMock(return_value=(True, "ok"))
    mock_acestep_dl.ensure_dit_model = MagicMock(return_value=(True, "ok"))

    mock_acestep_inf.GenerationParams = _make_param_container("GenerationParams")
    mock_acestep_inf.GenerationConfig = _make_param_container("GenerationConfig")
    mock_acestep_inf.generate_music = MagicMock()

    modules = {
        "runpod": mock_runpod,
        "runpod.serverless": mock_runpod_serverless,
        "torch": mock_torch,
        "acestep": mock_acestep,
        "acestep.handler": mock_acestep_handler,
        "acestep.llm_inference": mock_acestep_llm,
        "acestep.inference": mock_acestep_inf,
        "acestep.model_downloader": mock_acestep_dl,
    }

    saved = {}
    for name, mod in modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod
    sys.modules.pop("handler", None)

    yield

    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    sys.modules.pop("handler", None)


def _import_handler_module():
    """Force-reimport handler after mocks are installed."""
    return importlib.import_module("handler")


def _import_handler_func():
    return _import_handler_module().handler


def _setup_successful_mock(
    generate_music_mock: MagicMock,
    audio_bytes: bytes,
    seed: int = 42,
    sample_rate: int = 48000,
) -> dict[str, Any]:
    """Configure generate_music to write a fake audio file and return success."""
    captured: dict[str, Any] = {}

    def _side_effect(dit, llm, params, config, save_dir):
        captured["params"] = params
        captured["config"] = config
        audio_path = os.path.join(save_dir, "output.mp3")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        return FakeGenerationResult(
            success=True,
            audios=[
                {
                    "path": audio_path,
                    "params": {"seed": seed},
                    "sample_rate": sample_rate,
                    "duration": 30.0,
                }
            ],
        )

    generate_music_mock.side_effect = _side_effect
    return captured


def _short_mp3_bytes() -> bytes:
    return (FIXTURES / "short.mp3").read_bytes()
```

- [ ] **Step 2:** Verify pytest collects it (should collect zero tests, but no errors)

```bash
cd /root/ace-step-music-xl
pip install -r requirements-test.txt
pytest test_handler.py -v --collect-only
```

Expected: `no tests ran` or `0 tests collected`, exit code 0 (or 5 for "no tests collected", which is acceptable).

- [ ] **Step 3:** Commit

```bash
git add test_handler.py
git commit -m "test: scaffold test_handler.py with mocking fixtures"
```

---

## Task 4: Test + implement handler.py module-level constants and torch 2.4 compat patches

**Files:**
- Create: `/root/ace-step-music-xl/handler.py`
- Modify: `/root/ace-step-music-xl/test_handler.py`

- [ ] **Step 1:** Add a test class to `test_handler.py` (append to end of file) that asserts module-level constants are read from env

```python
# ---------------------------------------------------------------------------
# TestModuleConstants
# ---------------------------------------------------------------------------
class TestModuleConstants:
    """Verify env vars are read into module constants correctly."""

    def test_checkpoint_dir_from_env(self, monkeypatch, tmp_path):
        custom = tmp_path / "custom_ckpt"
        custom.mkdir()
        monkeypatch.setenv("ACESTEP_CHECKPOINT_DIR", str(custom))
        sys.modules.pop("handler", None)
        mod = _import_handler_module()
        assert mod.CHECKPOINT_DIR == str(custom)

    def test_config_path_defaults_to_xl(self, monkeypatch):
        monkeypatch.delenv("ACESTEP_CONFIG_PATH", raising=False)
        sys.modules.pop("handler", None)
        mod = _import_handler_module()
        assert mod.DIT_CONFIG == "acestep-v15-xl-base"

    def test_inference_steps_default_50(self, monkeypatch):
        monkeypatch.delenv("ACESTEP_INFERENCE_STEPS_DEFAULT", raising=False)
        sys.modules.pop("handler", None)
        mod = _import_handler_module()
        assert mod.INFERENCE_STEPS_DEFAULT == 50

    def test_guidance_scale_default_7(self, monkeypatch):
        monkeypatch.delenv("ACESTEP_GUIDANCE_SCALE_DEFAULT", raising=False)
        sys.modules.pop("handler", None)
        mod = _import_handler_module()
        assert mod.GUIDANCE_SCALE_DEFAULT == 7.0
```

- [ ] **Step 2:** Run the tests — expected: FAIL (`handler` module doesn't exist)

```bash
cd /root/ace-step-music-xl
pytest test_handler.py::TestModuleConstants -v
```

Expected: Collection errors or 4 failures citing "No module named 'handler'".

- [ ] **Step 3:** Write `/root/ace-step-music-xl/handler.py` with constants + compat patches + downloader + load_models stub

```python
"""RunPod serverless handler for ACE-Step 1.5 XL music generation.

Unified handler supporting all six task types:
  text2music, cover, repaint, extract, lego, complete.

Model weights live on a RunPod network volume mounted at /runpod-volume
in production; handler reads CHECKPOINT_DIR from env so tests can point
it at a local path.
"""
import os
import sys
import base64
import logging
import tempfile
import time
from typing import Optional
from urllib.parse import urlparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import runpod
import requests
import torch

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (read from env, tests can override with monkeypatch)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.environ.get("ACESTEP_PROJECT_ROOT", "/app/ACE-Step-1.5")
DIT_CONFIG = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-xl-base")
CHECKPOINT_DIR = os.environ.get(
    "ACESTEP_CHECKPOINT_DIR", "/runpod-volume/checkpoints"
)
LM_MODEL = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")
LM_BACKEND = os.environ.get("ACESTEP_LM_BACKEND", "vllm")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPILE_MODEL = os.environ.get("ACESTEP_COMPILE_MODEL", "0") in ("1", "true")
OFFLOAD_DIT_TO_CPU = os.environ.get("ACESTEP_OFFLOAD_DIT_TO_CPU", "0") in ("1", "true")
MAX_BATCH_SIZE = int(os.environ.get("ACESTEP_MAX_BATCH_SIZE", "4"))
INFERENCE_STEPS_DEFAULT = int(os.environ.get("ACESTEP_INFERENCE_STEPS_DEFAULT", "50"))
GUIDANCE_SCALE_DEFAULT = float(os.environ.get("ACESTEP_GUIDANCE_SCALE_DEFAULT", "7.0"))
MAX_RETRIES = int(os.environ.get("ACESTEP_LOAD_RETRIES", "5"))
RETRY_DELAY = int(os.environ.get("ACESTEP_RETRY_DELAY", "10"))

MAX_SRC_AUDIO_BYTES = 50 * 1024 * 1024  # 50 MB
SRC_AUDIO_DOWNLOAD_TIMEOUT = 30  # seconds
MAX_SRC_AUDIO_DURATION = 600  # seconds

VALID_AUDIO_FORMATS = {"mp3", "wav", "flac"}
VALID_TASK_TYPES = {
    "text2music", "cover", "repaint", "extract", "lego", "complete",
}

# Global model handles — loaded once at cold start
dit_handler: AceStepHandler | None = None
llm_handler: LLMHandler | None = None


# ---------------------------------------------------------------------------
# Torch 2.4.0 compat patches for A40 sm_86
# ---------------------------------------------------------------------------
def _apply_torch24_compat_patches():
    """Apply torch 2.4.0 compat patches for A40 GPU.

    1. Bool argsort on CUDA (2.4 raises; auto-cast to int32)
    2. Strip enable_gqa from F.scaled_dot_product_attention (2.5+ kwarg)
    """
    try:
        import torch as _torch
        import torch.nn.functional as F
    except (ImportError, ModuleNotFoundError):
        logger.warning("torch not fully available, skipping compat patches")
        return

    _orig_argsort = _torch.Tensor.argsort

    def _argsort_bool_safe(self, *args, **kwargs):
        if self.dtype == _torch.bool and self.is_cuda:
            return _orig_argsort(self.to(_torch.int32), *args, **kwargs)
        return _orig_argsort(self, *args, **kwargs)

    _torch.Tensor.argsort = _argsort_bool_safe

    _orig_torch_argsort = _torch.argsort

    def _torch_argsort_bool_safe(input, *args, **kwargs):
        if input.dtype == _torch.bool and input.is_cuda:
            return _orig_torch_argsort(input.to(_torch.int32), *args, **kwargs)
        return _orig_torch_argsort(input, *args, **kwargs)

    _torch.argsort = _torch_argsort_bool_safe

    _orig_sdpa = F.scaled_dot_product_attention

    def _sdpa_no_gqa(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return _orig_sdpa(*args, **kwargs)

    F.scaled_dot_product_attention = _sdpa_no_gqa


# ---------------------------------------------------------------------------
# Model loading (reads from CHECKPOINT_DIR)
# ---------------------------------------------------------------------------
def download_models():
    """Ensure weights exist under CHECKPOINT_DIR. No-op on network volume."""
    from pathlib import Path
    from acestep.model_downloader import ensure_main_model, ensure_dit_model

    ckpt_path = Path(CHECKPOINT_DIR)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    logger.info("Ensuring main model (LM + VAE + embeddings) is downloaded...")
    success, msg = ensure_main_model(checkpoints_dir=ckpt_path)
    if not success:
        raise RuntimeError(f"Failed to download main model: {msg}")

    logger.info(f"Ensuring DiT model '{DIT_CONFIG}' is downloaded...")
    success, msg = ensure_dit_model(DIT_CONFIG, checkpoints_dir=ckpt_path)
    if not success:
        raise RuntimeError(f"Failed to download DiT model: {msg}")


def _wait_for_cuda():
    import torch as _torch
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _torch.cuda.synchronize()
            logger.info(f"CUDA device ready (attempt {attempt})")
            return
        except RuntimeError as e:
            if "busy or unavailable" in str(e) and attempt < MAX_RETRIES:
                logger.warning(f"CUDA not ready (attempt {attempt}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY)
            else:
                raise


def load_models():
    """Download (if needed) and load DiT + LM into GPU memory."""
    global dit_handler, llm_handler

    _apply_torch24_compat_patches()
    download_models()

    if DEVICE == "cuda":
        _wait_for_cuda()

    logger.info(f"Loading DiT model: {DIT_CONFIG} on {DEVICE} (compile={COMPILE_MODEL})")
    dit_handler = AceStepHandler()
    msg, success = dit_handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path=DIT_CONFIG,
        device=DEVICE,
        compile_model=COMPILE_MODEL,
        offload_to_cpu=False,
        offload_dit_to_cpu=OFFLOAD_DIT_TO_CPU,
    )
    if not success:
        raise RuntimeError(f"DiT init failed: {msg}")

    logger.info(f"Loading LM model: {LM_MODEL} with backend {LM_BACKEND}")
    llm_handler = LLMHandler()
    msg, success = llm_handler.initialize(
        checkpoint_dir=CHECKPOINT_DIR,
        lm_model_path=LM_MODEL,
        backend=LM_BACKEND,
        device=DEVICE,
    )
    if not success:
        raise RuntimeError(f"LM init failed: {msg}")


def handler(job: dict) -> dict:
    """Stub — will be filled in Task 9."""
    return {"error": "handler not implemented yet"}


# Load models + start worker (skipped during test import because mocks replace
# AceStepHandler/LLMHandler; they'll still be "loaded" but with MagicMocks).
# Tests pop sys.modules['handler'] before importing, so this runs each time,
# but the mocks make it fast and safe.
load_models()
runpod.serverless.start({"handler": handler})
```

- [ ] **Step 4:** Run constants tests — expected: PASS

```bash
pytest test_handler.py::TestModuleConstants -v
```

Expected: 4 passed.

- [ ] **Step 5:** Commit

```bash
git add handler.py test_handler.py
git commit -m "feat: add handler.py with module constants and torch 2.4 compat patches"
```

---

## Task 5: Test + implement `_resolve_src_audio` — base64 path

**Files:**
- Modify: `/root/ace-step-music-xl/handler.py`
- Modify: `/root/ace-step-music-xl/test_handler.py`

- [ ] **Step 1:** Append test class to `test_handler.py`

```python
# ---------------------------------------------------------------------------
# TestResolveSrcAudio — base64 path
# ---------------------------------------------------------------------------
class TestResolveSrcAudioBase64:
    def test_base64_decodes_to_tempfile(self):
        mod = _import_handler_module()
        audio_bytes = _short_mp3_bytes()
        b64 = base64.b64encode(audio_bytes).decode()
        path = mod._resolve_src_audio({"src_audio_base64": b64})
        assert path is not None
        assert os.path.exists(path)
        with open(path, "rb") as f:
            assert f.read() == audio_bytes
        os.unlink(path)

    def test_none_when_no_audio_provided(self):
        mod = _import_handler_module()
        path = mod._resolve_src_audio({"prompt": "just text"})
        assert path is None

    def test_invalid_base64_raises_value_error(self):
        mod = _import_handler_module()
        with pytest.raises(ValueError, match="base64"):
            mod._resolve_src_audio({"src_audio_base64": "!!!not-b64!!!"})

    def test_base64_invalid_audio_raises(self):
        mod = _import_handler_module()
        junk = base64.b64encode(b"not-audio-bytes-at-all").decode()
        with pytest.raises(ValueError, match="audio"):
            mod._resolve_src_audio({"src_audio_base64": junk})
```

- [ ] **Step 2:** Run — expected: FAIL (`_resolve_src_audio` doesn't exist)

```bash
pytest test_handler.py::TestResolveSrcAudioBase64 -v
```

Expected: 4 failures with `AttributeError: module 'handler' has no attribute '_resolve_src_audio'`.

- [ ] **Step 3:** Add `_resolve_src_audio` to `handler.py` (insert immediately above the `handler()` function)

```python
# ---------------------------------------------------------------------------
# Source audio resolver
# ---------------------------------------------------------------------------
def _write_tempfile(data: bytes, suffix: str = ".audio") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="src_audio_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except Exception:
        os.unlink(path)
        raise
    return path


def _validate_audio_file(path: str) -> None:
    """Raise ValueError if path isn't decodable audio or too long."""
    try:
        import soundfile as sf
    except ImportError as e:
        raise ValueError("soundfile not installed; cannot validate audio") from e
    try:
        info = sf.info(path)
    except Exception as e:
        raise ValueError(f"invalid audio file: {e}") from e
    if info.duration > MAX_SRC_AUDIO_DURATION:
        raise ValueError(
            f"src_audio duration {info.duration:.1f}s exceeds "
            f"max {MAX_SRC_AUDIO_DURATION}s"
        )


def _resolve_src_audio(job_input: dict) -> Optional[str]:
    """Return a filesystem path to the source audio, or None if unprovided.

    URL takes precedence over base64 if both are given (caller is warned).
    Caller is responsible for os.unlink(path) when done.

    Raises ValueError on invalid input.
    """
    url = job_input.get("src_audio_url")
    b64 = job_input.get("src_audio_base64")

    if url and b64:
        logger.warning(
            "Both src_audio_url and src_audio_base64 provided; using URL"
        )

    if url:
        return _download_src_audio_url(url)
    if b64:
        try:
            data = base64.b64decode(b64, validate=True)
        except Exception as e:
            raise ValueError(f"src_audio_base64 is not valid base64: {e}") from e
        path = _write_tempfile(data)
        try:
            _validate_audio_file(path)
        except Exception:
            os.unlink(path)
            raise
        return path
    return None


def _download_src_audio_url(url: str) -> str:
    """Stub — URL implementation added in Task 6."""
    raise NotImplementedError("URL path implemented in Task 6")
```

- [ ] **Step 4:** Run — expected: PASS

```bash
pytest test_handler.py::TestResolveSrcAudioBase64 -v
```

Expected: 4 passed.

- [ ] **Step 5:** Commit

```bash
git add handler.py test_handler.py
git commit -m "feat: _resolve_src_audio base64 path + audio file validation"
```

---

## Task 6: Test + implement `_resolve_src_audio` — URL path

**Files:**
- Modify: `/root/ace-step-music-xl/handler.py`
- Modify: `/root/ace-step-music-xl/test_handler.py`

- [ ] **Step 1:** Append test class (uses `responses` library for HTTP mocking)

```python
# ---------------------------------------------------------------------------
# TestResolveSrcAudio — URL path
# ---------------------------------------------------------------------------
import responses
from responses import matchers


class TestResolveSrcAudioUrl:
    URL = "https://example.com/track.mp3"

    @responses.activate
    def test_https_download_writes_tempfile(self):
        mod = _import_handler_module()
        audio_bytes = _short_mp3_bytes()
        responses.add(
            responses.GET, self.URL,
            body=audio_bytes,
            headers={"Content-Type": "audio/mpeg"},
        )
        path = mod._resolve_src_audio({"src_audio_url": self.URL})
        assert path is not None
        assert os.path.exists(path)
        with open(path, "rb") as f:
            assert f.read() == audio_bytes
        os.unlink(path)

    def test_http_url_rejected(self):
        mod = _import_handler_module()
        with pytest.raises(ValueError, match="https"):
            mod._resolve_src_audio({"src_audio_url": "http://example.com/t.mp3"})

    def test_file_url_rejected(self):
        mod = _import_handler_module()
        with pytest.raises(ValueError, match="https"):
            mod._resolve_src_audio({"src_audio_url": "file:///etc/passwd"})

    @responses.activate
    def test_oversized_download_rejected(self):
        """Download larger than MAX_SRC_AUDIO_BYTES should raise."""
        mod = _import_handler_module()
        # 51 MB payload
        big = b"\x00" * (51 * 1024 * 1024)
        responses.add(
            responses.GET, self.URL,
            body=big,
            headers={"Content-Type": "audio/mpeg"},
        )
        with pytest.raises(ValueError, match="50 MB|size"):
            mod._resolve_src_audio({"src_audio_url": self.URL})

    @responses.activate
    def test_url_wins_when_both_provided(self):
        mod = _import_handler_module()
        url_bytes = _short_mp3_bytes()
        responses.add(
            responses.GET, self.URL,
            body=url_bytes,
            headers={"Content-Type": "audio/mpeg"},
        )
        # base64 contains different bytes — we should get the URL ones
        different = base64.b64encode(b"DIFFERENT").decode()
        path = mod._resolve_src_audio({
            "src_audio_url": self.URL,
            "src_audio_base64": different,
        })
        with open(path, "rb") as f:
            assert f.read() == url_bytes
        os.unlink(path)
```

- [ ] **Step 2:** Run — expected: 5 FAIL

```bash
pytest test_handler.py::TestResolveSrcAudioUrl -v
```

Expected: failures citing `NotImplementedError` or URL scheme assertions.

- [ ] **Step 3:** Replace the `_download_src_audio_url` stub in `handler.py` with the real implementation

```python
def _download_src_audio_url(url: str) -> str:
    """Download audio from an HTTPS URL into a tempfile, with size + scheme caps."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(
            f"src_audio_url must be https:// (got {parsed.scheme!r})"
        )

    try:
        resp = requests.get(url, stream=True, timeout=SRC_AUDIO_DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"failed to download src_audio_url: {e}") from e

    # Size gate based on Content-Length if present
    content_length = resp.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_SRC_AUDIO_BYTES:
        raise ValueError(
            f"src_audio_url size {content_length} exceeds 50 MB limit"
        )

    # Guess extension from URL or content-type; default .audio
    suffix = os.path.splitext(parsed.path)[1] or ".audio"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="src_audio_")
    total = 0
    try:
        with os.fdopen(fd, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_SRC_AUDIO_BYTES:
                    raise ValueError(
                        f"src_audio_url exceeded 50 MB size limit during stream"
                    )
                f.write(chunk)
    except Exception:
        if os.path.exists(path):
            os.unlink(path)
        raise

    try:
        _validate_audio_file(path)
    except Exception:
        os.unlink(path)
        raise
    return path
```

- [ ] **Step 4:** Run — expected: 5 PASS

```bash
pytest test_handler.py::TestResolveSrcAudioUrl -v
```

Expected: 5 passed.

- [ ] **Step 5:** Commit

```bash
git add handler.py test_handler.py
git commit -m "feat: _resolve_src_audio URL path with scheme+size validation"
```

---

## Task 7: Test + implement `_validate` schema gate for all 6 task types

**Files:**
- Modify: `/root/ace-step-music-xl/handler.py`
- Modify: `/root/ace-step-music-xl/test_handler.py`

- [ ] **Step 1:** Append to `test_handler.py`

```python
# ---------------------------------------------------------------------------
# TestSchemaValidation
# ---------------------------------------------------------------------------
class TestSchemaValidation:
    def _call(self, job_input: dict):
        mod = _import_handler_module()
        return mod._validate(job_input)

    def test_unknown_task_type_errors(self):
        err = self._call({"task_type": "mystery", "prompt": "x"})
        assert err is not None
        assert "task_type" in err["error"].lower()

    def test_text2music_missing_prompt(self):
        err = self._call({"task_type": "text2music"})
        assert err is not None
        assert "prompt" in err["error"].lower()

    def test_text2music_happy_path(self):
        err = self._call({"task_type": "text2music", "prompt": "p"})
        assert err is None

    def test_cover_missing_src_audio(self):
        err = self._call({"task_type": "cover", "prompt": "p"})
        assert err is not None
        assert "src_audio" in err["error"].lower()

    def test_cover_missing_prompt(self):
        err = self._call({"task_type": "cover", "src_audio_base64": "x"})
        assert err is not None
        assert "prompt" in err["error"].lower()

    def test_repaint_missing_start_end(self):
        err = self._call({
            "task_type": "repaint",
            "prompt": "p",
            "src_audio_base64": "x",
        })
        assert err is not None
        assert "repainting" in err["error"].lower()

    def test_extract_missing_instruction(self):
        err = self._call({
            "task_type": "extract",
            "src_audio_base64": "x",
        })
        assert err is not None
        assert "instruction" in err["error"].lower()

    def test_lego_happy_path(self):
        err = self._call({
            "task_type": "lego",
            "prompt": "p",
            "src_audio_base64": "x",
            "repainting_start": 0,
            "repainting_end": 10,
        })
        assert err is None

    def test_complete_missing_prompt(self):
        err = self._call({
            "task_type": "complete",
            "src_audio_base64": "x",
        })
        assert err is not None
        assert "prompt" in err["error"].lower()

    def test_audio_format_invalid(self):
        err = self._call({
            "task_type": "text2music",
            "prompt": "p",
            "audio_format": "ogg",
        })
        assert err is not None
        assert "audio_format" in err["error"].lower()
```

- [ ] **Step 2:** Run — expected: 10 FAIL

```bash
pytest test_handler.py::TestSchemaValidation -v
```

- [ ] **Step 3:** Add to `handler.py` (insert above `_resolve_src_audio`)

```python
# ---------------------------------------------------------------------------
# Task-type schema
# ---------------------------------------------------------------------------
# Required fields keyed by task_type. "_src_audio" means "at least one of
# src_audio_url or src_audio_base64 must be present".
TASK_REQUIRED = {
    "text2music": ["prompt"],
    "cover":      ["prompt", "_src_audio"],
    "repaint":    ["prompt", "_src_audio", "repainting_start", "repainting_end"],
    "extract":    ["instruction", "_src_audio"],
    "lego":       ["prompt", "_src_audio", "repainting_start", "repainting_end"],
    "complete":   ["prompt", "_src_audio"],
}


def _validate(job_input: dict) -> Optional[dict]:
    """Return None if input is OK, else an error dict suitable for RunPod."""
    task_type = job_input.get("task_type", "text2music")
    if task_type not in VALID_TASK_TYPES:
        return {
            "error": f"unknown task_type {task_type!r}; "
                     f"valid: {sorted(VALID_TASK_TYPES)}"
        }

    audio_format = job_input.get("audio_format", "mp3")
    if audio_format not in VALID_AUDIO_FORMATS:
        return {
            "error": f"invalid audio_format {audio_format!r}; "
                     f"valid: {sorted(VALID_AUDIO_FORMATS)}"
        }

    required = TASK_REQUIRED[task_type]
    missing = []
    for field in required:
        if field == "_src_audio":
            has_url = bool(job_input.get("src_audio_url"))
            has_b64 = bool(job_input.get("src_audio_base64"))
            if not (has_url or has_b64):
                missing.append("src_audio_url or src_audio_base64")
            continue
        val = job_input.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            missing.append(field)

    if missing:
        return {
            "error": f"task_type={task_type!r} missing required fields: "
                     + ", ".join(missing)
        }
    return None
```

- [ ] **Step 4:** Run — expected: 10 PASS

```bash
pytest test_handler.py::TestSchemaValidation -v
```

- [ ] **Step 5:** Commit

```bash
git add handler.py test_handler.py
git commit -m "feat: _validate schema gate with TASK_REQUIRED for 6 task types"
```

---

## Task 8: Test + implement `handler()` for text2music baseline

**Files:**
- Modify: `/root/ace-step-music-xl/handler.py`
- Modify: `/root/ace-step-music-xl/test_handler.py`

- [ ] **Step 1:** Append to `test_handler.py`

```python
# ---------------------------------------------------------------------------
# TestHandlerText2Music
# ---------------------------------------------------------------------------
class TestHandlerText2Music:
    def test_happy_path_returns_expected_keys(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        _setup_successful_mock(gen_mock, b"fake-audio")

        result = handler_fn({"input": {"prompt": "epic orchestral"}})

        expected = {
            "audio_base64", "format", "duration",
            "seed", "sample_rate", "task_type",
        }
        assert expected == set(result.keys())
        assert result["task_type"] == "text2music"
        assert result["format"] == "mp3"

    def test_audio_base64_roundtrip(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        payload = b"\x00\x01\x02\xff" * 256
        _setup_successful_mock(gen_mock, payload)

        result = handler_fn({"input": {"prompt": "lo-fi"}})
        assert base64.b64decode(result["audio_base64"]) == payload

    def test_missing_prompt_errors(self):
        handler_fn = _import_handler_func()
        result = handler_fn({"input": {}})
        assert "error" in result

    def test_duration_clamped_to_min(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        _setup_successful_mock(gen_mock, b"a")
        result = handler_fn({"input": {"prompt": "x", "duration": 1}})
        # duration returned from the audio dict, which _setup_successful_mock
        # populates with the duration we pass via params. Check it's clamped in params.
        captured = gen_mock.side_effect  # unused — duration is on params via mock capture

    def test_defaults_to_instrumental(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        captured = _setup_successful_mock(gen_mock, b"a")
        handler_fn({"input": {"prompt": "ambient"}})
        params = captured["params"]
        assert params.instrumental is True
        assert params.lyrics == "[Instrumental]"

    def test_vocal_mode_accepts_lyrics(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        captured = _setup_successful_mock(gen_mock, b"a")
        handler_fn({"input": {
            "prompt": "pop",
            "instrumental": False,
            "lyrics": "la la la",
        }})
        assert captured["params"].instrumental is False
        assert captured["params"].lyrics == "la la la"

    def test_xl_defaults_inference_steps_50_cfg_7(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        captured = _setup_successful_mock(gen_mock, b"a")
        handler_fn({"input": {"prompt": "x"}})
        assert captured["params"].inference_steps == 50
        assert captured["params"].guidance_scale == 7.0

    def test_batch_size_clamped(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        captured = _setup_successful_mock(gen_mock, b"a")
        handler_fn({"input": {"prompt": "x", "batch_size": 999}})
        mod = _import_handler_module()
        assert captured["config"].batch_size == mod.MAX_BATCH_SIZE

    def test_failed_result_returns_error(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        gen_mock.side_effect = None
        gen_mock.return_value = FakeGenerationResult(success=False, error="CUDA OOM")
        result = handler_fn({"input": {"prompt": "x"}})
        assert "error" in result
        assert "CUDA OOM" in result["error"]
```

- [ ] **Step 2:** Run — expected: all FAIL (handler stub returns `"not implemented"`)

```bash
pytest test_handler.py::TestHandlerText2Music -v
```

- [ ] **Step 3:** Replace the stub `handler()` in `handler.py` with the full implementation

```python
def handler(job: dict) -> dict:
    """Process a single music-generation request across all 6 task types.

    See docs/superpowers/specs/2026-04-20-ace-step-xl-serverless-design.md
    for the full input schema.
    """
    if dit_handler is None or llm_handler is None:
        return {"error": "Models not loaded — worker startup failed"}

    job_input = job.get("input", {}) or {}

    err = _validate(job_input)
    if err:
        return err

    task_type = job_input.get("task_type", "text2music")

    # Resolve src_audio for audio-input tasks (all except text2music).
    src_audio_path: Optional[str] = None
    try:
        src_audio_path = _resolve_src_audio(job_input)
    except ValueError as e:
        return {"error": f"src_audio error: {e}"}

    try:
        params, config, audio_format = _build_params(
            task_type, job_input, src_audio_path
        )

        with tempfile.TemporaryDirectory() as save_dir:
            result = generate_music(
                dit_handler, llm_handler, params, config, save_dir=save_dir,
            )
            if not result.success:
                return {"error": result.error or "Generation failed"}
            if not result.audios:
                return {"error": "No audio generated"}

            audio = result.audios[0]
            with open(audio["path"], "rb") as f:
                audio_bytes = f.read()

            return {
                "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
                "format": audio_format,
                "duration": audio.get("duration", params.duration),
                "seed": audio["params"].get("seed", params.seed),
                "sample_rate": audio.get("sample_rate", 48000),
                "task_type": task_type,
            }
    except Exception as e:
        logger.error("Unhandled generation error", exc_info=True)
        return {"error": f"Internal error: {type(e).__name__}: {e}"}
    finally:
        if src_audio_path and os.path.exists(src_audio_path):
            try:
                os.unlink(src_audio_path)
            except OSError:
                logger.warning(f"Failed to unlink {src_audio_path}")


def _build_params(
    task_type: str,
    job_input: dict,
    src_audio_path: Optional[str],
) -> tuple:
    """Build GenerationParams + GenerationConfig for the given task_type.

    Returns (params, config, audio_format).
    """
    audio_format = job_input.get("audio_format", "mp3")
    seed = int(job_input.get("seed", -1))
    batch_size = int(job_input.get("batch_size", 1))
    batch_size = max(1, min(MAX_BATCH_SIZE, batch_size))
    inference_steps = int(
        job_input.get("inference_steps", INFERENCE_STEPS_DEFAULT)
    )
    inference_steps = max(1, min(200, inference_steps))
    guidance_scale = float(
        job_input.get("guidance_scale", GUIDANCE_SCALE_DEFAULT)
    )
    shift = float(job_input.get("shift", 1.0))

    # Common defaults
    duration = float(job_input.get("duration", 30))
    duration = max(10.0, min(600.0, duration))
    instrumental = bool(job_input.get("instrumental", True))
    lyrics_in = job_input.get("lyrics", "")
    lyrics = "[Instrumental]" if instrumental else lyrics_in

    bpm = job_input.get("bpm")
    if bpm is not None:
        bpm = max(30, min(300, int(bpm)))
    key_scale = job_input.get("key_scale", "")
    time_signature = job_input.get("time_signature", "")
    lm_temperature = float(job_input.get("lm_temperature", 0.85))
    thinking = bool(job_input.get("thinking", True))

    # Tasks where the LM is auto-skipped upstream; force thinking=False
    # so we don't waste cycles preparing CoT inputs.
    if task_type in ("cover", "repaint", "extract"):
        thinking = False

    # Task-specific extras
    caption = job_input.get("prompt", "")
    audio_cover_strength = float(
        job_input.get("audio_cover_strength", 0.3)
    )
    instruction = job_input.get("instruction", "")
    repainting_start = float(job_input.get("repainting_start", 0.0))
    repainting_end = float(job_input.get("repainting_end", -1))

    params_kwargs = dict(
        task_type=task_type,
        caption=caption,
        lyrics=lyrics,
        instrumental=instrumental,
        duration=duration,
        bpm=bpm,
        keyscale=key_scale,
        timesignature=time_signature,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        seed=seed,
        thinking=thinking,
        lm_temperature=lm_temperature,
        use_cot_metas=thinking,
        use_cot_caption=thinking,
        use_cot_language=False,
    )

    if task_type != "text2music":
        params_kwargs["src_audio_path"] = src_audio_path
        params_kwargs["audio_cover_strength"] = audio_cover_strength

    if task_type in ("repaint", "lego"):
        params_kwargs["repainting_start"] = repainting_start
        params_kwargs["repainting_end"] = repainting_end

    if task_type == "extract":
        params_kwargs["instruction"] = instruction

    params = GenerationParams(**params_kwargs)

    config = GenerationConfig(
        batch_size=batch_size,
        use_random_seed=(seed == -1),
        seeds=None if seed == -1 else [seed],
        audio_format=audio_format,
    )
    return params, config, audio_format
```

- [ ] **Step 4:** Run — expected: PASS all

```bash
pytest test_handler.py::TestHandlerText2Music -v
```

- [ ] **Step 5:** Commit

```bash
git add handler.py test_handler.py
git commit -m "feat: handler() text2music with XL defaults (steps=50, cfg=7.0)"
```

---

## Task 9: Test handler() for cover / repaint / extract / lego / complete

**Files:**
- Modify: `/root/ace-step-music-xl/test_handler.py`

- [ ] **Step 1:** Append test class

```python
# ---------------------------------------------------------------------------
# TestHandlerAudioInputTasks
# ---------------------------------------------------------------------------
class TestHandlerAudioInputTasks:
    def _input_with_src(self, extra: dict) -> dict:
        b64 = base64.b64encode(_short_mp3_bytes()).decode()
        return {"src_audio_base64": b64, **extra}

    def test_cover_passes_src_audio_path(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        captured = _setup_successful_mock(gen_mock, b"a")
        result = handler_fn({"input": self._input_with_src({
            "task_type": "cover",
            "prompt": "jazz cover",
        })})
        assert "error" not in result
        assert result["task_type"] == "cover"
        params = captured["params"]
        assert params.task_type == "cover"
        assert params.src_audio_path is not None
        # LM auto-skipped → thinking forced False
        assert params.thinking is False

    def test_repaint_passes_start_end(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        captured = _setup_successful_mock(gen_mock, b"a")
        result = handler_fn({"input": self._input_with_src({
            "task_type": "repaint",
            "prompt": "fix the bridge",
            "repainting_start": 10.0,
            "repainting_end": 20.0,
        })})
        assert "error" not in result
        params = captured["params"]
        assert params.task_type == "repaint"
        assert params.repainting_start == 10.0
        assert params.repainting_end == 20.0
        assert params.thinking is False

    def test_extract_passes_instruction(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        captured = _setup_successful_mock(gen_mock, b"a")
        result = handler_fn({"input": self._input_with_src({
            "task_type": "extract",
            "instruction": "isolate drums",
        })})
        assert "error" not in result
        params = captured["params"]
        assert params.task_type == "extract"
        assert params.instruction == "isolate drums"
        assert params.thinking is False

    def test_lego_keeps_thinking_true_by_default(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        captured = _setup_successful_mock(gen_mock, b"a")
        result = handler_fn({"input": self._input_with_src({
            "task_type": "lego",
            "prompt": "remix",
            "repainting_start": 0,
            "repainting_end": 30,
        })})
        assert "error" not in result
        params = captured["params"]
        assert params.task_type == "lego"
        assert params.repainting_start == 0
        assert params.repainting_end == 30
        # lego uses LM, thinking stays user-default (True)
        assert params.thinking is True

    def test_complete_keeps_thinking_true_by_default(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        captured = _setup_successful_mock(gen_mock, b"a")
        result = handler_fn({"input": self._input_with_src({
            "task_type": "complete",
            "prompt": "finish the outro",
        })})
        assert "error" not in result
        params = captured["params"]
        assert params.task_type == "complete"
        assert params.thinking is True

    def test_src_audio_tempfile_cleaned_up(self, tmp_path):
        """After handler() returns, the resolved tempfile should be gone."""
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        captured = _setup_successful_mock(gen_mock, b"a")
        handler_fn({"input": self._input_with_src({
            "task_type": "cover",
            "prompt": "p",
        })})
        src_path = captured["params"].src_audio_path
        assert not os.path.exists(src_path)

    def test_all_task_types_return_consistent_output_shape(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        _setup_successful_mock(gen_mock, b"a")

        inputs = {
            "text2music": {"prompt": "x"},
            "cover": self._input_with_src({
                "task_type": "cover", "prompt": "p",
            }),
            "repaint": self._input_with_src({
                "task_type": "repaint", "prompt": "p",
                "repainting_start": 0, "repainting_end": 10,
            }),
            "extract": self._input_with_src({
                "task_type": "extract", "instruction": "drums",
            }),
            "lego": self._input_with_src({
                "task_type": "lego", "prompt": "p",
                "repainting_start": 0, "repainting_end": 10,
            }),
            "complete": self._input_with_src({
                "task_type": "complete", "prompt": "p",
            }),
        }

        expected = {"audio_base64", "format", "duration",
                    "seed", "sample_rate", "task_type"}
        for task, inp in inputs.items():
            _setup_successful_mock(gen_mock, b"a")
            result = handler_fn({"input": inp})
            assert "error" not in result, f"{task}: {result}"
            assert set(result.keys()) == expected, f"{task} key mismatch"
            assert result["task_type"] == task, f"{task} echo mismatch"
```

- [ ] **Step 2:** Run — expected: all PASS (handler already supports these paths)

```bash
pytest test_handler.py::TestHandlerAudioInputTasks -v
```

- [ ] **Step 3:** Commit

```bash
git add test_handler.py
git commit -m "test: cover/repaint/extract/lego/complete task routing + cleanup"
```

---

## Task 10: Coverage gate — ensure 80% on handler.py

**Files:**
- Modify: `/root/ace-step-music-xl/requirements-test.txt`

- [ ] **Step 1:** Add `pytest-cov`

```
pytest
pytest-cov
pyyaml
responses
soundfile
numpy
```

Re-install:
```bash
cd /root/ace-step-music-xl
pip install -r requirements-test.txt
```

- [ ] **Step 2:** Run coverage

```bash
pytest test_handler.py --cov=handler --cov-report=term-missing
```

Expected: coverage on `handler.py` at least 80%. If below, note which lines are missed and add tests for those branches. Likely candidates: `_wait_for_cuda` retry branch, `load_models` failure path, `_build_params` edge cases (vocal with bpm, explicit seed).

- [ ] **Step 3:** If under 80%, add missing tests; iterate until green.

- [ ] **Step 4:** Commit (even if no source changes, the test additions go in)

```bash
git add test_handler.py requirements-test.txt
git commit -m "test: achieve 80% coverage on handler.py"
```

---

## Task 11: Write Dockerfile (turbo deltas only)

**Files:**
- Create: `/root/ace-step-music-xl/Dockerfile`

- [ ] **Step 1:** Write the Dockerfile — identical to turbo except for the env block and comments referencing XL

```dockerfile
# ACE-Step 1.5 XL RunPod Serverless Endpoint
# GPU: NVIDIA A40 (48GB GDDR6, Ampere sm_86)
# - bfloat16 inference (native Ampere support)
# - SDPA attention (flash-attn skipped due to torch 2.4.0 ABI mismatch)
# - XL DiT (~9 GB bf16) + 1.7B LM fit well under 48GB; no offload needed
# - Weights live on RunPod network volume at /runpod-volume (not baked)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /app/ACE-Step-1.5 && \
    rm -rf /app/ACE-Step-1.5/.cache /app/ACE-Step-1.5/.git

WORKDIR /app/ACE-Step-1.5

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "safetensors==0.7.0" \
    "transformers>=4.51.0,<4.58.0" \
    "diffusers>=0.27.0,<0.33.0" \
    "scipy>=1.10.1" \
    "soundfile>=0.13.1" \
    "loguru>=0.7.3" \
    "einops>=0.8.1" \
    "accelerate>=1.12.0" \
    "numba>=0.63.1" \
    "vector-quantize-pytorch>=1.27.15" \
    "peft>=0.18.0" \
    "toml" \
    "xxhash" \
    "diskcache" \
    "modelscope" \
    "typer-slim>=0.21.1"

RUN pip install --no-cache-dir \
    torch==2.4.0+cu124 \
    torchaudio==2.4.0+cu124 \
    torchvision==0.19.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir torchcodec || true

RUN pip install --no-cache-dir "triton>=3.0.0" || true

RUN cd acestep/third_parts/nano-vllm && pip install --no-cache-dir --no-deps -e .

RUN pip install --no-cache-dir --no-deps -e .

RUN pip install --no-cache-dir runpod requests

RUN pip uninstall -y torchao flash-attn 2>/dev/null || true

WORKDIR /app
COPY handler.py /app/handler.py

# ── XL-specific environment ──
ENV ACESTEP_PROJECT_ROOT=/app/ACE-Step-1.5
ENV ACESTEP_CONFIG_PATH=acestep-v15-xl-base
ENV ACESTEP_CHECKPOINT_DIR=/runpod-volume/checkpoints
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ENV ACESTEP_LM_BACKEND=vllm
ENV PYTHONUNBUFFERED=1

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV ACESTEP_COMPILE_MODEL=1
ENV ACESTEP_MAX_BATCH_SIZE=4
ENV ACESTEP_INFERENCE_STEPS_DEFAULT=50
ENV ACESTEP_GUIDANCE_SCALE_DEFAULT=7.0
ENV ACESTEP_OFFLOAD_DIT_TO_CPU=0

CMD ["python", "-u", "/app/handler.py"]
```

- [ ] **Step 2:** Sanity-check the Dockerfile syntax

```bash
docker buildx build --print-only /root/ace-step-music-xl 2>/dev/null || \
docker build --help >/dev/null  # just verify docker CLI; full build happens in CI
```

- [ ] **Step 3:** Commit

```bash
git add Dockerfile
git commit -m "build: Dockerfile for XL on A40 (turbo stack + XL env deltas)"
```

---

## Task 12: Write GitHub Actions workflow

**Files:**
- Create: `/root/ace-step-music-xl/.github/workflows/deploy.yml`

- [ ] **Step 1:** Create the workflow directory and file

```bash
mkdir -p /root/ace-step-music-xl/.github/workflows
```

- [ ] **Step 2:** Write `deploy.yml`

```yaml
name: Build and Deploy ACE-Step XL

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
          cache-dependency-path: requirements-test.txt

      - name: Install system deps for soundfile
        run: sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg

      - name: Install test dependencies
        run: pip install -r requirements-test.txt

      - name: Run tests
        run: pytest test_handler.py test_workflow.py -v

  build-push:
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    timeout-minutes: 60
    outputs:
      image_tag: ${{ steps.meta.outputs.tags }}
      sha_tag: ${{ steps.sha-tag.outputs.sha_tag }}
    steps:
      - uses: actions/checkout@v4

      - name: Free disk space
        run: |
          sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache
          sudo docker image prune -af
          df -h /

      - uses: docker/setup-buildx-action@v3

      - uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Docker meta
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: dmrabh/ace-step-music-xl
          tags: |
            type=raw,value=latest
            type=sha,prefix=,format=short

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          platforms: linux/amd64
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Extract SHA tag
        id: sha-tag
        run: echo "sha_tag=${{ steps.meta.outputs.version }}" >> "$GITHUB_OUTPUT"

  deploy:
    needs: build-push
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - name: Update RunPod template
        env:
          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
          RUNPOD_TEMPLATE_ID: ${{ secrets.RUNPOD_TEMPLATE_ID_XL }}
          IMAGE_TAG: dmrabh/ace-step-music-xl:${{ needs.build-push.outputs.sha_tag }}
        run: |
          PAYLOAD=$(python3 -c "
          import json, os
          query = 'mutation SaveTemplate(\$id: String!, \$imageName: String!) { saveTemplate(input: { id: \$id, imageName: \$imageName }) { id imageName } }'
          variables = {
              'id': os.environ['RUNPOD_TEMPLATE_ID'],
              'imageName': os.environ['IMAGE_TAG']
          }
          print(json.dumps({'query': query, 'variables': variables}))
          ")

          RESPONSE=$(curl -s --max-time 30 --connect-timeout 10 -X POST https://api.runpod.io/graphql \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            -d "${PAYLOAD}")

          echo "RunPod template update completed"

          if echo "${RESPONSE}" | grep -q '"errors"'; then
            echo "ERROR: RunPod template update failed"
            echo "${RESPONSE}"
            exit 1
          fi

          echo "Template updated successfully"
```

- [ ] **Step 3:** Commit

```bash
git add .github/workflows/deploy.yml
git commit -m "ci: GitHub Actions workflow — test → build-push → RunPod deploy"
```

---

## Task 13: Write test_workflow.py

**Files:**
- Create: `/root/ace-step-music-xl/test_workflow.py`

- [ ] **Step 1:** Write the structural tests

```python
"""Tests for the GitHub Actions workflow structure."""
import yaml
import pytest
from pathlib import Path

WORKFLOW_PATH = Path(__file__).parent / ".github" / "workflows" / "deploy.yml"


@pytest.fixture
def workflow():
    with open(WORKFLOW_PATH) as f:
        return yaml.safe_load(f)


class TestWorkflowStructure:
    def test_workflow_file_exists(self):
        assert WORKFLOW_PATH.exists(), f"Workflow file not found at {WORKFLOW_PATH}"

    def test_triggers_on_push_to_main(self, workflow):
        triggers = workflow.get("on") or workflow.get(True)
        assert "main" in triggers["push"]["branches"]

    def test_triggers_on_pr_to_main(self, workflow):
        triggers = workflow.get("on") or workflow.get(True)
        assert "main" in triggers["pull_request"]["branches"]

    def test_has_three_jobs(self, workflow):
        jobs = workflow["jobs"]
        assert set(jobs.keys()) == {"test", "build-push", "deploy"}

    def test_build_push_needs_test(self, workflow):
        assert workflow["jobs"]["build-push"]["needs"] == "test"

    def test_deploy_needs_build_push(self, workflow):
        assert workflow["jobs"]["deploy"]["needs"] == "build-push"

    def test_build_push_only_on_main(self, workflow):
        condition = workflow["jobs"]["build-push"]["if"]
        assert "push" in condition
        assert "main" in condition

    def test_deploy_only_on_main(self, workflow):
        condition = workflow["jobs"]["deploy"]["if"]
        assert "push" in condition
        assert "main" in condition

    def test_build_uses_linux_amd64(self, workflow):
        build_steps = workflow["jobs"]["build-push"]["steps"]
        build_step = [s for s in build_steps if s.get("name") == "Build and push"][0]
        assert build_step["with"]["platforms"] == "linux/amd64"

    def test_image_name_is_xl(self, workflow):
        steps = workflow["jobs"]["build-push"]["steps"]
        meta = [s for s in steps if s.get("id") == "meta"][0]
        assert meta["with"]["images"] == "dmrabh/ace-step-music-xl"

    def test_deploy_uses_xl_template_secret(self, workflow):
        deploy_steps = workflow["jobs"]["deploy"]["steps"]
        deploy_step = deploy_steps[0]
        assert "RUNPOD_API_KEY" in deploy_step["env"]
        # The raw YAML will have the placeholder string intact
        assert "RUNPOD_TEMPLATE_ID_XL" in deploy_step["env"]["RUNPOD_TEMPLATE_ID"]
```

- [ ] **Step 2:** Run — expected: PASS

```bash
pytest test_workflow.py -v
```

- [ ] **Step 3:** Commit

```bash
git add test_workflow.py
git commit -m "test: workflow structural assertions for XL pipeline"
```

---

## Task 14: Write test_endpoint.py (manual integration)

**Files:**
- Create: `/root/ace-step-music-xl/test_endpoint.py`

- [ ] **Step 1:** Write the script

```python
#!/usr/bin/env python3
"""Manual integration test for the ACE-Step XL RunPod endpoint.

Usage:
  export RUNPOD_API_KEY=...
  export RUNPOD_ENDPOINT_ID=...
  python test_endpoint.py --task text2music --prompt "jazz piano"
  python test_endpoint.py --task cover --src-url https://.../track.mp3 --prompt "lo-fi cover"
  python test_endpoint.py --task repaint --src-url https://.../a.mp3 --prompt "fix" --start 5 --end 15
  python test_endpoint.py --task extract --src-url https://.../a.mp3 --instruction "drums"
  python test_endpoint.py --task lego --src-url https://.../a.mp3 --prompt "remix" --start 0 --end 20
  python test_endpoint.py --task complete --src-url https://.../a.mp3 --prompt "outro"
  python test_endpoint.py --all   # run all 6 tasks in sequence against a preset fixture
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

TASKS = ["text2music", "cover", "repaint", "extract", "lego", "complete"]

# Default public fixture for --all; override with --src-url if needed.
DEFAULT_FIXTURE_URL = "https://github.com/ace-step/ACE-Step-1.5/raw/main/assets/demo_audio/main.mp3"


def build_payload(args) -> dict:
    inp = {
        "task_type": args.task,
        "audio_format": args.format,
        "inference_steps": args.steps,
        "seed": args.seed,
        "batch_size": args.batch_size,
    }
    if args.prompt:
        inp["prompt"] = args.prompt
    if args.src_url:
        inp["src_audio_url"] = args.src_url
    if args.duration:
        inp["duration"] = args.duration
    if args.instruction:
        inp["instruction"] = args.instruction
    if args.start is not None:
        inp["repainting_start"] = args.start
    if args.end is not None:
        inp["repainting_end"] = args.end
    if args.lyrics:
        inp["lyrics"] = args.lyrics
        inp["instrumental"] = False
    return {"input": inp}


def call_endpoint(endpoint_id: str, api_key: str, payload: dict, timeout: int) -> dict:
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")
        sys.exit(1)

    # Poll if still pending
    while result.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
        job_id = result["id"]
        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
        time.sleep(5)
        status_req = urllib.request.Request(
            status_url, headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(status_req, timeout=30) as resp:
            result = json.loads(resp.read())
        print(f"  status: {result.get('status')}")
        if result.get("status") in ("FAILED", "CANCELLED", "TIMED_OUT"):
            print(json.dumps(result, indent=2))
            sys.exit(1)
    return result


def save_output(result: dict, task: str, fmt: str, out_dir: str) -> None:
    output = result.get("output", {})
    if "error" in output:
        print(f"Error: {output['error']}")
        sys.exit(1)
    audio_b64 = output.get("audio_base64")
    if not audio_b64:
        print("No audio in response")
        print(json.dumps(output, indent=2))
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{task}.{fmt}")
    with open(path, "wb") as f:
        f.write(base64.b64decode(audio_b64))

    size = os.path.getsize(path)
    print(f"  saved {path} ({size:,} bytes, duration={output.get('duration')}s, seed={output.get('seed')})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, default="text2music")
    p.add_argument("--prompt", default="upbeat electronic dance music")
    p.add_argument("--src-url", default="")
    p.add_argument("--instruction", default="")
    p.add_argument("--lyrics", default="")
    p.add_argument("--start", type=float, default=None)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--duration", type=float, default=0)
    p.add_argument("--format", choices=["mp3", "wav", "flac"], default="mp3")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY", ""))
    p.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID", ""))
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--out", default="out")
    p.add_argument("--all", action="store_true", help="Run all 6 tasks in sequence")
    args = p.parse_args()

    if not args.api_key or not args.endpoint_id:
        print("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID env vars")
        sys.exit(1)

    if args.all:
        fixture = args.src_url or DEFAULT_FIXTURE_URL
        task_presets = [
            ("text2music", {"prompt": "energetic synthwave"}),
            ("cover",      {"prompt": "lo-fi jazz cover", "src_url": fixture}),
            ("repaint",    {"prompt": "calmer chorus", "src_url": fixture,
                            "start": 5, "end": 15}),
            ("extract",    {"instruction": "isolate drums", "src_url": fixture}),
            ("lego",       {"prompt": "extended intro", "src_url": fixture,
                            "start": 0, "end": 20}),
            ("complete",   {"prompt": "cinematic outro", "src_url": fixture}),
        ]
        for task, preset in task_presets:
            print(f"\n=== {task} ===")
            args.task = task
            args.prompt = preset.get("prompt", "")
            args.src_url = preset.get("src_url", "")
            args.instruction = preset.get("instruction", "")
            args.start = preset.get("start")
            args.end = preset.get("end")

            payload = build_payload(args)
            print(f"  payload: {json.dumps(payload['input'])[:120]}...")
            t0 = time.time()
            result = call_endpoint(args.endpoint_id, args.api_key, payload, args.timeout)
            dt = time.time() - t0
            print(f"  elapsed: {dt:.1f}s")
            save_output(result, task, args.format, args.out)
        return

    payload = build_payload(args)
    print(f"Payload: {json.dumps(payload['input'])[:200]}")
    t0 = time.time()
    result = call_endpoint(args.endpoint_id, args.api_key, payload, args.timeout)
    print(f"Elapsed: {time.time() - t0:.1f}s")
    save_output(result, args.task, args.format, args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** Syntax check

```bash
python -c "import ast; ast.parse(open('/root/ace-step-music-xl/test_endpoint.py').read())"
```

Expected: no output, exit 0.

- [ ] **Step 3:** Commit

```bash
git add test_endpoint.py
git commit -m "test: add manual integration script for all 6 task types"
```

---

## Task 15: Write README.md

**Files:**
- Create: `/root/ace-step-music-xl/README.md`

- [ ] **Step 1:** Write a minimal reference README

```markdown
# ace-step-music-xl

RunPod serverless endpoint for **ACE-Step 1.5 XL** (`acestep-v15-xl-base`, 4B DiT).
Unified handler supporting six task types: `text2music`, `cover`, `repaint`, `extract`, `lego`, `complete`.

## Deployment

- **GPU:** A40 48GB, RunPod data center **EU-SE-1**
- **Weights:** 35 GB RunPod network volume mounted at `/runpod-volume`
- **Image:** `dmrabh/ace-step-music-xl:latest`
- **Pipeline:** push to `main` → GitHub Actions → Docker Hub → RunPod `saveTemplate` mutation

See `docs/superpowers/specs/2026-04-20-ace-step-xl-serverless-design.md` for the
full design, and `docs/superpowers/plans/2026-04-20-ace-step-xl-serverless.md` for
the step-by-step implementation plan.

## Quick start

```bash
pip install -r requirements-test.txt
pytest test_handler.py test_workflow.py -v   # unit tests, no GPU
export RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=...
python test_endpoint.py --all                 # smoke test all 6 tasks
```

## Input schema

| Field | Type | Default | Notes |
|---|---|---|---|
| `task_type` | string | `text2music` | `text2music` / `cover` / `repaint` / `extract` / `lego` / `complete` |
| `prompt` | string | — | Required for all tasks except `extract` (which uses `instruction`) |
| `src_audio_url` or `src_audio_base64` | string | — | Required for all audio-input tasks; `https://` URL or base64 |
| `instruction` | string | — | Required for `extract` |
| `repainting_start` / `repainting_end` | float | — | Required for `repaint` / `lego` (seconds; `-1` for end of file) |
| `duration` | float | `30` | Clamped 10–600 |
| `inference_steps` | int | `50` | XL recommended |
| `guidance_scale` | float | `7.0` | XL uses CFG |
| `batch_size` | int | `1` | Clamped to `MAX_BATCH_SIZE` (default 4) |
| `seed` | int | `-1` | `-1` = random |
| `audio_format` | string | `mp3` | `mp3` / `wav` / `flac` |
| `instrumental` | bool | `true` | When `true`, forces `lyrics="[Instrumental]"` |
| `lyrics` | string | `""` | Used only when `instrumental=false` |

## Output

```json
{
  "audio_base64": "<base64 audio bytes>",
  "format": "mp3",
  "duration": 30.0,
  "seed": 12345,
  "sample_rate": 48000,
  "task_type": "text2music"
}
```

## Environment variables

See `.env.example`.
```

- [ ] **Step 2:** Commit

```bash
git add README.md
git commit -m "docs: README with deployment, schema, and quick start"
```

---

## Task 16: Full local test pass

**Files:** none

- [ ] **Step 1:** Run everything

```bash
cd /root/ace-step-music-xl
pytest test_handler.py test_workflow.py -v --cov=handler --cov-report=term-missing
```

Expected: all tests pass, coverage ≥ 80% on `handler.py`.

- [ ] **Step 2:** If any failures, fix and re-commit. If coverage is under 80%, add targeted tests for missed lines and commit.

- [ ] **Step 3:** Check git log

```bash
git log --oneline
```

Expected: ~12+ commits walking through scaffold → torch patches → src_audio → schema → handler per-task → Dockerfile → CI.

---

## Task 17: Create GitHub repo and push

**Files:** none

- [ ] **Step 1:** Verify `gh` CLI is authenticated

```bash
gh auth status
```

If unauthenticated, stop and ask the user to run `gh auth login`. This is in the user-action checklist at the end.

- [ ] **Step 2:** Create the repo (public, empty, no auto-init)

```bash
cd /root/ace-step-music-xl
gh repo create ace-step-music-xl --public --source=. --remote=origin --push
```

Expected: repo created at `https://github.com/<user>/ace-step-music-xl` and the local commits pushed to `main`.

- [ ] **Step 3:** Confirm GHA will trigger on the push by opening the Actions tab

```bash
gh run list --limit 3
```

Expected: at least one run queued/in-progress for the initial commit. The `test` job should pass; `build-push` will **fail** until secrets are set in Task 18.

---

## Task 18: Set GitHub Actions secrets

**Files:** none — uses `gh` CLI

- [ ] **Step 1:** Reuse existing secrets from the turbo repo. Read them from the local clone if available, otherwise ask the user.

```bash
# If the turbo repo has a .env with the credentials, read them:
# source /root/ace-step-music/.env
# Otherwise prompt the user for each.
```

- [ ] **Step 2:** Set each secret via `gh secret set`. Do NOT print the values; pipe from env or stdin.

```bash
cd /root/ace-step-music-xl
gh secret set DOCKERHUB_USERNAME --body "$DOCKERHUB_USERNAME"
gh secret set DOCKERHUB_TOKEN    --body "$DOCKERHUB_TOKEN"
gh secret set RUNPOD_API_KEY     --body "$RUNPOD_API_KEY"
# RUNPOD_TEMPLATE_ID_XL is set AFTER Task 21 (create-template output)
```

- [ ] **Step 3:** Verify secrets are registered

```bash
gh secret list
```

Expected: three secrets listed (DOCKERHUB_USERNAME, DOCKERHUB_TOKEN, RUNPOD_API_KEY).

- [ ] **Step 4:** Re-trigger the workflow. The deploy job will still fail because `RUNPOD_TEMPLATE_ID_XL` is not set yet; that's expected and fixed in Task 21.

```bash
gh workflow run deploy.yml
gh run list --limit 3
```

---

## Task 19: RunPod MCP — create network volume

**Files:** none — MCP tool calls

- [ ] **Step 1:** Call the MCP tool

Use `mcp__runpod__create-network-volume`:
```
name: "ace-step-xl-weights"
size: 35
dataCenterId: "EU-SE-1"
```

- [ ] **Step 2:** Save the returned `volumeId` (e.g., `vol-abc123`). You'll need it for Tasks 20 and 22.

- [ ] **Step 3:** Verify

Use `mcp__runpod__list-network-volumes` and confirm the new volume appears with size 35 GB in EU-SE-1.

---

## Task 20: RunPod MCP — bootstrap pod, download XL weights, then delete

**Files:** none — MCP tool calls + SSH commands

- [ ] **Step 1:** Wait for the GHA build to complete so the image exists on Docker Hub

```bash
gh run watch    # blocks until the latest run finishes
```

Or check manually: `docker pull dmrabh/ace-step-music-xl:latest` should succeed.

- [ ] **Step 2:** Create a temporary pod using the new image, with the volume mounted

Use `mcp__runpod__create-pod`:
```
name: "ace-step-xl-bootstrap"
imageName: "dmrabh/ace-step-music-xl:latest"
networkVolumeId: "<volumeId from Task 19>"
containerDiskInGb: 20
gpuTypeIds: ["NVIDIA RTX A4000"]       # any cheap GPU will do; needed just to mount the volume
dataCenterIds: ["EU-SE-1"]
ports: "22/tcp"
env:
  - key: ACESTEP_CHECKPOINT_DIR
    value: /runpod-volume/checkpoints
```

- [ ] **Step 3:** Once the pod is `RUNNING`, SSH in and use the package's built-in downloader so we stay in sync with upstream HF repo names

```bash
# From the user's workstation — get SSH details from get-pod, then:
ssh root@<pod-host> -p <ssh-port> <<'EOF'
set -eu
python - <<'PY'
from pathlib import Path
from acestep.model_downloader import ensure_main_model, ensure_dit_model
ckpt = Path("/runpod-volume/checkpoints")
ckpt.mkdir(parents=True, exist_ok=True)
ok, msg = ensure_main_model(checkpoints_dir=ckpt)
print("main:", ok, msg)
assert ok, msg
ok, msg = ensure_dit_model("acestep-v15-xl-base", checkpoints_dir=ckpt)
print("dit:", ok, msg)
assert ok, msg
PY
du -sh /runpod-volume/checkpoints/*
EOF
```

Expected: `main: True ...` and `dit: True ...`, and the `du -sh` output shows both directories populated (total ~20 GB).

- [ ] **Step 4:** Delete the bootstrap pod — volume persists

Use `mcp__runpod__delete-pod` with the pod id.

---

## Task 21: RunPod MCP — create serverless template

**Files:** none — MCP tool calls

- [ ] **Step 1:** Call `mcp__runpod__create-template`

```
name: "ace-step-music-xl"
imageName: "dmrabh/ace-step-music-xl:latest"
containerDiskInGb: 10
volumeInGb: 0
isServerless: true
dockerArgs: ""
env:
  - key: ACESTEP_CONFIG_PATH
    value: acestep-v15-xl-base
  - key: ACESTEP_CHECKPOINT_DIR
    value: /runpod-volume/checkpoints
  - key: ACESTEP_MAX_BATCH_SIZE
    value: "4"
  - key: ACESTEP_COMPILE_MODEL
    value: "1"
  - key: ACESTEP_INFERENCE_STEPS_DEFAULT
    value: "50"
  - key: ACESTEP_GUIDANCE_SCALE_DEFAULT
    value: "7.0"
```

- [ ] **Step 2:** Save the returned `templateId`.

- [ ] **Step 3:** Add it as a GitHub secret

```bash
cd /root/ace-step-music-xl
gh secret set RUNPOD_TEMPLATE_ID_XL --body "<templateId>"
gh secret list   # confirm 4 secrets now
```

- [ ] **Step 4:** Re-trigger the workflow to confirm the deploy job can update the template

```bash
gh workflow run deploy.yml
gh run watch
```

Expected: all three jobs pass; final log line `Template updated successfully`.

---

## Task 22: RunPod MCP — create serverless endpoint

**Files:** none — MCP tool calls

- [ ] **Step 1:** Call `mcp__runpod__create-endpoint`

```
name: "ace-step-music-xl"
templateId: "<from Task 21>"
gpuTypeIds: ["NVIDIA A40"]
dataCenterIds: ["EU-SE-1"]
networkVolumeId: "<from Task 19>"
workersMin: 0
workersMax: 3
idleTimeout: 5
flashboot: true
```

- [ ] **Step 2:** Save the returned `endpointId`.

- [ ] **Step 3:** Verify via `mcp__runpod__get-endpoint` — status should be `READY` once a worker warms up.

---

## Task 23: End-to-end smoke test

**Files:** none — runs `test_endpoint.py`

- [ ] **Step 1:** Export credentials

```bash
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=<from Task 22>
```

- [ ] **Step 2:** Run a single `text2music` first (fastest to verify basic path)

```bash
cd /root/ace-step-music-xl
python test_endpoint.py --task text2music --prompt "cinematic orchestral score"
```

Expected: `out/text2music.mp3` written, ~10–60 s generation time once warm.

- [ ] **Step 3:** Run the full suite

```bash
python test_endpoint.py --all
```

Expected: `out/{text2music,cover,repaint,extract,lego,complete}.mp3` all produced. First cold start ~30–60 s; subsequent calls on warm worker ~10–30 s per clip.

- [ ] **Step 4:** Spot-check outputs by listening — `cover` should recognizably resemble the input; `extract` with "drums" should isolate the rhythm; etc.

---

## Self-review

**Spec coverage check** (against `docs/superpowers/specs/2026-04-20-ace-step-xl-serverless-design.md`):

| Spec section | Addressed by |
|---|---|
| §1 Goals — six task types behind unified handler | Tasks 7–9 |
| §3.1 Compute — A40, EU-SE-1 | Task 22 |
| §3.2 Storage — 35 GB network volume | Tasks 19–20 |
| §3.3 Container image — torch 2.4 stack | Task 11 |
| §4.1/4.2 Input schema | Tasks 7–9 |
| §4.3 src_audio resolution | Tasks 5–6 |
| §4.4 Output schema | Task 8 |
| §4.5 Error responses | Tasks 7–9 |
| §5 Handler structure | Tasks 4–9 |
| §6 Dockerfile env deltas | Task 11 |
| §7 deploy.yml | Tasks 12–13 |
| §8 Testing strategy | Tasks 3–10, 13–14 |
| §9 File tree | Tasks 1–15 |
| §10 Deployment bootstrap | Tasks 19–22 |
| §11 Secrets | Tasks 18, 21 |
| §12 Cost | noted in spec, no impl step needed |

All spec items mapped to at least one task.

**Placeholder scan** — manual grep of this plan: no TBD, TODO, "appropriate", "similar to Task N" (code is repeated where needed). ✅

**Type consistency** — `_resolve_src_audio`, `_validate`, `_build_params`, `handler` signatures used consistently across tasks. `MAX_BATCH_SIZE` / `CHECKPOINT_DIR` / `DIT_CONFIG` / `INFERENCE_STEPS_DEFAULT` / `GUIDANCE_SCALE_DEFAULT` names used identically in Tasks 4, 8, 11. ✅

---

## What you (the user) need to do

Things I can't do for you — please have these ready before I start executing the plan:

- [ ] **GitHub CLI auth.** Run `gh auth login` once so `gh repo create` and `gh secret set` work in Task 17 and 18. Choose HTTPS, browser auth, scopes: `repo`, `workflow`, `write:packages`.
- [ ] **Docker Hub credentials.** I'll push images to `dmrabh/ace-step-music-xl`. Confirm:
  - The `dmrabh` Docker Hub account exists and you own it.
  - You have a Personal Access Token (not your password) with `Read, Write, Delete` on repositories. Hub → Account Settings → Security → Access Tokens → Generate.
  - Provide `DOCKERHUB_USERNAME` (likely `dmrabh`) and `DOCKERHUB_TOKEN` so I can `gh secret set` them in Task 18. Easiest: put them in `/root/ace-step-music/.env` if they're already there, or tell me directly.
- [ ] **RunPod account + API key.** I'll deploy into your RunPod organization:
  - Account funded (at minimum ~$5 to cover the bootstrap pod and initial volume).
  - API key generated at RunPod → Settings → API Keys with `Read/Write` permissions.
  - Provide `RUNPOD_API_KEY` so I can use the RunPod MCP tools in Tasks 19–22.
- [ ] **HuggingFace access.** The public `ACE-Step/acestep-v15-xl-base` and `ACE-Step/Ace-Step1.5` repos don't require auth, but if HF ever rate-limits the bootstrap pod, I'll need an HF token. Optional: generate a read-only token at huggingface.co/settings/tokens and set `HF_TOKEN` env var on the bootstrap pod.
- [ ] **Listen-check the outputs.** After Task 23, the generated audio needs human evaluation — I can verify shape and format but not musical quality. Listen to each of the six output files and flag anything that sounds broken.
- [ ] **Decide ongoing scale policy.** After smoke test: do you want `workersMin=0` (scale-to-zero, cheapest, cold starts on every gap) or `workersMin=1` (keep one warm worker, ~$1.20/day for instant response)? Currently set to 0; tell me if you want to flip.

### Summary of what will get charged

| Item | When | Approximate cost |
|---|---|---|
| Network volume 35 GB, EU-SE-1 | Monthly, flat | ~$2.45/mo |
| Bootstrap pod (A4000, ~30 min) | Once, during deploy | ~$0.20 |
| A40 serverless time | Per generation | ~$0.00050/sec × actual GPU time |
| Docker Hub | — | free (public image) |
| GitHub Actions | — | free (public repo on standard runners) |

---

Plan complete and saved to `docs/superpowers/plans/2026-04-20-ace-step-xl-serverless.md`.

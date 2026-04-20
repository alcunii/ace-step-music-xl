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


def handler(job: dict) -> dict:
    """Stub — will be filled in Task 8."""
    return {"error": "handler not implemented yet"}


# Load models + start worker (skipped during test import because mocks replace
# AceStepHandler/LLMHandler; they'll still be "loaded" but with MagicMocks).
# Tests pop sys.modules['handler'] before importing, so this runs each time,
# but the mocks make it fast and safe.
load_models()
runpod.serverless.start({"handler": handler})

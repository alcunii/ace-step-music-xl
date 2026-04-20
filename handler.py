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
    "ACESTEP_CHECKPOINTS_DIR",
    os.environ.get("ACESTEP_CHECKPOINT_DIR", "/runpod-volume/checkpoints"),
)
# The upstream ACE-Step package reads ACESTEP_CHECKPOINTS_DIR (plural) inside
# AceStepHandler.initialize_service — not the checkpoints_dir kwarg. Set it
# here so both the downloader and the service use the same path.
os.environ["ACESTEP_CHECKPOINTS_DIR"] = CHECKPOINT_DIR
LM_MODEL = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")
LM_BACKEND = os.environ.get("ACESTEP_LM_BACKEND", "vllm")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPILE_MODEL = os.environ.get("ACESTEP_COMPILE_MODEL", "0") in ("1", "true")
OFFLOAD_DIT_TO_CPU = os.environ.get("ACESTEP_OFFLOAD_DIT_TO_CPU", "0") in ("1", "true")
MAX_BATCH_SIZE = int(os.environ.get("ACESTEP_MAX_BATCH_SIZE", "2"))
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

    def _sdpa_gqa_compat(*args, **kwargs):
        """Implement enable_gqa=True for torch 2.4 by repeat-interleaving K/V heads.

        PyTorch 2.5 added the `enable_gqa` kwarg which tells SDPA that KV has
        fewer heads than Q (Grouped-Query Attention). On 2.4 the kwarg is
        unknown AND the dimension check fails. Simulate it by broadcasting
        K and V to match Q's head count before calling SDPA.
        """
        enable_gqa = kwargs.pop("enable_gqa", False)
        if not enable_gqa:
            return _orig_sdpa(*args, **kwargs)

        # Positional args order: query, key, value, attn_mask?, dropout_p?, is_causal?, scale?
        if len(args) < 3:
            return _orig_sdpa(*args, **kwargs)
        query, key, value = args[0], args[1], args[2]
        rest = args[3:]

        q_heads = query.shape[-3]
        kv_heads = key.shape[-3]
        if kv_heads > 0 and q_heads % kv_heads == 0 and q_heads != kv_heads:
            repeat = q_heads // kv_heads
            key = key.repeat_interleave(repeat, dim=-3)
            value = value.repeat_interleave(repeat, dim=-3)
        return _orig_sdpa(query, key, value, *rest, **kwargs)

    F.scaled_dot_product_attention = _sdpa_gqa_compat


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
    except ImportError as e:  # pragma: no cover
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
    shift = float(job_input.get("shift", 3.0))
    use_adg = bool(job_input.get("use_adg", False))
    cfg_interval_start = float(job_input.get("cfg_interval_start", 0.0))
    cfg_interval_end = float(job_input.get("cfg_interval_end", 1.0))
    infer_method = job_input.get("infer_method", "ode")

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
        use_adg=use_adg,
        cfg_interval_start=cfg_interval_start,
        cfg_interval_end=cfg_interval_end,
        infer_method=infer_method,
    )

    if task_type != "text2music":
        params_kwargs["src_audio"] = src_audio_path
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


# Load models + start worker (skipped during test import because mocks replace
# AceStepHandler/LLMHandler; they'll still be "loaded" but with MagicMocks).
# Tests pop sys.modules['handler'] before importing, so this runs each time,
# but the mocks make it fast and safe.
load_models()
runpod.serverless.start({"handler": handler})

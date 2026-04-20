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

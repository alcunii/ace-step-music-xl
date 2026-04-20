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


# ---------------------------------------------------------------------------
# TestHandlerAudioInputTasks
# ---------------------------------------------------------------------------
class TestHandlerAudioInputTasks:
    def _input_with_src(self, extra: dict) -> dict:
        b64 = base64.b64encode(_short_mp3_bytes()).decode()
        return {"src_audio_base64": b64, **extra}

    def test_cover_passes_src_audio(self):
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
        assert params.src_audio is not None
        # LM auto-skipped -> thinking forced False
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
        src_path = captured["params"].src_audio
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


# ---------------------------------------------------------------------------
# TestLoadModelsFailures — coverage for load_models() failure branches
# ---------------------------------------------------------------------------
class TestLoadModelsFailures:
    """Verify RuntimeError is raised when init sub-steps report failure."""

    def test_download_main_model_failure_raises(self):
        sys.modules.pop("handler", None)
        dl = sys.modules["acestep.model_downloader"]
        dl.ensure_main_model.return_value = (False, "hf down")
        try:
            with pytest.raises(RuntimeError, match="Failed to download main model"):
                importlib.import_module("handler")
        finally:
            dl.ensure_main_model.return_value = (True, "ok")

    def test_download_dit_model_failure_raises(self):
        sys.modules.pop("handler", None)
        dl = sys.modules["acestep.model_downloader"]
        dl.ensure_dit_model.return_value = (False, "dit down")
        try:
            with pytest.raises(RuntimeError, match="Failed to download DiT model"):
                importlib.import_module("handler")
        finally:
            dl.ensure_dit_model.return_value = (True, "ok")

    def test_dit_init_failure_raises(self):
        sys.modules.pop("handler", None)
        ace_cls = sys.modules["acestep.handler"].AceStepHandler
        ace_cls.return_value.initialize_service.return_value = ("boom", False)
        try:
            with pytest.raises(RuntimeError, match="DiT init failed"):
                importlib.import_module("handler")
        finally:
            ace_cls.return_value.initialize_service.return_value = ("ok", True)

    def test_lm_init_failure_raises(self):
        sys.modules.pop("handler", None)
        llm_cls = sys.modules["acestep.llm_inference"].LLMHandler
        llm_cls.return_value.initialize.return_value = ("llm boom", False)
        try:
            with pytest.raises(RuntimeError, match="LM init failed"):
                importlib.import_module("handler")
        finally:
            llm_cls.return_value.initialize.return_value = ("ok", True)


# ---------------------------------------------------------------------------
# TestHandlerErrorBranches — coverage for handler() error paths
# ---------------------------------------------------------------------------
class TestHandlerErrorBranches:
    """Cover the 'no audio returned' and 'unhandled exception' branches."""

    def test_no_audio_returned_errors(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        gen_mock.side_effect = None
        gen_mock.return_value = FakeGenerationResult(success=True, audios=[])
        result = handler_fn({"input": {"prompt": "x"}})
        assert "error" in result
        assert "No audio" in result["error"]

    def test_unhandled_exception_returns_internal_error(self):
        handler_fn = _import_handler_func()
        gen_mock = sys.modules["acestep.inference"].generate_music
        gen_mock.side_effect = RuntimeError("unexpected kaboom")
        result = handler_fn({"input": {"prompt": "x"}})
        assert "error" in result
        assert "Internal error" in result["error"]
        assert "kaboom" in result["error"]

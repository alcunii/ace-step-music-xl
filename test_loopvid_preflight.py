from unittest.mock import patch

import pytest

from scripts.loopvid.preflight import (
    PreflightError,
    check_env_vars,
    check_ffmpeg_available,
    check_endpoint_workers,
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
    check_ffmpeg_available()  # ffmpeg is installed for the rest of the suite to work


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

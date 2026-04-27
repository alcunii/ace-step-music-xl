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

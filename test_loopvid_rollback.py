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

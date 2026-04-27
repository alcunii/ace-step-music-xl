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
    "music": ("music",),                        # whole music/ subdir
    "image": ("still.png",),
    "video": ("video",),                        # whole video/ subdir
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
    run_dir.rename(target)
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

    for item in list(run_dir.iterdir()):
        if item.name in keep_paths:
            continue
        shutil.move(str(item), str(failed / item.name))

    m = load_manifest(run_dir)
    keep_steps = {"plan"}                         # plan is always kept implicitly
    for k in keep:
        keep_steps.add(k)
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

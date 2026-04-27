"""Run manifest — single source of truth for step state.

Atomic-written via .tmp + os.replace. If the orchestrator crashes between
two steps, the manifest accurately reflects the world."""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
STEP_NAMES = ("plan", "music", "image", "video", "loop_build", "mux")


@dataclass
class StepStatus:
    status: str = "pending"     # pending | in_progress | done | failed
    attempts: int = 0


@dataclass
class RunManifest:
    run_id: str
    schema_version: int
    created_at: str
    last_updated: str
    args: dict
    endpoints: dict
    steps: dict
    cost_estimate_usd: float = 0.0
    cost_actual_usd: float = 0.0
    failures: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "args": self.args,
            "endpoints": self.endpoints,
            "steps": self.steps,
            "cost_estimate_usd": self.cost_estimate_usd,
            "cost_actual_usd": self.cost_actual_usd,
            "failures": self.failures,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunManifest":
        return cls(**d)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_manifest(run_id: str, args: dict, endpoints: Optional[dict] = None) -> RunManifest:
    now = _now_iso()
    return RunManifest(
        run_id=run_id,
        schema_version=SCHEMA_VERSION,
        created_at=now,
        last_updated=now,
        args=dict(args),
        endpoints=endpoints or {},
        steps={name: {"status": "pending", "attempts": 0} for name in STEP_NAMES},
    )


def _path(run_dir: Path) -> Path:
    return Path(run_dir) / MANIFEST_FILENAME


def save_manifest(run_dir: Path, m: RunManifest) -> None:
    """Atomic write: .tmp → os.replace."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    m.last_updated = _now_iso()
    target = _path(run_dir)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(m.to_dict(), indent=2, sort_keys=True))
    os.replace(tmp, target)


def load_manifest(run_dir: Path) -> RunManifest:
    p = _path(run_dir)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found at {p}")
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"corrupt manifest at {p}: {e}") from e
    return RunManifest.from_dict(d)


def mark_step_done(run_dir: Path, step: str, extra: Optional[dict] = None) -> None:
    m = load_manifest(run_dir)
    m.steps[step] = {
        **m.steps.get(step, {}),
        "status": "done",
        "committed_at": _now_iso(),
        **(extra or {}),
    }
    save_manifest(run_dir, m)


def mark_step_in_progress(run_dir: Path, step: str, extra: Optional[dict] = None) -> None:
    m = load_manifest(run_dir)
    cur = m.steps.get(step, {})
    m.steps[step] = {
        **cur,
        "status": "in_progress",
        "attempts": cur.get("attempts", 0) + 1,
        **(extra or {}),
    }
    save_manifest(run_dir, m)


def mark_step_failed(run_dir: Path, step: str, error: str, attempts: int) -> None:
    m = load_manifest(run_dir)
    m.steps[step] = {
        **m.steps.get(step, {}),
        "status": "failed",
        "attempts": attempts,
    }
    m.failures.append({
        "step": step, "error": error,
        "ts": _now_iso(), "attempts": attempts,
    })
    save_manifest(run_dir, m)

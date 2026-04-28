#!/usr/bin/env python3
"""Capybara + Tea Loop — CLI entry.

Generates a 60-min 1280x704 looping music video with a locked Studio Ghibli
capybara+tea aesthetic. Setting varies per run (random pick or --setting).

No LLM call. Pure template via scripts/loopvid/capybara_preset.py.

Usage:
  python3 scripts/capybara_tea_loop.py
  python3 scripts/capybara_tea_loop.py --setting forest_hot_spring
  python3 scripts/capybara_tea_loop.py --duration 300 --seed 42
  python3 scripts/capybara_tea_loop.py --resume capybara-20260428T120000Z
  python3 scripts/capybara_tea_loop.py --rollback capybara-20260428T120000Z

Spec: docs/superpowers/specs/2026-04-28-capybara-tea-loop-design.md
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from pathlib import Path

# Ensure the loopvid package is importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loopvid.capybara_preset import (  # type: ignore
    CAPYBARA_GENRE,
    CAPYBARA_LTX_NEGATIVE,
    CAPYBARA_SEEDREAM_CONSTRAINTS,
    CAPYBARA_SETTINGS,
    PRESET_SENTINEL_KEY,
    build_plan_dict,
    get_setting_by_key,
    pick_setting,
)
from loopvid.cost import cost_breakdown_lines, estimate_run_cost  # type: ignore
from loopvid.orchestrator import OrchestratorConfig, run_orchestrator  # type: ignore
from loopvid.rollback import (  # type: ignore
    RollbackError, rollback_forensic, rollback_hard, rollback_with_keep,
)


DEFAULT_LTX_ENDPOINT = "1g0pvlx8ar6qns"
DEFAULT_ACE_STEP_ENDPOINT = "nwqnd0duxc6o38"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "out" / "capybara_tea"


def _autogen_run_id() -> str:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"capybara-{ts}"


def _parse_csv(s):
    if s is None:
        return None
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollback", help="Rollback the named run instead of running pipeline")
    p.add_argument("--resume", help="Resume the named run from its last completed step")
    p.add_argument("--run-id", help="Explicit run id (default: autogen capybara-<ts>)")

    valid_keys = sorted(s["key"] for s in CAPYBARA_SETTINGS)
    p.add_argument("--setting", choices=valid_keys, default=None,
                   help=f"Pick a specific scene; default: random from {len(valid_keys)} curated")
    p.add_argument("--seed", type=int, default=None,
                   help="Deterministic seed for setting selection (and downstream RNG)")
    p.add_argument("--duration", type=int, default=3600,
                   help="Target seconds (default 3600 = 60 min)")

    p.add_argument("--ltx-endpoint",
                   default=os.environ.get("RUNPOD_LTX_ENDPOINT_ID", DEFAULT_LTX_ENDPOINT))
    p.add_argument("--ace-step-endpoint",
                   default=os.environ.get("RUNPOD_ACE_STEP_ENDPOINT_ID", DEFAULT_ACE_STEP_ENDPOINT))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))

    p.add_argument("--only", help="Comma-separated steps to run only")
    p.add_argument("--skip", help="Comma-separated steps to skip")
    p.add_argument("--force", action="store_true", help="Re-run completed steps")
    p.add_argument("--dry-run", action="store_true", help="Show plan + cost, no API calls")

    p.add_argument("--max-cost", type=float, help="Abort if estimated cost exceeds this USD value")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompts")

    p.add_argument("--keep", help="--rollback: comma-separated keepers (music,image,video)")
    p.add_argument("--hard", action="store_true", help="--rollback: hard delete (with confirm)")

    return p


def cmd_rollback(args, out_dir: Path) -> int:
    run_dir = out_dir / args.rollback
    try:
        if args.hard:
            confirmed = args.yes or _confirm(f"Permanently delete {run_dir}? [y/N] ")
            rollback_hard(run_dir, confirm=confirmed)
            print(f"✓ deleted {run_dir}")
        elif args.keep:
            keep = _parse_csv(args.keep)
            failed = rollback_with_keep(run_dir, keep=keep)
            print(f"✓ kept {keep} in {run_dir}; rest moved to {failed}")
        else:
            failed = rollback_forensic(run_dir)
            print(f"✓ moved {run_dir} to {failed}")
        return 0
    except RollbackError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


def cmd_run(args, out_dir: Path) -> int:
    setting = (get_setting_by_key(args.setting) if args.setting
               else pick_setting(seed=args.seed))
    print(f"▸ setting: {setting['key']}")

    run_id = args.resume or args.run_id or _autogen_run_id()
    plan_dict = build_plan_dict(setting)

    cfg = OrchestratorConfig(
        run_id=run_id,
        run_dir=out_dir / run_id,
        genre=CAPYBARA_GENRE,
        mood=plan_dict["mood"],
        duration_sec=args.duration,
        ace_step_endpoint=args.ace_step_endpoint,
        ltx_endpoint=args.ltx_endpoint,
        runpod_api_key=os.environ.get("RUNPOD_API_KEY", ""),
        openrouter_api_key="",  # not used — preset_plan_dict skips the LLM
        replicate_api_token=os.environ.get("REPLICATE_API_TOKEN", ""),
        only=_parse_csv(args.only),
        skip=_parse_csv(args.skip),
        force=args.force,
        dry_run=args.dry_run,
        max_cost=args.max_cost,
        seedream_constraints=CAPYBARA_SEEDREAM_CONSTRAINTS,
        ltx_negative=CAPYBARA_LTX_NEGATIVE,
        extra_archetype_keys={PRESET_SENTINEL_KEY},
        extra_motion_archetypes={PRESET_SENTINEL_KEY},
        preset_plan_dict=plan_dict,
    )

    if not args.dry_run and not args.yes:
        skip = set(cfg.skip or ())
        cost = estimate_run_cost(duration_sec=cfg.duration_sec, skip=skip)
        print(f"Estimated cost: ${cost:.2f}")
        for line in cost_breakdown_lines(duration_sec=cfg.duration_sec, skip=skip):
            print(line)
        if not _confirm("Continue? [y/N] "):
            print("aborted.")
            return 1

    final = run_orchestrator(cfg)
    print(f"Done: {final}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.rollback:
        return cmd_rollback(args, out_dir)
    return cmd_run(args, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())

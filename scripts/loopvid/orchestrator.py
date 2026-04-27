"""Top-level orchestrator — walks the 6 pipeline steps in order."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from scripts.loopvid.constants import (
    SEGMENT_COUNT_60MIN, SEGMENT_DURATION_SEC,
)
from scripts.loopvid.cost import estimate_run_cost, cost_breakdown_lines, enforce_budget, segments_for_duration
from scripts.loopvid.image_pipeline import generate_still, build_seedream_prompt
from scripts.loopvid.llm_planner import plan
from scripts.loopvid.loop_build import concat_clips_with_xfades, add_loop_seam_fade
from scripts.loopvid.manifest import (
    load_manifest, new_manifest, save_manifest,
    mark_step_done, mark_step_in_progress,
)
from scripts.loopvid.music_pipeline import run_music_pipeline, stitch_segments
from scripts.loopvid.mux import final_assembly
from scripts.loopvid.plan_schema import validate_plan_dict
from scripts.loopvid.preflight import run_preflight
from scripts.loopvid.video_pipeline import (
    run_video_pipeline, slice_audio_chunks, stable_clip_seed,
)


@dataclass
class OrchestratorConfig:
    run_id: str
    run_dir: Path
    genre: str
    mood: str
    duration_sec: int
    ace_step_endpoint: str
    ltx_endpoint: str
    runpod_api_key: str
    openrouter_api_key: str
    replicate_api_token: str
    only: Optional[tuple] = None
    skip: Optional[tuple] = None
    force: bool = False
    dry_run: bool = False
    max_cost: Optional[float] = None


STEP_ORDER = ("plan", "music", "image", "video", "loop_build", "mux")


def _should_run(step: str, cfg: OrchestratorConfig, current_status: str) -> bool:
    if cfg.only and step not in cfg.only:
        return False
    if cfg.skip and step in cfg.skip:
        return False
    if cfg.force:
        return True
    return current_status != "done"


def _print(msg: str) -> None:
    print(msg, flush=True)


def run_orchestrator(cfg: OrchestratorConfig) -> Path:
    """Execute pipeline. Returns path to final.mp4."""
    cfg.run_dir = Path(cfg.run_dir)

    if cfg.dry_run:
        _print(f"[DRY RUN] genre={cfg.genre} mood={cfg.mood} duration={cfg.duration_sec}s")
        for line in cost_breakdown_lines(duration_sec=cfg.duration_sec):
            _print(line)
        return cfg.run_dir / "final.mp4"

    estimated = estimate_run_cost(duration_sec=cfg.duration_sec)
    if cfg.max_cost is not None:
        enforce_budget(estimated, cfg.max_cost)

    run_preflight(
        runpod_api_key=cfg.runpod_api_key,
        ace_step_endpoint=cfg.ace_step_endpoint,
        ltx_endpoint=cfg.ltx_endpoint,
    )

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    try:
        m = load_manifest(cfg.run_dir)
    except FileNotFoundError:
        m = new_manifest(cfg.run_id, {
            "genre": cfg.genre, "mood": cfg.mood, "duration_sec": cfg.duration_sec,
        }, endpoints={"ltx": cfg.ltx_endpoint, "ace_step": cfg.ace_step_endpoint})
        save_manifest(cfg.run_dir, m)

    plan_path = cfg.run_dir / "plan.json"

    # Step 1: plan
    if _should_run("plan", cfg, m.steps["plan"]["status"]):
        mark_step_in_progress(cfg.run_dir, "plan")
        _print("▸ plan (LLM)")
        plan_obj = plan(
            genre=cfg.genre, mood=cfg.mood,
            api_key=cfg.openrouter_api_key,
            raw_response_path=str(cfg.run_dir / "plan_raw.json"),
        )
        plan_path.write_text(json.dumps({
            "genre": plan_obj.genre,
            "mood": plan_obj.mood,
            "music_palette": plan_obj.music_palette,
            "music_segment_descriptors": plan_obj.music_segment_descriptors,
            "music_bpm": plan_obj.music_bpm,
            "seedream_scene": plan_obj.seedream_scene,
            "seedream_style": plan_obj.seedream_style,
            "motion_prompts": plan_obj.motion_prompts,
            "motion_archetype": plan_obj.motion_archetype,
            "image_archetype_key": plan_obj.image_archetype_key,
        }, indent=2, sort_keys=True))
        mark_step_done(cfg.run_dir, "plan")
        _print("✓ plan committed")
    else:
        _print("✓ plan (cached)")

    plan_dict = json.loads(plan_path.read_text())
    plan_obj = validate_plan_dict(plan_dict)

    # Step 2: music
    music_dir = cfg.run_dir / "music"
    master_path = music_dir / "master.mp3"
    if _should_run("music", cfg, m.steps["music"]["status"]):
        mark_step_in_progress(cfg.run_dir, "music")
        n_segments = segments_for_duration(cfg.duration_sec)
        _print(f"▸ music ({n_segments} segments × {SEGMENT_DURATION_SEC}s)")
        prompts = [
            f"{plan_obj.music_palette}, {seg['descriptors']}"
            for seg in plan_obj.music_segment_descriptors[:n_segments]
        ]
        seeds = [stable_clip_seed(cfg.run_id, i) for i in range(1, n_segments + 1)]
        seg_paths = run_music_pipeline(
            prompts=prompts, duration_sec=SEGMENT_DURATION_SEC, seeds=seeds,
            out_dir=music_dir,
            endpoint_id=cfg.ace_step_endpoint, api_key=cfg.runpod_api_key,
            on_segment_done=lambda i, p: _print(f"  ✓ seg {i} ({p.stat().st_size:,} B)"),
        )
        if not master_path.exists() and seg_paths:
            stitch_segments(seg_paths, master_path)
        mark_step_done(cfg.run_dir, "music", extra={"master_committed": True})
        _print("✓ music master committed")
    else:
        _print("✓ music (cached)")

    # Step 3: image
    still_path = cfg.run_dir / "still.png"
    if _should_run("image", cfg, m.steps["image"]["status"]):
        mark_step_in_progress(cfg.run_dir, "image")
        _print("▸ image (Seedream)")
        prompt_str = build_seedream_prompt(plan_obj.seedream_scene, plan_obj.seedream_style)
        pred_id = generate_still(
            prompt=prompt_str, api_token=cfg.replicate_api_token, out_path=still_path,
        )
        mark_step_done(cfg.run_dir, "image", extra={"prediction_id": str(pred_id)})
        _print(f"✓ image committed (prediction_id={pred_id})")
    else:
        _print("✓ image (cached)")

    # Step 4: video
    video_dir = cfg.run_dir / "video"
    audio_chunks_dir = video_dir / "audio_chunks"
    if _should_run("video", cfg, m.steps["video"]["status"]):
        mark_step_in_progress(cfg.run_dir, "video")
        _print("▸ video (LTX × 6)")
        chunks = slice_audio_chunks(master_path, audio_chunks_dir)
        run_video_pipeline(
            run_id=cfg.run_id,
            still_path=still_path,
            audio_chunks=chunks,
            motion_prompts=plan_obj.motion_prompts,
            out_dir=video_dir,
            endpoint_id=cfg.ltx_endpoint, api_key=cfg.runpod_api_key,
            on_clip_done=lambda i, p: _print(f"  ✓ clip {i}"),
        )
        mark_step_done(cfg.run_dir, "video")
        _print("✓ video clips committed")
    else:
        _print("✓ video (cached)")

    # Step 5: loop_build
    concat_path = video_dir / "concat_42s.mp4"
    seamed_path = video_dir / "loop_seamed.mp4"
    if _should_run("loop_build", cfg, m.steps["loop_build"]["status"]):
        mark_step_in_progress(cfg.run_dir, "loop_build")
        _print("▸ loop_build (ffmpeg)")
        clips = sorted(video_dir.glob("clip_*.mp4"))
        if clips:
            concat_clips_with_xfades(clips, concat_path)
            add_loop_seam_fade(concat_path, seamed_path)
        mark_step_done(cfg.run_dir, "loop_build")
        _print("✓ loop_seamed.mp4 committed")
    else:
        _print("✓ loop_build (cached)")

    # Step 6: mux
    final_path = cfg.run_dir / "final.mp4"
    if _should_run("mux", cfg, m.steps["mux"]["status"]):
        mark_step_in_progress(cfg.run_dir, "mux")
        _print("▸ mux (ffmpeg)")
        final_assembly(
            seamed_path, master_path, final_path,
            target_sec=cfg.duration_sec, work_dir=cfg.run_dir / "_work",
        )
        mark_step_done(cfg.run_dir, "mux")
        _print(f"✓ final.mp4: {final_path}")
    else:
        _print(f"✓ mux (cached): {final_path}")

    return final_path

import json
from unittest.mock import patch, MagicMock

import pytest

from scripts.loopvid.orchestrator import run_orchestrator, OrchestratorConfig
from scripts.loopvid.manifest import load_manifest, new_manifest, save_manifest


@pytest.fixture
def cfg(tmp_path):
    return OrchestratorConfig(
        run_id="run-1",
        run_dir=tmp_path / "run-1",
        genre="ambient",
        mood="x",
        duration_sec=3600,
        ace_step_endpoint="ep-music",
        ltx_endpoint="ep-video",
        runpod_api_key="rp",
        openrouter_api_key="or",
        replicate_api_token="rep",
        only=None,
        skip=None,
        force=False,
        dry_run=False,
    )


def _fake_plan_dict():
    return {
        "genre": "ambient", "mood": "x",
        "music_palette": "Ambient drone, 60 BPM, no percussion, no vocals, "
                         "warm pads, plate reverb",
        "music_segment_descriptors": [{"phase": f"p{i}", "descriptors": f"d{i}"} for i in range(1, 12)],
        "music_bpm": 60,
        "seedream_scene": "Mountain ridge at golden hour",
        "seedream_style": "35mm film, soft golden light",
        "motion_prompts": [f"m{i}" for i in range(1, 7)],
        "motion_archetype": "mist",
        "image_archetype_key": "mountain_ridge_dusk",
    }


def _fake_plan_obj():
    from scripts.loopvid.plan_schema import validate_plan_dict
    return validate_plan_dict(_fake_plan_dict())


def test_dry_run_does_not_call_apis(cfg):
    new_cfg = OrchestratorConfig(**{**cfg.__dict__, "dry_run": True})
    with patch("scripts.loopvid.orchestrator.plan") as mp, \
         patch("scripts.loopvid.orchestrator.run_music_pipeline") as mm, \
         patch("scripts.loopvid.orchestrator.generate_still") as mi, \
         patch("scripts.loopvid.orchestrator.run_video_pipeline") as mv:
        run_orchestrator(new_cfg)
        mp.assert_not_called()
        mm.assert_not_called()
        mi.assert_not_called()
        mv.assert_not_called()


def test_orchestrator_creates_manifest(cfg):
    with patch("scripts.loopvid.orchestrator.run_preflight"), \
         patch("scripts.loopvid.orchestrator.plan") as mp, \
         patch("scripts.loopvid.orchestrator.run_music_pipeline"), \
         patch("scripts.loopvid.orchestrator.generate_still"), \
         patch("scripts.loopvid.orchestrator.run_video_pipeline"), \
         patch("scripts.loopvid.orchestrator.stitch_segments"), \
         patch("scripts.loopvid.orchestrator.slice_audio_chunks", return_value=[]), \
         patch("scripts.loopvid.orchestrator.concat_clips_with_xfades"), \
         patch("scripts.loopvid.orchestrator.add_loop_seam_fade"), \
         patch("scripts.loopvid.orchestrator.final_assembly"):
        mp.return_value = _fake_plan_obj()
        run_orchestrator(cfg)
        assert (cfg.run_dir / "manifest.json").exists()


def test_resume_skips_done_steps(cfg):
    """Pre-mark plan as done; the orchestrator should not call plan() again."""
    cfg.run_dir.mkdir(parents=True)
    m = new_manifest("run-1", {})
    m.steps["plan"]["status"] = "done"
    save_manifest(cfg.run_dir, m)
    (cfg.run_dir / "plan.json").write_text(json.dumps(_fake_plan_dict()))

    with patch("scripts.loopvid.orchestrator.run_preflight"), \
         patch("scripts.loopvid.orchestrator.plan") as mp, \
         patch("scripts.loopvid.orchestrator.run_music_pipeline"), \
         patch("scripts.loopvid.orchestrator.generate_still"), \
         patch("scripts.loopvid.orchestrator.run_video_pipeline"), \
         patch("scripts.loopvid.orchestrator.stitch_segments"), \
         patch("scripts.loopvid.orchestrator.slice_audio_chunks", return_value=[]), \
         patch("scripts.loopvid.orchestrator.concat_clips_with_xfades"), \
         patch("scripts.loopvid.orchestrator.add_loop_seam_fade"), \
         patch("scripts.loopvid.orchestrator.final_assembly"):
        run_orchestrator(cfg)
        mp.assert_not_called()


def test_only_runs_specified_steps(cfg):
    cfg.run_dir.mkdir(parents=True)
    m = new_manifest("run-1", {})
    for s in ("plan", "music", "image", "video", "loop_build"):
        m.steps[s]["status"] = "done"
    save_manifest(cfg.run_dir, m)
    (cfg.run_dir / "plan.json").write_text(json.dumps(_fake_plan_dict()))

    with patch("scripts.loopvid.orchestrator.run_preflight"), \
         patch("scripts.loopvid.orchestrator.final_assembly") as mf:
        new_cfg = OrchestratorConfig(**{**cfg.__dict__, "only": ("mux",)})
        run_orchestrator(new_cfg)
        mf.assert_called_once()

import json
from pathlib import Path
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


def test_orchestrator_config_has_new_optional_fields():
    cfg = OrchestratorConfig(
        run_id="r", run_dir=Path("/tmp/r"), genre="g", mood="", duration_sec=60,
        ace_step_endpoint="a", ltx_endpoint="l",
        runpod_api_key="", openrouter_api_key="", replicate_api_token="",
    )
    assert cfg.seedream_constraints is None
    assert cfg.ltx_negative is None
    assert cfg.extra_archetype_keys is None
    assert cfg.extra_motion_archetypes is None
    assert cfg.preset_plan_dict is None


def test_preset_plan_dict_skips_llm(tmp_path, monkeypatch):
    """When preset_plan_dict is set, orchestrator skips the LLM call and
    writes the dict directly to plan.json."""
    from scripts.loopvid import orchestrator as orch
    llm_called = {"yes": False}
    def fake_plan(**kwargs):
        llm_called["yes"] = True
        raise AssertionError("LLM should not be called when preset_plan_dict is set")
    monkeypatch.setattr(orch, "plan", fake_plan)
    # Stub everything after the plan step to short-circuit
    monkeypatch.setattr(orch, "run_preflight", lambda **k: None)
    monkeypatch.setattr(orch, "run_music_pipeline", lambda **k: [])
    monkeypatch.setattr(orch, "generate_still", lambda **k: "stub-pred")
    monkeypatch.setattr(orch, "run_video_pipeline", lambda **k: [])
    monkeypatch.setattr(orch, "slice_audio_chunks", lambda *a, **k: [])
    monkeypatch.setattr(orch, "concat_clips_with_xfades", lambda *a, **k: None)
    monkeypatch.setattr(orch, "add_loop_seam_fade", lambda *a, **k: None)
    monkeypatch.setattr(orch, "final_assembly", lambda *a, **k: None)

    preset = {
        "genre": "lofi", "mood": "",
        "music_palette": "lofi in the style of Nujabes, instrumental, 75 bpm",
        "music_segment_descriptors": [
            {"phase": "settle", "descriptors": "soft intro"} for _ in range(11)
        ],
        "music_bpm": 75,
        "seedream_scene": "scene", "seedream_style": "style",
        "motion_prompts": ["p"] * 6,
        "motion_archetype": "capybara_tea",
        "image_archetype_key": "capybara_tea",
    }
    cfg = OrchestratorConfig(
        run_id="r", run_dir=tmp_path / "r", genre="lofi", mood="", duration_sec=60,
        ace_step_endpoint="a", ltx_endpoint="l",
        runpod_api_key="k", openrouter_api_key="", replicate_api_token="t",
        skip=("music", "image", "video", "loop_build", "mux"),
        preset_plan_dict=preset,
        extra_archetype_keys={"capybara_tea"},
        extra_motion_archetypes={"capybara_tea"},
    )
    orch.run_orchestrator(cfg)
    assert llm_called["yes"] is False
    assert (tmp_path / "r" / "plan.json").exists()
    saved = json.loads((tmp_path / "r" / "plan.json").read_text())
    assert saved["music_palette"] == preset["music_palette"]


def test_orchestrator_config_passes_ace_step_preset(tmp_path):
    """cfg.ace_step_preset is set when caller passes it."""
    from scripts.loopvid.constants import ACE_STEP_TURBO_PRESET
    from scripts.loopvid.orchestrator import OrchestratorConfig

    cfg = OrchestratorConfig(
        run_id="test", run_dir=tmp_path, genre="lofi", mood="calm",
        duration_sec=60,
        ace_step_endpoint="ep", ltx_endpoint="lt",
        runpod_api_key="k", openrouter_api_key="", replicate_api_token="",
        ace_step_preset=ACE_STEP_TURBO_PRESET,
    )
    assert cfg.ace_step_preset is ACE_STEP_TURBO_PRESET


def test_orchestrator_config_default_ace_step_preset_is_none(tmp_path):
    """No ace_step_preset → None → music_pipeline falls back to ACE_STEP_PRESET."""
    from scripts.loopvid.orchestrator import OrchestratorConfig

    cfg = OrchestratorConfig(
        run_id="t", run_dir=tmp_path, genre="lofi", mood="calm",
        duration_sec=60,
        ace_step_endpoint="ep", ltx_endpoint="lt",
        runpod_api_key="k", openrouter_api_key="", replicate_api_token="",
    )
    assert cfg.ace_step_preset is None

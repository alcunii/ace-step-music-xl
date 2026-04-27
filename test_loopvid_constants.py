"""Snapshot tests for prompt constants — guards against accidental drift.

Update SHA1 values ONLY after a deliberate, audited change to a constant."""
import hashlib

from scripts.loopvid.constants import (
    ACE_STEP_PRESET,
    SEEDREAM_HARD_CONSTRAINTS,
    LTX_NEGATIVE_PROMPT,
    GENRE_ARCHETYPES,
    SEGMENT_DURATION_SEC,
    SEGMENT_COUNT_60MIN,
    CROSSFADE_SEC,
    CLIP_COUNT,
    CLIP_NUM_FRAMES,
    CLIP_FPS,
    CLIP_WIDTH,
    CLIP_HEIGHT,
    INTER_CLIP_XFADE_SEC,
    LOOP_SEAM_XFADE_SEC,
)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def test_ace_step_preset_matches_official():
    assert ACE_STEP_PRESET == {
        "inference_steps": 64,
        "guidance_scale": 8.0,
        "shift": 3.0,
        "use_adg": True,
        "cfg_interval_start": 0.0,
        "cfg_interval_end": 1.0,
        "infer_method": "ode",
    }


def test_seedream_hard_constraints_text_starts_with_clean_composition():
    assert SEEDREAM_HARD_CONSTRAINTS.startswith("Clean composition with absolutely no text")
    assert "no people" in SEEDREAM_HARD_CONSTRAINTS.lower()
    assert "16:9" in SEEDREAM_HARD_CONSTRAINTS


def test_ltx_negative_prompt_inherits_handler_default():
    assert "blurry" in LTX_NEGATIVE_PROMPT
    assert "watermark" in LTX_NEGATIVE_PROMPT
    assert "scene change" in LTX_NEGATIVE_PROMPT
    assert "morphing" in LTX_NEGATIVE_PROMPT


def test_genre_archetypes_complete_set():
    keys = set(GENRE_ARCHETYPES.keys())
    assert keys == {
        "rainy_window_desk",
        "mountain_ridge_dusk",
        "candle_dark_wood",
        "dim_bar_booth",
        "study_window_book",
        "observatory_dome",
    }
    for key, val in GENRE_ARCHETYPES.items():
        assert "visual" in val
        assert "anchored" in val
        assert "ambient_motion" in val


def test_video_clip_constants():
    assert CLIP_COUNT == 6
    assert CLIP_NUM_FRAMES == 169
    assert (CLIP_NUM_FRAMES - 1) % 8 == 0
    assert CLIP_FPS == 24
    assert CLIP_WIDTH == 1280
    assert CLIP_HEIGHT == 704
    assert CLIP_WIDTH % 32 == 0
    assert CLIP_HEIGHT % 32 == 0


def test_music_segment_constants_yield_at_least_60_minutes():
    total = SEGMENT_COUNT_60MIN * SEGMENT_DURATION_SEC - (SEGMENT_COUNT_60MIN - 1) * CROSSFADE_SEC
    assert total >= 3600, f"music plan only yields {total}s, need >=3600"


def test_loop_seam_constants_are_short():
    assert 0 < INTER_CLIP_XFADE_SEC <= 1.0
    assert 0 < LOOP_SEAM_XFADE_SEC <= 1.0

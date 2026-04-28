import re

import pytest

from scripts.loopvid.capybara_preset import (
    CAPYBARA_GENRE,
    CAPYBARA_MUSIC_BPM,
    CAPYBARA_MUSIC_PALETTE,
    CAPYBARA_SEGMENT_DESCRIPTORS,
    CAPYBARA_SEEDREAM_STYLE,
    CAPYBARA_SEEDREAM_CONSTRAINTS,
    CAPYBARA_LTX_NEGATIVE,
    CAPYBARA_SETTINGS,
    PRESET_SENTINEL_KEY,
)
from scripts.loopvid.constants import (
    MUSIC_PALETTE_MAX_CHARS, MUSIC_SEGMENT_DESCRIPTOR_MAX_CHARS,
    SEGMENT_COUNT_60MIN,
)


def test_genre_is_lofi():
    assert CAPYBARA_GENRE == "lofi"


def test_music_palette_within_cap():
    assert len(CAPYBARA_MUSIC_PALETTE) <= MUSIC_PALETTE_MAX_CHARS


def test_music_palette_anchors_to_nujabes():
    assert "Nujabes" in CAPYBARA_MUSIC_PALETTE
    assert "lofi" in CAPYBARA_MUSIC_PALETTE.lower()


def test_music_bpm_is_75():
    assert CAPYBARA_MUSIC_BPM == 75


def test_segment_descriptors_count_matches_segment_count():
    assert len(CAPYBARA_SEGMENT_DESCRIPTORS) == SEGMENT_COUNT_60MIN


def test_segment_descriptor_each_within_cap():
    for i, seg in enumerate(CAPYBARA_SEGMENT_DESCRIPTORS):
        assert "phase" in seg
        assert "descriptors" in seg
        assert len(seg["descriptors"]) <= MUSIC_SEGMENT_DESCRIPTOR_MAX_CHARS, \
            f"descriptor {i} too long: {seg['descriptors']!r}"


def test_segment_phase_arc_is_3_1_5_2():
    phases = [seg["phase"] for seg in CAPYBARA_SEGMENT_DESCRIPTORS]
    assert phases == (
        ["settle"] * 3 + ["hold"] * 1 + ["deepen"] * 5 + ["dissolve"] * 2
    )


def test_seedream_style_mentions_ghibli():
    assert "Ghibli" in CAPYBARA_SEEDREAM_STYLE
    assert "watercolor" in CAPYBARA_SEEDREAM_STYLE.lower()


def test_seedream_constraints_blocks_humans_only_not_animals():
    assert "no human" in CAPYBARA_SEEDREAM_CONSTRAINTS.lower()
    # Must NOT contain the generic "no faces" / "no eyes" that would block the capybara
    assert "no faces, no hands" not in CAPYBARA_SEEDREAM_CONSTRAINTS
    assert "no fingers." in CAPYBARA_SEEDREAM_CONSTRAINTS  # the human-narrowed form is OK


def test_seedream_constraints_drops_photographic_realism():
    # Anime style — must NOT inherit the photographic-only constraint
    assert "Photographic realism" not in CAPYBARA_SEEDREAM_CONSTRAINTS


def test_seedream_constraints_camera_locked_preserved():
    assert "fixed camera" in CAPYBARA_SEEDREAM_CONSTRAINTS.lower()
    assert "no pan" in CAPYBARA_SEEDREAM_CONSTRAINTS.lower()


def test_ltx_negative_blocks_humans_only_not_animals():
    assert "human face" in CAPYBARA_LTX_NEGATIVE
    # Must NOT block the capybara's own anatomy
    assert "fur detail" not in CAPYBARA_LTX_NEGATIVE
    assert "paws" not in CAPYBARA_LTX_NEGATIVE
    assert "animal legs" not in CAPYBARA_LTX_NEGATIVE


def test_ltx_negative_anime_failure_modes():
    assert "chibi style" in CAPYBARA_LTX_NEGATIVE
    assert "3D CGI" in CAPYBARA_LTX_NEGATIVE


def test_ltx_negative_blocks_multiple_capybaras():
    assert "multiple capybaras" in CAPYBARA_LTX_NEGATIVE


def test_settings_list_has_10_entries():
    assert len(CAPYBARA_SETTINGS) == 10


def test_settings_each_has_required_keys():
    for s in CAPYBARA_SETTINGS:
        for k in ("key", "scene", "lighting", "palette"):
            assert k in s, f"missing {k} in {s}"


def test_settings_keys_are_unique():
    keys = [s["key"] for s in CAPYBARA_SETTINGS]
    assert len(keys) == len(set(keys))


def test_settings_keys_are_snake_case():
    pat = re.compile(r"^[a-z][a-z0-9_]+$")
    for s in CAPYBARA_SETTINGS:
        assert pat.match(s["key"]), f"non-snake_case key: {s['key']}"


def test_settings_each_scene_mentions_capybara_and_tea():
    for s in CAPYBARA_SETTINGS:
        assert "capybara" in s["scene"].lower(), f"scene missing capybara: {s['key']}"
        assert "tea" in s["scene"].lower(), f"scene missing tea: {s['key']}"


def test_preset_sentinel_key():
    assert PRESET_SENTINEL_KEY == "capybara_tea"


from scripts.loopvid.capybara_preset import pick_setting, get_setting_by_key


def test_pick_setting_returns_a_setting_dict():
    s = pick_setting()
    assert s in CAPYBARA_SETTINGS


def test_pick_setting_deterministic_with_seed():
    a = pick_setting(seed=42)
    b = pick_setting(seed=42)
    assert a["key"] == b["key"]


def test_pick_setting_different_seeds_eventually_diverge():
    keys = {pick_setting(seed=i)["key"] for i in range(50)}
    # With 10 settings and 50 different seeds, we expect to cover most/all keys
    assert len(keys) >= 5


def test_get_setting_by_key_returns_match():
    s = get_setting_by_key("forest_hot_spring")
    assert s["key"] == "forest_hot_spring"
    assert "capybara" in s["scene"].lower()


def test_get_setting_by_key_unknown_raises_with_valid_keys_listed():
    with pytest.raises(ValueError) as exc:
        get_setting_by_key("not_a_real_key")
    msg = str(exc.value)
    assert "not_a_real_key" in msg
    # Error must list at least one known valid key for discoverability
    assert "forest_hot_spring" in msg


from scripts.loopvid.capybara_preset import build_motion_prompts


def test_build_motion_prompts_returns_six_strings():
    s = CAPYBARA_SETTINGS[0]
    prompts = build_motion_prompts(s)
    assert len(prompts) == 6
    assert all(isinstance(p, str) for p in prompts)


def test_build_motion_prompts_loop_seam_invariant():
    """Clip 1 and clip 6 must be string-equal so the loop seam is invisible."""
    for s in CAPYBARA_SETTINGS:
        prompts = build_motion_prompts(s)
        assert prompts[0] == prompts[5], f"seam broken for {s['key']}"


def test_build_motion_prompts_share_scene_prefix():
    s = CAPYBARA_SETTINGS[0]
    prompts = build_motion_prompts(s)
    for i, p in enumerate(prompts):
        assert s["scene"] in p, f"clip {i} missing scene"
        assert "Ghibli" in p, f"clip {i} missing style anchor"
        assert "Camera locked" in p, f"clip {i} missing camera-lock"


def test_build_motion_prompts_each_clip_distinct_except_seam():
    s = CAPYBARA_SETTINGS[0]
    prompts = build_motion_prompts(s)
    # clip 1 == clip 6 (seam); clips 2-5 must all be different from each other AND from rest
    assert len(set(prompts[1:5])) == 4, "clips 2-5 must each be distinct"
    assert prompts[0] not in prompts[1:5], "rest state must differ from active clips"


def test_build_motion_prompts_micro_motion_elements_present():
    s = CAPYBARA_SETTINGS[0]
    prompts = build_motion_prompts(s)
    # Each non-seam clip adds exactly one named micro-motion
    assert "leaf" in prompts[1].lower()
    assert "ear" in prompts[2].lower()
    assert "blink" in prompts[3].lower()
    assert "tail" in prompts[4].lower()

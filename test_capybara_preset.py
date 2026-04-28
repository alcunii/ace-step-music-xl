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

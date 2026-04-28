import pytest

from scripts.loopvid.plan_schema import Plan, validate_plan_dict, PlanSchemaError


VALID = {
    "genre": "lofi",
    "mood": "rainy night",
    "music_palette": "lofi in the style of Nujabes, instrumental, 70 bpm",
    "music_segment_descriptors": [
        {"phase": f"phase-{i}", "descriptors": f"desc-{i}"} for i in range(1, 12)
    ],
    "music_bpm": 70,
    "seedream_scene": "A wooden desk by a rainy window at dusk, a notebook and "
                      "a small lamp glowing softly",
    "seedream_style": "Shot on 35mm film, Kodak Portra 400, soft warm rim lighting",
    "motion_prompts": [
        "rain begins gently on the window glass, soft drops",
        "rain continues, slight intensification",
        "lamp flicker grows warmer",
        "rain peaks, steady rhythm",
        "rain softens",
        "rain returns to gentle drops, lamp settles",
    ],
    "motion_archetype": "rain",
    "image_archetype_key": "rainy_window_desk",
}


def test_validate_passes_on_valid():
    plan = validate_plan_dict(VALID)
    assert isinstance(plan, Plan)
    assert plan.genre == "lofi"
    assert len(plan.motion_prompts) == 6


def test_motion_prompts_must_be_exactly_6():
    bad = {**VALID, "motion_prompts": VALID["motion_prompts"][:5]}
    with pytest.raises(PlanSchemaError, match="motion_prompts"):
        validate_plan_dict(bad)


def test_music_segment_descriptors_must_be_exactly_11():
    bad = {**VALID, "music_segment_descriptors": VALID["music_segment_descriptors"][:10]}
    with pytest.raises(PlanSchemaError, match="11"):
        validate_plan_dict(bad)


def test_image_archetype_must_be_from_allowed_set():
    bad = {**VALID, "image_archetype_key": "made_up_archetype"}
    with pytest.raises(PlanSchemaError, match="archetype"):
        validate_plan_dict(bad)


def test_missing_required_field_raises():
    bad = {k: v for k, v in VALID.items() if k != "music_palette"}
    with pytest.raises(PlanSchemaError, match="music_palette"):
        validate_plan_dict(bad)


def test_motion_archetype_must_be_from_allowed_set():
    bad = {**VALID, "motion_archetype": "unicorn"}
    with pytest.raises(PlanSchemaError, match="motion_archetype"):
        validate_plan_dict(bad)


def test_music_palette_rejects_over_80_chars():
    # Deliberately bloated palette in the old multi-descriptor style.
    bloated = (
        "Dusty vinyl crackle, warm analog Rhodes chords, muted sub-bass, "
        "soft jazz-inflected guitar licks, bit-crushed drum machine"
    )
    assert len(bloated) > 80
    bad = {**VALID, "music_palette": bloated}
    with pytest.raises(PlanSchemaError, match="music_palette"):
        validate_plan_dict(bad)


def test_music_palette_accepts_single_anchor_at_cap():
    # Boundary: exactly 80 chars must pass.
    palette = "lofi in the style of Nujabes, instrumental, 70 bpm, mellow jazzy ambient feel"
    assert len(palette) <= 80
    ok = {**VALID, "music_palette": palette}
    plan = validate_plan_dict(ok)
    assert plan.music_palette == palette


def test_segment_descriptor_rejects_over_30_chars():
    bad_segs = list(VALID["music_segment_descriptors"])
    bad_segs[0] = {
        "phase": "settle",
        "descriptors": "expanding guitar layers, increased filter resonance, lush",
    }
    bad = {**VALID, "music_segment_descriptors": bad_segs}
    with pytest.raises(PlanSchemaError, match="descriptors"):
        validate_plan_dict(bad)


def test_extra_archetype_keys_accepts_custom_image_archetype():
    custom = {**VALID, "image_archetype_key": "capybara_tea"}
    plan = validate_plan_dict(custom, extra_archetype_keys={"capybara_tea"})
    assert plan.image_archetype_key == "capybara_tea"


def test_extra_archetype_keys_default_still_rejects_unknown():
    custom = {**VALID, "image_archetype_key": "capybara_tea"}
    with pytest.raises(PlanSchemaError, match="archetype"):
        validate_plan_dict(custom)


def test_extra_motion_archetypes_accepts_custom_motion_archetype():
    custom = {**VALID, "motion_archetype": "capybara_tea"}
    plan = validate_plan_dict(custom, extra_motion_archetypes={"capybara_tea"})
    assert plan.motion_archetype == "capybara_tea"


def test_extra_motion_archetypes_default_still_rejects_unknown():
    custom = {**VALID, "motion_archetype": "capybara_tea"}
    with pytest.raises(PlanSchemaError, match="motion_archetype"):
        validate_plan_dict(custom)

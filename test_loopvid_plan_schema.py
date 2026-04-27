import pytest

from scripts.loopvid.plan_schema import Plan, validate_plan_dict, PlanSchemaError


VALID = {
    "genre": "lofi",
    "mood": "rainy night",
    "music_palette": "Lofi instrumental, 70 BPM, vinyl crackle, jazz piano, "
                     "soft rim shot drums, warm tape saturation",
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

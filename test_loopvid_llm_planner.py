import json

import pytest
import responses

from scripts.loopvid.llm_planner import plan, OPENROUTER_URL


VALID_RESPONSE = {
    "choices": [{"message": {"content": json.dumps({
        "genre": "lofi",
        "mood": "rainy night",
        "music_palette": "Lofi instrumental, 70 BPM, vinyl crackle, jazz piano, "
                         "soft rim shot drums, warm tape saturation",
        "music_segment_descriptors": [
            {"phase": f"phase-{i}", "descriptors": f"desc-{i}"} for i in range(1, 12)
        ],
        "music_bpm": 70,
        "seedream_scene": "A wooden desk by a rainy window at dusk",
        "seedream_style": "Shot on 35mm film, Kodak Portra 400, soft warm rim",
        "motion_prompts": [f"motion {i}" for i in range(1, 7)],
        "motion_archetype": "rain",
        "image_archetype_key": "rainy_window_desk",
    })}}]
}


@responses.activate
def test_plan_returns_valid_plan_on_success():
    responses.add(responses.POST, OPENROUTER_URL, json=VALID_RESPONSE, status=200)
    p = plan(genre="lofi", mood="rainy night", api_key="k")
    assert p.genre == "lofi"
    assert len(p.motion_prompts) == 6


@responses.activate
def test_plan_uses_gemini_3_flash_preview_model():
    responses.add(responses.POST, OPENROUTER_URL, json=VALID_RESPONSE, status=200)
    plan(genre="lofi", mood="rainy night", api_key="k")
    body = json.loads(responses.calls[0].request.body)
    assert body["model"] == "google/gemini-3-flash-preview"


@responses.activate
def test_plan_retries_on_5xx():
    responses.add(responses.POST, OPENROUTER_URL, status=503)
    responses.add(responses.POST, OPENROUTER_URL, status=503)
    responses.add(responses.POST, OPENROUTER_URL, json=VALID_RESPONSE, status=200)
    p = plan(genre="lofi", mood="x", api_key="k", retry_sleep=0)
    assert p.genre == "lofi"
    assert len(responses.calls) == 3


@responses.activate
def test_plan_aborts_after_max_5xx_retries():
    for _ in range(4):
        responses.add(responses.POST, OPENROUTER_URL, status=503)
    with pytest.raises(RuntimeError, match="503"):
        plan(genre="lofi", mood="x", api_key="k", retry_sleep=0)


@responses.activate
def test_plan_retries_on_schema_validation_failure():
    bad = {
        "choices": [{"message": {"content": json.dumps({
            "genre": "lofi", "mood": "x",
            # MISSING required music_palette
        })}}]
    }
    responses.add(responses.POST, OPENROUTER_URL, json=bad, status=200)
    responses.add(responses.POST, OPENROUTER_URL, json=VALID_RESPONSE, status=200)
    p = plan(genre="lofi", mood="x", api_key="k", retry_sleep=0)
    assert p.genre == "lofi"
    assert len(responses.calls) == 2


def test_plan_raises_clear_error_when_api_key_missing():
    with pytest.raises(ValueError, match="api_key"):
        plan(genre="lofi", mood="x", api_key="")

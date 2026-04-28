import json

import pytest
import responses

from scripts.loopvid.image_pipeline import (
    build_seedream_prompt,
    generate_still,
    REPLICATE_PREDICTIONS_URL,
)
from scripts.loopvid.constants import SEEDREAM_HARD_CONSTRAINTS


def test_build_seedream_prompt_appends_constraints():
    p = build_seedream_prompt("a red barn", "shot on 35mm")
    assert "a red barn" in p
    assert "shot on 35mm" in p
    assert SEEDREAM_HARD_CONSTRAINTS in p


def test_build_seedream_prompt_orders_scene_style_constraints():
    p = build_seedream_prompt("scene", "style")
    assert p.index("scene") < p.index("style") < p.index(SEEDREAM_HARD_CONSTRAINTS)


@responses.activate
def test_generate_still_does_not_send_seed_or_negative_or_image_input(tmp_path):
    submit_url = REPLICATE_PREDICTIONS_URL.format(model="bytedance/seedream-4.5")
    responses.add(responses.POST, submit_url,
                  json={"id": "pred-1", "status": "starting"}, status=201)
    responses.add(responses.GET, "https://api.replicate.com/v1/predictions/pred-1",
                  json={"id": "pred-1", "status": "succeeded",
                        "output": "https://example.com/out.png"}, status=200)
    responses.add(responses.GET, "https://example.com/out.png",
                  body=b"\x89PNG\r\n\x1a\n" + b"x" * 100, status=200)

    out = tmp_path / "still.png"
    generate_still(prompt="x", api_token="t", out_path=out, poll_interval=0)

    sent = json.loads(responses.calls[0].request.body)["input"]
    assert "seed" not in sent
    assert "negative_prompt" not in sent
    assert "image_input" not in sent
    assert sent["aspect_ratio"] == "16:9"


@responses.activate
def test_generate_still_writes_atomic(tmp_path):
    submit_url = REPLICATE_PREDICTIONS_URL.format(model="bytedance/seedream-4.5")
    responses.add(responses.POST, submit_url,
                  json={"id": "p", "status": "starting"}, status=201)
    responses.add(responses.GET, "https://api.replicate.com/v1/predictions/p",
                  json={"id": "p", "status": "succeeded",
                        "output": "https://example.com/img.png"}, status=200)
    responses.add(responses.GET, "https://example.com/img.png",
                  body=b"\x89PNG\r\n\x1a\nFAKE", status=200)

    out = tmp_path / "still.png"
    pred_id = generate_still(prompt="x", api_token="t", out_path=out, poll_interval=0)
    assert out.exists()
    assert not (tmp_path / "still.png.tmp").exists()
    assert pred_id == "p"


@responses.activate
def test_generate_still_returns_prediction_id_for_manifest(tmp_path):
    submit_url = REPLICATE_PREDICTIONS_URL.format(model="bytedance/seedream-4.5")
    responses.add(responses.POST, submit_url,
                  json={"id": "abc-123", "status": "starting"}, status=201)
    responses.add(responses.GET, "https://api.replicate.com/v1/predictions/abc-123",
                  json={"status": "succeeded", "output": "https://example.com/out.png"}, status=200)
    responses.add(responses.GET, "https://example.com/out.png",
                  body=b"\x89PNG\r\n", status=200)

    pred_id = generate_still(prompt="x", api_token="t", out_path=tmp_path / "s.png", poll_interval=0)
    assert pred_id == "abc-123"


@responses.activate
def test_generate_still_raises_on_failed(tmp_path):
    submit_url = REPLICATE_PREDICTIONS_URL.format(model="bytedance/seedream-4.5")
    responses.add(responses.POST, submit_url,
                  json={"id": "p", "status": "starting"}, status=201)
    responses.add(responses.GET, "https://api.replicate.com/v1/predictions/p",
                  json={"status": "failed", "error": "OOM"}, status=200)

    with pytest.raises(RuntimeError, match="OOM"):
        generate_still(prompt="x", api_token="t", out_path=tmp_path / "s.png", poll_interval=0)


def test_build_seedream_prompt_uses_custom_constraints_when_given():
    custom = "Studio Ghibli watercolor anime style, no chibi exaggerations"
    p = build_seedream_prompt("scene", "style", constraints=custom)
    assert "scene" in p
    assert "style" in p
    assert custom in p
    assert SEEDREAM_HARD_CONSTRAINTS not in p


def test_build_seedream_prompt_default_constraints_unchanged():
    p = build_seedream_prompt("scene", "style")
    assert SEEDREAM_HARD_CONSTRAINTS in p

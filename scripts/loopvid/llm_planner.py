"""OpenRouter Gemini 3 Flash planner with structured output + retry."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests

from scripts.loopvid.constants import GENRE_ARCHETYPES, SEGMENT_COUNT_60MIN, CLIP_COUNT
from scripts.loopvid.plan_schema import Plan, PlanSchemaError, validate_plan_dict

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3-flash-preview"
TEMPERATURE = 0.7
MAX_5XX_RETRIES = 3
MAX_SCHEMA_RETRIES = 3
EXP_BACKOFF_BASE_SEC = 1


def _build_system_prompt() -> str:
    archetypes = ", ".join(sorted(GENRE_ARCHETYPES.keys()))
    return (
        "You are a planner for a 1-hour looping ambient/instrumental music-video "
        "generator. Given (genre, mood), produce a plan as JSON matching the schema.\n\n"
        "CONSTRAINTS YOU MUST OBEY:\n"
        f"- Music: {SEGMENT_COUNT_60MIN} segments, each ~360s, sharing a locked sonic "
        "palette (10-15 descriptors, ~400 chars). Each segment adds 2-3 phase descriptors "
        "implementing a breathing arc: 3 settle → 1 hold → 5 deepen-and-release → 2 dissolve.\n"
        f"- Image: pick the closest archetype from {{{archetypes}}}. Customize the specific "
        "objects, lighting, color grade. Camera must be locked.\n"
        f"- Motion: {CLIP_COUNT} prompts, all sharing the same base motion (single ambient "
        "source from the still). Clip 1 and clip 6 must depict the rest-state (so the loop "
        "seam is invisible). Clips 2-5 escalate then descalate amplitude.\n\n"
        "YOU MUST NOT:\n"
        "- Mention text, signs, faces, hands, fingers, mirrors, scene cuts, fast motion in "
        "any prompt (the constants module appends explicit constraints later).\n"
        "- Invent new genre archetypes; pick from the list.\n"
        "- Vary the still across clips (one still, six motion variations)."
    )


def _build_response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "genre": {"type": "string"},
            "mood": {"type": "string"},
            "music_palette": {"type": "string"},
            "music_segment_descriptors": {
                "type": "array", "minItems": SEGMENT_COUNT_60MIN, "maxItems": SEGMENT_COUNT_60MIN,
                "items": {
                    "type": "object",
                    "properties": {"phase": {"type": "string"}, "descriptors": {"type": "string"}},
                    "required": ["phase", "descriptors"],
                },
            },
            "music_bpm": {"type": "integer"},
            "seedream_scene": {"type": "string"},
            "seedream_style": {"type": "string"},
            "motion_prompts": {
                "type": "array", "minItems": CLIP_COUNT, "maxItems": CLIP_COUNT,
                "items": {"type": "string"},
            },
            "motion_archetype": {"type": "string"},
            "image_archetype_key": {"type": "string"},
        },
        "required": [
            "genre", "mood", "music_palette", "music_segment_descriptors", "music_bpm",
            "seedream_scene", "seedream_style", "motion_prompts", "motion_archetype",
            "image_archetype_key",
        ],
    }


def _post(api_key: str, messages: list, retry_sleep: int) -> dict:
    """POST to OpenRouter with 5xx retry + exponential backoff."""
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "plan", "schema": _build_response_schema(), "strict": True},
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(1, MAX_5XX_RETRIES + 1):
        resp = requests.post(OPENROUTER_URL, json=body, headers=headers, timeout=120)
        if 500 <= resp.status_code < 600:
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if attempt < MAX_5XX_RETRIES and retry_sleep > 0:
                time.sleep(retry_sleep * (4 ** (attempt - 1)))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"OpenRouter failed after {MAX_5XX_RETRIES} attempts: {last_err}")


def plan(
    *, genre: str, mood: str, api_key: str,
    raw_response_path: Optional[str] = None,
    retry_sleep: int = EXP_BACKOFF_BASE_SEC,
) -> Plan:
    """Call OpenRouter Gemini 3 Flash with structured output. Retries on 5xx
    and on schema-validation failure (with the validation error appended to
    the system prompt for the next attempt)."""
    if not api_key:
        raise ValueError("api_key is required (set OPENROUTER_API_KEY)")

    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": f"genre: {genre}\nmood: {mood}"},
    ]
    last_validation_err = None
    for attempt in range(1, MAX_SCHEMA_RETRIES + 1):
        if last_validation_err:
            messages = messages + [
                {"role": "user", "content": (
                    "Your previous response failed schema validation: "
                    f"{last_validation_err}. Respond again with valid JSON."
                )}
            ]
        body = _post(api_key, messages, retry_sleep=retry_sleep)
        content = body["choices"][0]["message"]["content"]
        if raw_response_path:
            Path(raw_response_path).write_text(content)
        try:
            d = json.loads(content)
            return validate_plan_dict(d)
        except (json.JSONDecodeError, PlanSchemaError) as e:
            last_validation_err = str(e)
            if attempt >= MAX_SCHEMA_RETRIES:
                raise RuntimeError(
                    f"Plan schema validation failed after {MAX_SCHEMA_RETRIES} attempts: {e}"
                )
    raise RuntimeError("unreachable")

"""Plan dataclass + validator for the LLM Planner output."""
from __future__ import annotations

from dataclasses import dataclass

from scripts.loopvid.constants import (
    GENRE_ARCHETYPES, SEGMENT_COUNT_60MIN, CLIP_COUNT,
    MUSIC_PALETTE_MAX_CHARS, MUSIC_SEGMENT_DESCRIPTOR_MAX_CHARS,
)

ALLOWED_MOTION_ARCHETYPES = {"rain", "candle", "mist", "smoke", "dust", "snow"}


class PlanSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Plan:
    genre: str
    mood: str
    music_palette: str
    music_segment_descriptors: list
    music_bpm: int
    seedream_scene: str
    seedream_style: str
    motion_prompts: list
    motion_archetype: str
    image_archetype_key: str


REQUIRED_FIELDS = {
    "genre": str,
    "mood": str,
    "music_palette": str,
    "music_segment_descriptors": list,
    "music_bpm": int,
    "seedream_scene": str,
    "seedream_style": str,
    "motion_prompts": list,
    "motion_archetype": str,
    "image_archetype_key": str,
}


def validate_plan_dict(
    d: dict,
    *,
    extra_archetype_keys: set[str] | None = None,
    extra_motion_archetypes: set[str] | None = None,
) -> Plan:
    for name, expected_type in REQUIRED_FIELDS.items():
        if name not in d:
            raise PlanSchemaError(f"missing required field: {name}")
        if not isinstance(d[name], expected_type):
            raise PlanSchemaError(
                f"field {name} must be {expected_type.__name__}, got {type(d[name]).__name__}"
            )

    if len(d["music_palette"]) > MUSIC_PALETTE_MAX_CHARS:
        raise PlanSchemaError(
            f"music_palette must be ≤ {MUSIC_PALETTE_MAX_CHARS} chars "
            f"(got {len(d['music_palette'])}). Single-anchor format required: "
            f"'{{genre}} in the style of {{one anchor}}, instrumental, {{bpm}} bpm'"
        )

    if len(d["music_segment_descriptors"]) != SEGMENT_COUNT_60MIN:
        raise PlanSchemaError(
            f"music_segment_descriptors must have exactly {SEGMENT_COUNT_60MIN} entries, "
            f"got {len(d['music_segment_descriptors'])}"
        )
    for i, seg in enumerate(d["music_segment_descriptors"]):
        if not isinstance(seg, dict) or "phase" not in seg or "descriptors" not in seg:
            raise PlanSchemaError(
                f"music_segment_descriptors[{i}] must have 'phase' and 'descriptors' keys"
            )
        if len(seg["descriptors"]) > MUSIC_SEGMENT_DESCRIPTOR_MAX_CHARS:
            raise PlanSchemaError(
                f"music_segment_descriptors[{i}].descriptors must be "
                f"≤ {MUSIC_SEGMENT_DESCRIPTOR_MAX_CHARS} chars (got {len(seg['descriptors'])}). "
                f"Use one short phase phrase, not a stack of adjectives."
            )

    if len(d["motion_prompts"]) != CLIP_COUNT:
        raise PlanSchemaError(
            f"motion_prompts must have exactly {CLIP_COUNT} entries, "
            f"got {len(d['motion_prompts'])}"
        )

    allowed_image = set(GENRE_ARCHETYPES.keys()) | (extra_archetype_keys or set())
    if d["image_archetype_key"] not in allowed_image:
        raise PlanSchemaError(
            f"image_archetype_key '{d['image_archetype_key']}' not in allowed set "
            f"{sorted(allowed_image)}"
        )

    allowed_motion = ALLOWED_MOTION_ARCHETYPES | (extra_motion_archetypes or set())
    if d["motion_archetype"] not in allowed_motion:
        raise PlanSchemaError(
            f"motion_archetype '{d['motion_archetype']}' not in allowed set "
            f"{sorted(allowed_motion)}"
        )

    return Plan(
        genre=d["genre"],
        mood=d["mood"],
        music_palette=d["music_palette"],
        music_segment_descriptors=list(d["music_segment_descriptors"]),
        music_bpm=int(d["music_bpm"]),
        seedream_scene=d["seedream_scene"],
        seedream_style=d["seedream_style"],
        motion_prompts=list(d["motion_prompts"]),
        motion_archetype=d["motion_archetype"],
        image_archetype_key=d["image_archetype_key"],
    )

"""Plan dataclass + validator for the LLM Planner output."""
from __future__ import annotations

from dataclasses import dataclass

from scripts.loopvid.constants import GENRE_ARCHETYPES, SEGMENT_COUNT_60MIN, CLIP_COUNT

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


def validate_plan_dict(d: dict) -> Plan:
    for name, expected_type in REQUIRED_FIELDS.items():
        if name not in d:
            raise PlanSchemaError(f"missing required field: {name}")
        if not isinstance(d[name], expected_type):
            raise PlanSchemaError(
                f"field {name} must be {expected_type.__name__}, got {type(d[name]).__name__}"
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

    if len(d["motion_prompts"]) != CLIP_COUNT:
        raise PlanSchemaError(
            f"motion_prompts must have exactly {CLIP_COUNT} entries, "
            f"got {len(d['motion_prompts'])}"
        )

    if d["image_archetype_key"] not in GENRE_ARCHETYPES:
        raise PlanSchemaError(
            f"image_archetype_key '{d['image_archetype_key']}' not in allowed set "
            f"{sorted(GENRE_ARCHETYPES.keys())}"
        )

    if d["motion_archetype"] not in ALLOWED_MOTION_ARCHETYPES:
        raise PlanSchemaError(
            f"motion_archetype '{d['motion_archetype']}' not in allowed set "
            f"{sorted(ALLOWED_MOTION_ARCHETYPES)}"
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

"""Capybara + tea loop preset — locked constants for scripts/capybara_tea_loop.py.

All creative content is hardcoded or template-substituted. No LLM in the loop.
Variability lives in the curated CAPYBARA_SETTINGS list (random pick per run)
plus Seedream's per-run sampling.

Spec: docs/superpowers/specs/2026-04-28-capybara-tea-loop-design.md
"""
from __future__ import annotations

# ── Identity ────────────────────────────────────────────────────────────
PRESET_SENTINEL_KEY: str = "capybara_tea"

# ── Music plan (locked) ─────────────────────────────────────────────────
CAPYBARA_GENRE: str = "lofi"
CAPYBARA_MUSIC_BPM: int = 75
CAPYBARA_MUSIC_PALETTE: str = "lofi in the style of Nujabes, instrumental, 75 bpm"

# Breathing arc: 3 settle → 1 hold → 5 deepen → 2 dissolve.
# Each descriptors string ≤ MUSIC_SEGMENT_DESCRIPTOR_MAX_CHARS (30).
CAPYBARA_SEGMENT_DESCRIPTORS: list[dict] = [
    {"phase": "settle",   "descriptors": "soft intro"},
    {"phase": "settle",   "descriptors": "warm settle"},
    {"phase": "settle",   "descriptors": "gentle bloom"},
    {"phase": "hold",     "descriptors": "steady core"},
    {"phase": "deepen",   "descriptors": "warm core"},
    {"phase": "deepen",   "descriptors": "deeper warmth"},
    {"phase": "deepen",   "descriptors": "patient pulse"},
    {"phase": "deepen",   "descriptors": "lifted core"},
    {"phase": "deepen",   "descriptors": "gentle release"},
    {"phase": "dissolve", "descriptors": "slow fade"},
    {"phase": "dissolve", "descriptors": "soft outro"},
]

# ── Seedream style + constraints (positive-form, no negative_prompt field) ──
CAPYBARA_SEEDREAM_STYLE: str = (
    "Studio Ghibli soft watercolor anime style, hand-painted background, "
    "gentle painterly brushwork, warm cinematic atmosphere, cozy storybook feel"
)

# Replaces SEEDREAM_HARD_CONSTRAINTS for capybara runs only. Drops:
#   "Photographic realism, no AI-style artifacts, no oversaturation,
#    no fake bokeh, no HDR look" (incompatible with watercolor anime)
#   Generic "no faces / no hands / no fingers" → narrowed to "no human ..."
#    so the capybara face/eyes are allowed to render.
CAPYBARA_SEEDREAM_CONSTRAINTS: str = (
    "Clean composition with absolutely no text, letters, numbers, words, "
    "captions, watermarks, signage, signs, neon signs, screen text, book "
    "text, handwriting, signatures, logos, brand marks, or any visible "
    "writing of any kind. "
    "No human figures, no human faces, no human hands, no fingers. "
    "Single focal capybara, ONE capybara only, no duplicates. Uncluttered "
    "painterly background, no mirrors, no reflective glass, no transparent "
    "objects. Static medium-wide shot, fixed camera, no pan, no zoom, no "
    "parallax. 16:9 cinematic widescreen. "
    "Avoid chibi exaggerations, avoid moe big-eye style, avoid 3D CGI, "
    "avoid glossy plastic shading, avoid cel-shaded toy look, avoid manga "
    "panel borders, avoid speech bubbles, avoid sticker-art look."
)

# ── LTX negative (real negative_prompt field) ───────────────────────────
# Replaces LTX_NEGATIVE_PROMPT for capybara runs only. Drops:
#   "hands, fingers, face, faces, eyes, mouth, lips" (generic) → narrowed
#    to "human hands, human fingers, human face, human faces"
#   "animal legs, paws, wings, fur detail" — these define the subject;
#    suppressing them is actively destructive in LTX-2.3's text encoder.
CAPYBARA_LTX_NEGATIVE: str = (
    "blurry, low quality, still frame, frames, watermark, overlay, "
    "titles, has blurbox, has subtitles, "
    "text, letters, numbers, words, captions, logo, "
    "signage, signs, neon text, screen text, on-screen text, "
    "human hands, human fingers, human face, human faces, "
    "fast motion, sudden movement, rapid motion, camera shake, "
    "camera pan, camera zoom, dolly, tracking shot, handheld, "
    "hard cut, scene change, scene transition, jump cut, "
    "reflections, mirror reflections, glass refraction, "
    "multiple capybaras, multiple animated subjects, crowd, "
    "splashing water, pouring liquid, explosion, fireworks, sparks, "
    "frame stutter, ghosting, motion smear, double exposure, "
    "morphing, warping, melting, glitch, "
    "chibi style, big anime eyes, moe style, 3D CGI, plastic shading, "
    "cel-shaded toy look"
)

# ── Curated settings — 10 entries, varied by run ────────────────────────
# Brand invariant: ONE round capybara + ONE steaming teacup + Ghibli setting.
CAPYBARA_SETTINGS: list[dict] = [
    {
        "key": "forest_hot_spring",
        "scene": "a single round capybara soaking in a small stone hot spring at the edge of a misty Ghibli-style cedar forest, a flat wooden tray with a steaming teacup beside it",
        "lighting": "soft early-morning sunbeams filtering through tall cedars, warm golden rays through mist",
        "palette": "deep forest green, mossy stone gray, warm steam-cream, hint of sunrise pink",
    },
    {
        "key": "library_window_nook",
        "scene": "a single round capybara curled on a cushion in a cozy wooden reading nook, low side table with a steaming teacup and one small stacked book",
        "lighting": "warm afternoon sunlight through a tall arched window, dust motes drifting in the beams",
        "palette": "honey wood, dusty cream, faded burgundy book spines, soft amber",
    },
    {
        "key": "autumn_garden_lantern",
        "scene": "a single round capybara seated on a low wooden engawa overlooking an autumn Japanese garden, a paper lantern hanging above, a steaming teacup on a small black tray",
        "lighting": "late-afternoon golden hour, soft horizontal light",
        "palette": "burnt orange, deep maple red, lantern amber, mossy stone",
    },
    {
        "key": "snowy_cabin_porch",
        "scene": "a single round capybara wrapped in a knitted blanket on a snowy wooden cabin porch, a steaming teacup on a low side table, gentle snowfall in the distance",
        "lighting": "overcast diffuse winter daylight, cool blue tones with warm porch-lamp glow",
        "palette": "snow white, slate gray, warm wool brown, soft amber",
    },
    {
        "key": "pastel_cafe_window",
        "scene": "a single round capybara seated in a pastel-pink café booth, low marble table with a steaming teacup and a single mochi on a saucer",
        "lighting": "soft midday window light, gentle backlight",
        "palette": "pastel pink, cream white, mint green accent, soft tan wood",
    },
    {
        "key": "moonlit_balcony_bamboo",
        "scene": "a single round capybara seated on a bamboo balcony overlooking a still pond at night, steaming teacup on a wooden block, distant fireflies",
        "lighting": "soft full-moon glow with warm lantern fill",
        "palette": "deep navy, silver moonlight, lantern gold, cool bamboo green",
    },
    {
        "key": "sakura_riverbank",
        "scene": "a single round capybara on a flat stone at the edge of a slow river under a sakura tree in full bloom, steaming teacup on the stone beside it",
        "lighting": "soft pink-tinted afternoon light filtered through cherry blossoms",
        "palette": "cherry pink, river blue-gray, moss green, soft cream",
    },
    {
        "key": "rainy_engawa",
        "scene": "a single round capybara on a wooden engawa watching gentle rain fall on a moss garden, steaming teacup on the polished wood beside it",
        "lighting": "overcast soft daylight, cool gray-green",
        "palette": "wet stone gray, moss green, warm wood amber, soft pearl",
    },
    {
        "key": "kotatsu_winter_room",
        "scene": "a single round capybara cozy under a kotatsu blanket in a tatami room, steaming teacup on the kotatsu surface, a single mandarin orange beside it",
        "lighting": "warm low lamp light, soft shadows",
        "palette": "kotatsu red, tatami straw, warm orange, soft amber",
    },
    {
        "key": "summer_meadow_sunset",
        "scene": "a single round capybara seated in a tall grass meadow at sunset, steaming teacup on a flat stone, distant rolling Ghibli-style hills",
        "lighting": "low golden sunset, soft horizontal warmth",
        "palette": "tall-grass green-gold, sunset orange, soft sky lavender, warm cream",
    },
]

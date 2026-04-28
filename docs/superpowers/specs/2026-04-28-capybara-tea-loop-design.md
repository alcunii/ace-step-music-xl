# Capybara + Tea Loop — Design

**Date:** 2026-04-28
**Owner:** content.factory.global@gmail.com
**Status:** Approved (brainstorming complete; awaiting implementation plan)
**Predecessor:** `2026-04-27-loop-music-video-design.md` (the general LLM-driven loopvid pipeline this forks from)

---

## 1. Goal

Produce a 60-minute looping music video with a fixed creative concept — **a Studio-Ghibli-style cute capybara enjoying tea** — where the visual setting varies per generation but the music, motion register, and aesthetic stay locked. The output is suited for "cozy ambient loop" YouTube channels: the brand is recognizable across uploads, but every video feels fresh.

The existing `scripts/loop_music_video.py` + `scripts/loopvid/llm_planner.py` pipeline takes `(genre, mood)` and uses Gemini 3 Flash to choose archetypes, anchors, and motion prompts. This design **forks** that pipeline into a parallel script that locks every creative choice except the still scene.

## 2. Constraints inherited unchanged

These are reused verbatim from the existing loopvid stack:

| Subsystem | Constraint | Source |
|---|---|---|
| Music | 11 × 360 s segments, 30 s crossfade, ACE-Step XL-base preset, ≤80-char palette, ≤30-char per-segment descriptor | `scripts/loopvid/constants.py` |
| Music | Single-artist anchor for the whole hour | locked palette pattern |
| Video | 6 × 7.04 s clips at 1280×704, 24 fps, 169 frames | `scripts/loopvid/constants.py` |
| Video | Camera always locked, no pan/zoom, clip 1 ≡ clip 6 for invisible loop seam | preserved here |
| Image | Seedream 4.5 (Replicate), 16:9, no `negative_prompt` field — constraints inlined as positive-form text | verified in `image_pipeline.py:38` |
| LTX | Real `negative_prompt` field accepted by `/root/ltx23-pro6000/handler.py` | verified in `handler.py:7,114-115` |

## 3. Architecture

### 3.1 New files

```
scripts/capybara_tea_loop.py            # entry point (forked from loop_music_video.py)
scripts/loopvid/capybara_preset.py      # locked constants + Plan factory
test_capybara_preset.py                 # unit + plan-validation tests
test_capybara_5min_smoke.py             # opt-in live E2E smoke test
```

### 3.2 Reused unchanged

```
scripts/loopvid/orchestrator.py
scripts/loopvid/image_pipeline.py    (one tiny additive change: see §3.4)
scripts/loopvid/music_pipeline.py
scripts/loopvid/video_pipeline.py
scripts/loopvid/loop_build.py
scripts/loopvid/mux.py
scripts/loopvid/runpod_client.py
scripts/loopvid/preflight.py
scripts/loopvid/manifest.py
scripts/loopvid/cost.py
scripts/loopvid/rollback.py
scripts/loopvid/plan_schema.py
scripts/loopvid/constants.py         (untouched — original SEEDREAM_HARD_CONSTRAINTS, LTX_NEGATIVE_PROMPT, GENRE_ANCHORS, GENRE_ARCHETYPES still in use by loop_music_video.py)
```

### 3.3 No LLM in this pipeline

`capybara_preset.py` is a **pure builder**: every field of `Plan` is either hardcoded or template-substituted from a curated setting. We deliberately skip `llm_planner.py` for this fork. Benefits:

- Eliminates the Gemini API call (~$0.01/run)
- Removes the schema-retry failure mode
- Output is deterministic given `--seed`
- Variability lives where we want it: (10 curated settings) × (Seedream's per-run sampling)

If micro-detail variability becomes desirable later (varying time-of-day, what's on the tray, capybara pose), we can add a tiny LLM call gated behind a `--llm-vary` flag without touching anything else. Out of scope for v1.

### 3.4 Two small additive wiring changes

**(a) `image_pipeline.build_seedream_prompt`** currently hard-imports `SEEDREAM_HARD_CONSTRAINTS`. Add an optional `constraints` keyword:

```python
def build_seedream_prompt(scene: str, style: str, constraints: str = SEEDREAM_HARD_CONSTRAINTS) -> str:
    return f"{scene}. {style}. {constraints}"
```

Backwards-compatible (default keeps existing behavior). Capybara preset passes `constraints=CAPYBARA_SEEDREAM_CONSTRAINTS`.

**(b) `plan_schema.validate_plan_dict`** currently validates `image_archetype_key` against `GENRE_ARCHETYPES.keys()` (closed set of 6 photographic archetypes) and `motion_archetype` against `ALLOWED_MOTION_ARCHETYPES = {"rain","candle","mist","smoke","dust","snow"}`. The capybara fork doesn't fit any of these. Add two optional kwargs:

```python
def validate_plan_dict(
    d: dict,
    *,
    extra_archetype_keys: set[str] | None = None,
    extra_motion_archetypes: set[str] | None = None,
) -> Plan:
    ...
    allowed_image = set(GENRE_ARCHETYPES.keys()) | (extra_archetype_keys or set())
    allowed_motion = ALLOWED_MOTION_ARCHETYPES | (extra_motion_archetypes or set())
    if d["image_archetype_key"] not in allowed_image: ...
    if d["motion_archetype"] not in allowed_motion: ...
```

Capybara preset calls `validate_plan_dict(d, extra_archetype_keys={"capybara_tea"}, extra_motion_archetypes={"capybara_tea"})`. The LLM-driven path (`llm_planner.py`) calls `validate_plan_dict(d)` unchanged — it never sees `"capybara_tea"` so the LLM can't accidentally pick it. Confirmed safe: `orchestrator.py:115-116` only writes these keys into the manifest as metadata; no downstream code looks them up against `GENRE_ARCHETYPES`.

### 3.5 Data flow

```
capybara_tea_loop.py
  ├── parse CLI: --setting <key> | --duration | --seed | --output-dir
  ├── pick setting:
  │     - if --setting given, look up by key (raise on miss with valid-key list)
  │     - else random.Random(seed).choice(CAPYBARA_SETTINGS)
  ├── capybara_preset.build_plan(setting) → validated Plan
  │     ├── music_palette = "lofi in the style of Nujabes, instrumental, 75 bpm"
  │     ├── music_segment_descriptors = locked 11-element 3-1-5-2 arc
  │     ├── seedream_scene = f"{setting['scene']}. {setting['lighting']}. {setting['palette']}"
  │     ├── seedream_style = CAPYBARA_SEEDREAM_STYLE          (Ghibli style anchor only)
  │     ├── motion_prompts = build_motion_prompts(setting)   → 6 strings, clip[0]==clip[5]
  │     ├── motion_archetype = "capybara_tea"  (sentinel for manifest metadata)
  │     ├── image_archetype_key = "capybara_tea"  (sentinel for manifest metadata)
  │     └── validated via validate_plan_dict(d, extra_archetype_keys={"capybara_tea"},
  │                                             extra_motion_archetypes={"capybara_tea"})
  └── orchestrator.run(plan, seedream_constraints=CAPYBARA_SEEDREAM_CONSTRAINTS,
                            ltx_negative=CAPYBARA_LTX_NEGATIVE)
        ├── image_pipeline (Seedream 4.5; constraints kw passed through)
        ├── music_pipeline (ACE-Step, 11×360s, locked Nujabes palette)
        ├── video_pipeline (LTX-2.3, 6 clips, ltx_negative passed through)
        └── loop_build → mux → final 60-min .mp4
```

The orchestrator gains two optional keyword parameters (`seedream_constraints`, `ltx_negative`) that default to the existing constants — same backwards-compat pattern as §3.4.

## 4. Locked content

### 4.1 Curated settings list

`CAPYBARA_SETTINGS` — 10 entries. Each has `key`, `scene`, `lighting`, `palette`. Brand invariant across all 10: **ONE round capybara + ONE steaming teacup + Ghibli-style natural setting, single focal subject, camera-locked.**

| Key | Setting summary |
|---|---|
| `forest_hot_spring` | misty Ghibli cedar forest, stone hot spring, tea on a wooden tray |
| `library_window_nook` | cozy wooden reading nook, low table, one stacked book |
| `autumn_garden_lantern` | engawa overlooking autumn Japanese garden, paper lantern overhead |
| `snowy_cabin_porch` | knitted blanket on a snowy porch, gentle snowfall in the distance |
| `pastel_cafe_window` | pastel-pink café booth, marble table, mochi on a saucer |
| `moonlit_balcony_bamboo` | bamboo balcony over a still pond, distant fireflies, lantern fill |
| `sakura_riverbank` | flat stone by a slow river under cherry blossoms in full bloom |
| `rainy_engawa` | engawa watching gentle rain on a moss garden |
| `kotatsu_winter_room` | tatami room, kotatsu blanket, mandarin orange beside the cup |
| `summer_meadow_sunset` | tall grass meadow at sunset, distant Ghibli rolling hills |

(Full strings for `scene` / `lighting` / `palette` are baked into `capybara_preset.py` per §6.)

### 4.2 Locked music plan

```python
CAPYBARA_GENRE         = "lofi"
CAPYBARA_MUSIC_BPM     = 75
CAPYBARA_MUSIC_PALETTE = "lofi in the style of Nujabes, instrumental, 75 bpm"  # 51 chars

CAPYBARA_SEGMENT_DESCRIPTORS = [
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
```

Phase distribution: 3 settle → 1 hold → 5 deepen → 2 dissolve (matches the existing planner's instruction to the LLM, just baked in).

### 4.3 Seedream prompt assembly (positive-form, no negative_prompt)

We split the locked text into two constants so they slot into the existing `build_seedream_prompt(scene, style, constraints)` signature cleanly:

- `CAPYBARA_SEEDREAM_STYLE` — the Ghibli style anchor only (passed as `style`)
- `CAPYBARA_SEEDREAM_CONSTRAINTS` — all guardrails (passed as `constraints`)

The full rendered prompt fed to Seedream 4.5 is:

```
{setting.scene}. {setting.lighting}. {setting.palette}. {CAPYBARA_SEEDREAM_STYLE}. {CAPYBARA_SEEDREAM_CONSTRAINTS}
```

**`CAPYBARA_SEEDREAM_STYLE`** = "Studio Ghibli soft watercolor anime style, hand-painted background, gentle painterly brushwork, warm cinematic atmosphere, cozy storybook feel"

**`CAPYBARA_SEEDREAM_CONSTRAINTS`** contains:

- **Universal blocks** (kept verbatim from `SEEDREAM_HARD_CONSTRAINTS`): no text/letters/numbers/captions/watermarks/signage/logos.
- **Humans blocked** (capybara is the focal subject): no human figures, no human faces, no human hands, no human fingers.
- **Composition + camera lock:** single focal capybara, ONE capybara only, no duplicates; uncluttered painterly background; no mirrors/reflective glass/transparent objects; static medium-wide shot, fixed camera, no pan, no zoom, no parallax; 16:9 cinematic widescreen.
- **Anime guardrails:** avoid chibi exaggerations, moe big-eye style, 3D CGI, glossy plastic shading, cel-shaded toy look, manga panel borders, speech bubbles, sticker-art look.

**Drops vs. original `SEEDREAM_HARD_CONSTRAINTS`** (intentional):
- "Photographic realism, no AI-style artifacts, no oversaturation, no fake bokeh, no HDR look" — incompatible with watercolor anime
- "No people, no human figures, no faces, no hands, no fingers" — narrowed to "No human..." so the capybara face/eyes are allowed to render

### 4.4 LTX negative prompt (real `negative_prompt` field)

```python
CAPYBARA_LTX_NEGATIVE = (
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
```

**Drops vs. original `LTX_NEGATIVE_PROMPT`** (intentional):
- `hands, fingers, face, faces, eyes, mouth, lips` (generic) → narrowed to `human hands, human fingers, human face, human faces` so the capybara renders properly
- `animal legs, paws, wings, fur detail` removed — these define the subject; suppressing them is actively destructive in LTX-2.3's text encoder

### 4.5 Tiny-arc 6-clip motion templates

Per-clip motion budget — every clip shares the same scene + steam wisps + capybara breathing baseline; clips 2-5 each add **exactly one** extra micro-motion that fades back to rest by clip 6:

| Clip | Role | Always present | + Added element |
|------|---|---|---|
| 1 | rest (loop seam) | steam wisps + capybara breathing | none |
| 2 | settle-in | same | a single leaf drifts downward at frame edge |
| 3 | core | same | one slow ear twitch + leaf settles |
| 4 | core | same | one slow eye blink + faint dust motes |
| 5 | release | same | one soft slow tail flick + light shifts gently |
| 6 | rest (loop seam) | steam wisps + capybara breathing | none — string-equal to clip 1 |

**Hard invariant enforced in tests:** `motion_prompts[0] == motion_prompts[5]` (string equality). The orchestrator's existing crossfade logic relies on this matching to make the loop seam invisible.

## 5. CLI surface

```
python scripts/capybara_tea_loop.py
python scripts/capybara_tea_loop.py --setting forest_hot_spring
python scripts/capybara_tea_loop.py --duration 30 --seed 42
python scripts/capybara_tea_loop.py --output-dir out/capybara/run-001/
```

| Flag | Default | Purpose |
|---|---|---|
| `--setting <key>` | random | Pick one of the 10 curated settings; raises if key unknown |
| `--duration <minutes>` | 60 | Total target length (scales segment count via existing logic) |
| `--seed <int>` | random | Deterministic setting + ACE-Step + Seedream + LTX seeds |
| `--output-dir <path>` | `out/capybara_tea/<timestamp>/` | Where the run writes `final.mp4` and `manifest.json` |

## 6. Module layout

```python
# scripts/loopvid/capybara_preset.py
from scripts.loopvid.constants import (
    SEGMENT_COUNT_60MIN, MUSIC_PALETTE_MAX_CHARS, MUSIC_SEGMENT_DESCRIPTOR_MAX_CHARS,
    CLIP_COUNT,
)
from scripts.loopvid.plan_schema import Plan, validate_plan_dict

CAPYBARA_GENRE: str
CAPYBARA_MUSIC_BPM: int
CAPYBARA_MUSIC_PALETTE: str
CAPYBARA_SEGMENT_DESCRIPTORS: list[dict]
CAPYBARA_SEEDREAM_STYLE: str               # Ghibli style anchor (goes in Plan.seedream_style)
CAPYBARA_SEEDREAM_CONSTRAINTS: str         # all guardrails (passed as build_seedream_prompt's `constraints` arg)
CAPYBARA_LTX_NEGATIVE: str
CAPYBARA_SETTINGS: list[dict]              # 10 entries
PRESET_SENTINEL_KEY: str = "capybara_tea"  # used for image_archetype_key + motion_archetype

def pick_setting(seed: int | None = None) -> dict:
    """Random pick from CAPYBARA_SETTINGS, deterministic given seed."""

def get_setting_by_key(key: str) -> dict:
    """Lookup by key; raises ValueError(valid_keys=...) if unknown."""

def build_motion_prompts(setting: dict) -> list[str]:
    """Returns 6 prompt strings; prompts[0] == prompts[5]."""

def build_plan(setting: dict) -> Plan:
    """Assemble + validate a Plan from a setting dict."""
```

```python
# scripts/capybara_tea_loop.py — top-level skeleton
def main() -> int:
    args = parse_args()
    setting = (get_setting_by_key(args.setting) if args.setting
               else pick_setting(seed=args.seed))
    plan = build_plan(setting)
    return orchestrator.run(
        plan,
        output_dir=args.output_dir,
        duration_minutes=args.duration,
        seed=args.seed,
        seedream_constraints=CAPYBARA_SEEDREAM_CONSTRAINTS,
        ltx_negative=CAPYBARA_LTX_NEGATIVE,
    )
```

## 7. Error handling

| Failure | Behavior |
|---|---|
| `--setting <key>` not in `CAPYBARA_SETTINGS` | Exit 2 with message listing all 10 valid keys |
| `build_plan()` produces a Plan that fails `validate_plan_dict()` | Raise `PlanSchemaError` (caught by orchestrator's existing handler); indicates a constants drift, must be fixed in code |
| Seedream / RunPod / ACE-Step API failure | Existing `orchestrator.py` retry + rollback logic handles it (see `2026-04-27-loop-music-video-design.md`) |
| Loop seam invariant violation (`motion_prompts[0] != motion_prompts[5]`) | Caught at unit-test level; should never reach runtime |

No new operational gotchas beyond those documented in `MEMORY.md` (`runpod_acestep_operational_gotchas.md`, `loopvid_hardening_backlog.md`). The existing 2 HIGH backlog items (non-atomic `plan.json` write, missing try/except wrap) apply to this fork too — addressed separately, not in this design.

## 8. Testing strategy

### 8.1 Unit tests (`test_capybara_preset.py`, offline)

| Test | Asserts |
|---|---|
| `test_settings_list_well_formed` | exactly 10 entries; each has `key`/`scene`/`lighting`/`palette`; keys unique and snake_case |
| `test_music_palette_within_cap` | `CAPYBARA_MUSIC_PALETTE` ≤ `MUSIC_PALETTE_MAX_CHARS` |
| `test_segment_descriptors_within_cap` | exactly 11 entries; each `descriptors` ≤ `MUSIC_SEGMENT_DESCRIPTOR_MAX_CHARS` |
| `test_segment_phase_arc` | phase sequence is exactly `3×settle, 1×hold, 5×deepen, 2×dissolve` |
| `test_build_plan_returns_validated_plan` | `build_plan(settings[0])` round-trips through `validate_plan_dict()` |
| `test_build_plan_six_motion_prompts` | `motion_prompts` length == `CLIP_COUNT` (6) |
| `test_loop_seam_invariant` | `motion_prompts[0] == motion_prompts[5]` |
| `test_motion_prompts_share_scene_prefix` | every clip prompt contains `setting["scene"]` |
| `test_seedream_prompt_concatenation` | `build_seedream_prompt(scene, CAPYBARA_SEEDREAM_STYLE, constraints=CAPYBARA_SEEDREAM_CONSTRAINTS)` contains scene, the Ghibli style anchor, and the universal-block phrase from constraints |
| `test_validate_plan_dict_extra_archetypes` | passing `extra_archetype_keys={"capybara_tea"}` lets `image_archetype_key="capybara_tea"` validate; without it, it raises |
| `test_pick_setting_deterministic_with_seed` | `pick_setting(seed=42)` is repeatable; different seeds eventually return different keys |
| `test_unknown_setting_key_raises` | `get_setting_by_key("nope")` raises with valid-key list in the error |

### 8.2 Plan-validation integration test (offline)

`test_capybara_plan_validates_for_all_settings` — loops over all 10 entries, asserts each `build_plan(setting)` passes `validate_plan_dict()`. Catches schema regressions when constants drift.

### 8.3 Live E2E smoke (opt-in)

`test_capybara_5min_smoke.py`, gated by `RUN_LIVE_SMOKE=1` env var:

```bash
RUN_LIVE_SMOKE=1 python scripts/capybara_tea_loop.py \
    --setting forest_hot_spring --duration 5 --seed 42 \
    --output-dir out/capybara/smoke-test/
```

Asserts: exit code 0; `final.mp4` exists, length ≈ 5 min ±5s; manifest has 6 LTX clip ids + 1 Seedream prediction id.

Not in CI — costs real $ and takes 6-8 min wall-clock. Documented in `--help`.

### 8.4 Out of scope

Visual quality of the still / video, music palette adherence, exact prompt wording — all inspected by eye, not asserted in tests. We test *structure* (length, seam, schema), not *content*.

## 9. Open questions / future work

- **Setting expansion:** v1 ships with 10 curated settings. After 10+ uploads, we may want 15-20 to keep the feed fresh. Adding entries is a constants-only change.
- **Optional `--llm-vary` flag:** if locked-template variability proves too tight, add a thin Gemini call to vary micro-details (time-of-day, what's on the tray) within a chosen setting. Out of scope for v1.
- **Brand sub-themes:** future variants like `capybara_coffee_loop.py`, `capybara_ramen_loop.py` could fork this preset module's structure. Not built yet.

## 10. Dependencies

No new pip dependencies. Reuses everything already pinned for the existing loopvid stack. New module is pure-Python templating + the existing `requests`/`runpod`/`replicate`/`ffmpeg` stack.

# Capybara + Tea Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fork the existing loopvid pipeline into a parallel script (`scripts/capybara_tea_loop.py`) that produces a 60-min Studio Ghibli capybara+tea looping music video with locked Nujabes lofi music, tiny-arc 6-clip motion, and a 10-entry curated setting list. No LLM in the loop — pure template.

**Architecture:** All creative content (music palette, segment descriptors, motion prompts, Seedream style/constraints, LTX negative) lives in a new `scripts/loopvid/capybara_preset.py` constants module. The entry script picks a setting (random or via `--setting`), builds a Plan dict via the preset, and invokes the existing `run_orchestrator()` with a `preset_plan_dict` that bypasses the LLM step. Three thin additive changes to existing modules (`build_seedream_prompt`, `build_clip_payload`, `validate_plan_dict`, `OrchestratorConfig`) are all backwards-compatible — the LLM-driven `loop_music_video.py` path keeps working unchanged.

**Tech Stack:** Python 3.11+, pytest, `responses` (HTTP mocking), existing loopvid stack (Replicate Seedream 4.5, RunPod ACE-Step + LTX-2.3, ffmpeg).

**Spec:** `docs/superpowers/specs/2026-04-28-capybara-tea-loop-design.md` (commit `c0aead1`).

---

## File Structure

**Create:**
- `scripts/loopvid/capybara_preset.py` — locked constants + Plan-dict factory
- `scripts/capybara_tea_loop.py` — CLI entry script
- `test_capybara_preset.py` — unit tests at repo root (matches `test_loopvid_*.py` convention)
- `test_capybara_tea_loop_smoke.py` — opt-in live 5-min E2E smoke test

**Modify (additive only — backwards-compatible):**
- `scripts/loopvid/plan_schema.py` — add `extra_archetype_keys` + `extra_motion_archetypes` kwargs to `validate_plan_dict()`
- `scripts/loopvid/image_pipeline.py` — add `constraints` kwarg to `build_seedream_prompt()`
- `scripts/loopvid/video_pipeline.py` — add `negative_prompt` kwarg to `build_clip_payload()` and `run_video_pipeline()`
- `scripts/loopvid/orchestrator.py` — add `seedream_constraints`, `ltx_negative`, `extra_archetype_keys`, `extra_motion_archetypes`, `preset_plan_dict` to `OrchestratorConfig`; thread to call sites

**Untouched (existing behavior preserved):**
- `scripts/loop_music_video.py` (the LLM-driven entry script — keeps working with default kwargs)
- `scripts/loopvid/constants.py` (no edits — original `SEEDREAM_HARD_CONSTRAINTS`, `LTX_NEGATIVE_PROMPT`, `GENRE_ARCHETYPES`, `ALLOWED_MOTION_ARCHETYPES` all stay)
- `scripts/loopvid/llm_planner.py`, `music_pipeline.py`, `loop_build.py`, `mux.py`, `runpod_client.py`, `preflight.py`, `manifest.py`, `cost.py`, `rollback.py`

---

## Task 1: Schema — extend `validate_plan_dict` with extra-archetype kwargs

**Files:**
- Modify: `scripts/loopvid/plan_schema.py:46-108`
- Test: `test_loopvid_plan_schema.py` (append two new tests)

- [ ] **Step 1.1: Write failing tests for the new kwargs**

Append to `test_loopvid_plan_schema.py`:

```python
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
```

- [ ] **Step 1.2: Run tests — verify they fail**

```bash
cd /root/ace-step-music-xl
pytest test_loopvid_plan_schema.py -v -k "extra_archetype or extra_motion"
```
Expected: 4 failures (`TypeError: validate_plan_dict() got an unexpected keyword argument 'extra_archetype_keys'` etc.).

- [ ] **Step 1.3: Implement the kwargs**

In `scripts/loopvid/plan_schema.py`, change the `validate_plan_dict` signature and the two archetype checks:

```python
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
```

- [ ] **Step 1.4: Run all schema tests — verify pass**

```bash
pytest test_loopvid_plan_schema.py -v
```
Expected: all tests pass (existing tests + 4 new).

- [ ] **Step 1.5: Commit**

```bash
git add scripts/loopvid/plan_schema.py test_loopvid_plan_schema.py
git commit -m "feat(plan_schema): add extra_archetype_keys/extra_motion_archetypes kwargs

Backwards-compatible: defaults preserve existing closed-set validation
(GENRE_ARCHETYPES + ALLOWED_MOTION_ARCHETYPES). Allows preset-driven
forks to register their own archetype sentinel without polluting the
LLM-instructed allowlist."
```

---

## Task 2: Image pipeline — add `constraints` kwarg to `build_seedream_prompt`

**Files:**
- Modify: `scripts/loopvid/image_pipeline.py:21-22`
- Test: `test_loopvid_image_pipeline.py` (append one new test)

- [ ] **Step 2.1: Write failing test**

Append to `test_loopvid_image_pipeline.py`:

```python
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
```

- [ ] **Step 2.2: Run tests — verify they fail**

```bash
pytest test_loopvid_image_pipeline.py -v -k "custom_constraints or default_constraints"
```
Expected: `test_build_seedream_prompt_uses_custom_constraints_when_given` fails (`TypeError: build_seedream_prompt() got an unexpected keyword argument 'constraints'`).

- [ ] **Step 2.3: Implement the kwarg**

Edit `scripts/loopvid/image_pipeline.py:21-22`:

```python
def build_seedream_prompt(scene: str, style: str, *, constraints: str = SEEDREAM_HARD_CONSTRAINTS) -> str:
    return f"{scene}. {style}. {constraints}"
```

- [ ] **Step 2.4: Run all image_pipeline tests — verify pass**

```bash
pytest test_loopvid_image_pipeline.py -v
```
Expected: all pass.

- [ ] **Step 2.5: Commit**

```bash
git add scripts/loopvid/image_pipeline.py test_loopvid_image_pipeline.py
git commit -m "feat(image_pipeline): add constraints kwarg to build_seedream_prompt

Backwards-compatible: default keeps SEEDREAM_HARD_CONSTRAINTS. Allows
preset-driven forks to inject their own positive-form constraints
(Seedream 4.5 has no negative_prompt field on Replicate)."
```

---

## Task 3: Video pipeline — accept custom `negative_prompt`

**Files:**
- Modify: `scripts/loopvid/video_pipeline.py:47-62, 80-124`
- Test: `test_loopvid_video_pipeline.py` (append two new tests)

- [ ] **Step 3.1: Write failing tests**

Append to `test_loopvid_video_pipeline.py`:

```python
def test_build_clip_payload_uses_custom_negative_when_given():
    payload = build_clip_payload(
        image_b64="img", audio_b64="aud", motion_prompt="p", seed=1,
        negative_prompt="custom negative",
    )
    assert payload["input"]["negative_prompt"] == "custom negative"


def test_build_clip_payload_default_negative_unchanged():
    payload = build_clip_payload(
        image_b64="img", audio_b64="aud", motion_prompt="p", seed=1,
    )
    assert payload["input"]["negative_prompt"] == LTX_NEGATIVE_PROMPT
```

(Add the import at the top if not present: `from scripts.loopvid.constants import LTX_NEGATIVE_PROMPT`.)

- [ ] **Step 3.2: Run tests — verify they fail**

```bash
pytest test_loopvid_video_pipeline.py -v -k "custom_negative or default_negative"
```
Expected: `test_build_clip_payload_uses_custom_negative_when_given` fails (`TypeError: build_clip_payload() got an unexpected keyword argument 'negative_prompt'`).

- [ ] **Step 3.3: Implement `negative_prompt` parameter on both functions**

Edit `scripts/loopvid/video_pipeline.py:47-62`:

```python
def build_clip_payload(
    *, image_b64: str, audio_b64: str, motion_prompt: str, seed: int,
    negative_prompt: str = LTX_NEGATIVE_PROMPT,
) -> dict:
    return {
        "input": {
            "image_base64": image_b64,
            "audio_base64": audio_b64,
            "prompt": motion_prompt,
            "negative_prompt": negative_prompt,
            "num_frames": CLIP_NUM_FRAMES,
            "fps": CLIP_FPS,
            "seed": seed,
            "width": CLIP_WIDTH,
            "height": CLIP_HEIGHT,
        }
    }
```

Edit `scripts/loopvid/video_pipeline.py:80-89` — the `run_video_pipeline` signature:

```python
def run_video_pipeline(
    *, run_id: str,
    still_path: Path,
    audio_chunks: list[Path],
    motion_prompts: list[str],
    out_dir: Path,
    endpoint_id: str,
    api_key: str,
    on_clip_done: Optional[Callable[[int, Path], None]] = None,
    negative_prompt: str = LTX_NEGATIVE_PROMPT,
) -> list[Path]:
```

And edit `scripts/loopvid/video_pipeline.py:106-111` — pass it down:

```python
        payload = build_clip_payload(
            image_b64=image_b64,
            audio_b64=audio_b64,
            motion_prompt=motion_prompts[i - 1],
            seed=stable_clip_seed(run_id, i),
            negative_prompt=negative_prompt,
        )
```

- [ ] **Step 3.4: Run all video_pipeline tests — verify pass**

```bash
pytest test_loopvid_video_pipeline.py -v
```
Expected: all pass.

- [ ] **Step 3.5: Commit**

```bash
git add scripts/loopvid/video_pipeline.py test_loopvid_video_pipeline.py
git commit -m "feat(video_pipeline): accept custom negative_prompt

Backwards-compatible: default keeps LTX_NEGATIVE_PROMPT. Allows preset
forks to override (e.g. capybara unblocks 'fur detail / paws / animal
eyes' which the generic negative blocks)."
```

---

## Task 4: Orchestrator — add new `OrchestratorConfig` fields

**Files:**
- Modify: `scripts/loopvid/orchestrator.py:30-46, 97-124, 160-167, 175-189`
- Test: `test_loopvid_orchestrator.py` (append tests for the new fields)

- [ ] **Step 4.1: Write failing tests**

First, look at the existing test patterns in `test_loopvid_orchestrator.py` to confirm the fixture style (run `head -80 test_loopvid_orchestrator.py`).

Append to `test_loopvid_orchestrator.py` (adapt fixture style as needed):

```python
def test_orchestrator_config_has_new_optional_fields():
    cfg = OrchestratorConfig(
        run_id="r", run_dir=Path("/tmp/r"), genre="g", mood="", duration_sec=60,
        ace_step_endpoint="a", ltx_endpoint="l",
        runpod_api_key="", openrouter_api_key="", replicate_api_token="",
    )
    assert cfg.seedream_constraints is None
    assert cfg.ltx_negative is None
    assert cfg.extra_archetype_keys is None
    assert cfg.extra_motion_archetypes is None
    assert cfg.preset_plan_dict is None
```

- [ ] **Step 4.2: Run test — verify it fails**

```bash
pytest test_loopvid_orchestrator.py::test_orchestrator_config_has_new_optional_fields -v
```
Expected: `TypeError` (unknown attribute) or `AttributeError`.

- [ ] **Step 4.3: Add the new fields**

Edit `scripts/loopvid/orchestrator.py:30-46`:

```python
@dataclass
class OrchestratorConfig:
    run_id: str
    run_dir: Path
    genre: str
    mood: str
    duration_sec: int
    ace_step_endpoint: str
    ltx_endpoint: str
    runpod_api_key: str
    openrouter_api_key: str
    replicate_api_token: str
    only: Optional[tuple] = None
    skip: Optional[tuple] = None
    force: bool = False
    dry_run: bool = False
    max_cost: Optional[float] = None
    seedream_constraints: Optional[str] = None
    ltx_negative: Optional[str] = None
    extra_archetype_keys: Optional[set] = None
    extra_motion_archetypes: Optional[set] = None
    preset_plan_dict: Optional[dict] = None
```

- [ ] **Step 4.4: Run test — verify pass**

```bash
pytest test_loopvid_orchestrator.py::test_orchestrator_config_has_new_optional_fields -v
```
Expected: PASS.

- [ ] **Step 4.5: Commit**

```bash
git add scripts/loopvid/orchestrator.py test_loopvid_orchestrator.py
git commit -m "feat(orchestrator): add 5 optional config fields for preset forks

seedream_constraints, ltx_negative, extra_archetype_keys,
extra_motion_archetypes, preset_plan_dict — all default None to preserve
LLM-driven path. Plumbing in next commit."
```

---

## Task 5: Orchestrator — plumb new fields through to call sites

**Files:**
- Modify: `scripts/loopvid/orchestrator.py:97-124, 160-167, 175-189`
- Test: `test_loopvid_orchestrator.py` (append integration test)

- [ ] **Step 5.1: Write failing test**

This test verifies that when `preset_plan_dict` is given, the LLM is NOT called and the dict is written to plan.json directly. Append to `test_loopvid_orchestrator.py` (adapt to existing fixture/mock style — likely uses `monkeypatch` to stub external calls):

```python
def test_preset_plan_dict_skips_llm(tmp_path, monkeypatch):
    """When preset_plan_dict is set, orchestrator skips the LLM call and
    writes the dict directly to plan.json."""
    from scripts.loopvid import orchestrator as orch
    llm_called = {"yes": False}
    def fake_plan(**kwargs):
        llm_called["yes"] = True
        raise AssertionError("LLM should not be called when preset_plan_dict is set")
    monkeypatch.setattr(orch, "plan", fake_plan)
    # Stub everything after the plan step to short-circuit
    monkeypatch.setattr(orch, "run_preflight", lambda **k: None)
    monkeypatch.setattr(orch, "run_music_pipeline", lambda **k: [])
    monkeypatch.setattr(orch, "generate_still", lambda **k: "stub-pred")
    monkeypatch.setattr(orch, "run_video_pipeline", lambda **k: [])
    monkeypatch.setattr(orch, "slice_audio_chunks", lambda *a, **k: [])
    monkeypatch.setattr(orch, "concat_clips_with_xfades", lambda *a, **k: None)
    monkeypatch.setattr(orch, "add_loop_seam_fade", lambda *a, **k: None)
    monkeypatch.setattr(orch, "final_assembly", lambda *a, **k: None)

    preset = {
        "genre": "lofi", "mood": "",
        "music_palette": "lofi in the style of Nujabes, instrumental, 75 bpm",
        "music_segment_descriptors": [
            {"phase": "settle", "descriptors": "soft intro"} for _ in range(11)
        ],
        "music_bpm": 75,
        "seedream_scene": "scene", "seedream_style": "style",
        "motion_prompts": ["p"] * 6,
        "motion_archetype": "capybara_tea",
        "image_archetype_key": "capybara_tea",
    }
    cfg = OrchestratorConfig(
        run_id="r", run_dir=tmp_path / "r", genre="lofi", mood="", duration_sec=60,
        ace_step_endpoint="a", ltx_endpoint="l",
        runpod_api_key="k", openrouter_api_key="", replicate_api_token="t",
        skip=("music", "image", "video", "loop_build", "mux"),
        preset_plan_dict=preset,
        extra_archetype_keys={"capybara_tea"},
        extra_motion_archetypes={"capybara_tea"},
    )
    orch.run_orchestrator(cfg)
    assert llm_called["yes"] is False
    assert (tmp_path / "r" / "plan.json").exists()
    saved = json.loads((tmp_path / "r" / "plan.json").read_text())
    assert saved["music_palette"] == preset["music_palette"]
```

(Add `import json` at the top if not present.)

- [ ] **Step 5.2: Run test — verify it fails**

```bash
pytest test_loopvid_orchestrator.py::test_preset_plan_dict_skips_llm -v
```
Expected: FAIL (LLM `plan()` is called regardless of preset_plan_dict, OR validate_plan_dict rejects "capybara_tea").

- [ ] **Step 5.3: Edit the plan step**

In `scripts/loopvid/orchestrator.py:97-119`, replace the plan step:

```python
    # Step 1: plan
    if _should_run("plan", cfg, m.steps["plan"]["status"]):
        mark_step_in_progress(cfg.run_dir, "plan")
        if cfg.preset_plan_dict is not None:
            _print("▸ plan (preset)")
            plan_dict_to_save = cfg.preset_plan_dict
        else:
            _print("▸ plan (LLM)")
            plan_obj = plan(
                genre=cfg.genre, mood=cfg.mood,
                api_key=cfg.openrouter_api_key,
                raw_response_path=str(cfg.run_dir / "plan_raw.json"),
            )
            plan_dict_to_save = {
                "genre": plan_obj.genre,
                "mood": plan_obj.mood,
                "music_palette": plan_obj.music_palette,
                "music_segment_descriptors": plan_obj.music_segment_descriptors,
                "music_bpm": plan_obj.music_bpm,
                "seedream_scene": plan_obj.seedream_scene,
                "seedream_style": plan_obj.seedream_style,
                "motion_prompts": plan_obj.motion_prompts,
                "motion_archetype": plan_obj.motion_archetype,
                "image_archetype_key": plan_obj.image_archetype_key,
            }
        plan_path.write_text(json.dumps(plan_dict_to_save, indent=2, sort_keys=True))
        mark_step_done(cfg.run_dir, "plan")
        _print("✓ plan committed")
    else:
        _print("✓ plan (cached)")
```

Then update the `validate_plan_dict` call at line 124:

```python
    plan_dict = json.loads(plan_path.read_text())
    plan_obj = validate_plan_dict(
        plan_dict,
        extra_archetype_keys=cfg.extra_archetype_keys,
        extra_motion_archetypes=cfg.extra_motion_archetypes,
    )
```

Then thread `seedream_constraints` to `build_seedream_prompt` at line 163:

```python
        prompt_str = build_seedream_prompt(
            plan_obj.seedream_scene, plan_obj.seedream_style,
            constraints=cfg.seedream_constraints or SEEDREAM_HARD_CONSTRAINTS,
        )
```

(Add `from scripts.loopvid.constants import SEEDREAM_HARD_CONSTRAINTS` to the imports at the top.)

Then thread `ltx_negative` to `run_video_pipeline` at line 179:

```python
        run_video_pipeline(
            run_id=cfg.run_id,
            still_path=still_path,
            audio_chunks=chunks,
            motion_prompts=plan_obj.motion_prompts,
            out_dir=video_dir,
            endpoint_id=cfg.ltx_endpoint, api_key=cfg.runpod_api_key,
            on_clip_done=lambda i, p: _print(f"  ✓ clip {i}"),
            negative_prompt=cfg.ltx_negative or LTX_NEGATIVE_PROMPT,
        )
```

(Add `LTX_NEGATIVE_PROMPT` to the existing constants import block.)

- [ ] **Step 5.4: Run all orchestrator tests — verify pass**

```bash
pytest test_loopvid_orchestrator.py -v
```
Expected: all pass (including the new test and the existing ones).

- [ ] **Step 5.5: Commit**

```bash
git add scripts/loopvid/orchestrator.py test_loopvid_orchestrator.py
git commit -m "feat(orchestrator): plumb preset config fields to call sites

- preset_plan_dict short-circuits LLM step, writes dict to plan.json
- extra_archetype_keys/extra_motion_archetypes pass through to validator
- seedream_constraints overrides build_seedream_prompt's default
- ltx_negative overrides run_video_pipeline's default

LLM-driven loop_music_video.py path is unchanged (all kwargs default to
None and fall back to existing constants)."
```

---

## Task 6: Capybara preset — locked constants module

**Files:**
- Create: `scripts/loopvid/capybara_preset.py`
- Create: `test_capybara_preset.py`

- [ ] **Step 6.1: Write failing tests**

Create `test_capybara_preset.py`:

```python
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
```

- [ ] **Step 6.2: Run tests — verify they fail (module doesn't exist yet)**

```bash
pytest test_capybara_preset.py -v
```
Expected: ImportError on `scripts.loopvid.capybara_preset`.

- [ ] **Step 6.3: Create `scripts/loopvid/capybara_preset.py`**

```python
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
    "No human figures, no human faces, no human hands, no human fingers. "
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
```

- [ ] **Step 6.4: Run tests — verify pass**

```bash
pytest test_capybara_preset.py -v
```
Expected: all 19 tests pass.

- [ ] **Step 6.5: Commit**

```bash
git add scripts/loopvid/capybara_preset.py test_capybara_preset.py
git commit -m "feat(capybara_preset): locked constants for capybara+tea loop

10-entry curated settings, locked Nujabes lofi palette, 11-segment
3-1-5-2 phase arc, Studio Ghibli style anchor, anime-aware Seedream
constraints + LTX negative (drops photographic-realism, narrows
'no faces/hands' to humans only so capybara renders properly)."
```

---

## Task 7: Capybara preset — setting selection helpers

**Files:**
- Modify: `scripts/loopvid/capybara_preset.py` (append functions)
- Modify: `test_capybara_preset.py` (append tests)

- [ ] **Step 7.1: Write failing tests**

Append to `test_capybara_preset.py`:

```python
from scripts.loopvid.capybara_preset import pick_setting, get_setting_by_key


def test_pick_setting_returns_a_setting_dict():
    s = pick_setting()
    assert s in CAPYBARA_SETTINGS


def test_pick_setting_deterministic_with_seed():
    a = pick_setting(seed=42)
    b = pick_setting(seed=42)
    assert a["key"] == b["key"]


def test_pick_setting_different_seeds_eventually_diverge():
    keys = {pick_setting(seed=i)["key"] for i in range(50)}
    # With 10 settings and 50 different seeds, we expect to cover most/all keys
    assert len(keys) >= 5


def test_get_setting_by_key_returns_match():
    s = get_setting_by_key("forest_hot_spring")
    assert s["key"] == "forest_hot_spring"
    assert "capybara" in s["scene"].lower()


def test_get_setting_by_key_unknown_raises_with_valid_keys_listed():
    with pytest.raises(ValueError) as exc:
        get_setting_by_key("not_a_real_key")
    msg = str(exc.value)
    assert "not_a_real_key" in msg
    # Error must list at least one known valid key for discoverability
    assert "forest_hot_spring" in msg
```

- [ ] **Step 7.2: Run tests — verify they fail**

```bash
pytest test_capybara_preset.py -v -k "pick_setting or get_setting_by_key"
```
Expected: ImportError or AttributeError.

- [ ] **Step 7.3: Implement the helpers**

Append to `scripts/loopvid/capybara_preset.py`:

```python
import random


def pick_setting(seed: int | None = None) -> dict:
    """Random pick from CAPYBARA_SETTINGS, deterministic if seed is given."""
    rng = random.Random(seed) if seed is not None else random.Random()
    return rng.choice(CAPYBARA_SETTINGS)


def get_setting_by_key(key: str) -> dict:
    """Lookup by key. Raises ValueError listing valid keys if unknown."""
    for s in CAPYBARA_SETTINGS:
        if s["key"] == key:
            return s
    valid = sorted(s["key"] for s in CAPYBARA_SETTINGS)
    raise ValueError(
        f"unknown setting key: {key!r}. Valid keys: {valid}"
    )
```

(`import random` goes at the top of the file with the other imports.)

- [ ] **Step 7.4: Run tests — verify pass**

```bash
pytest test_capybara_preset.py -v
```
Expected: all pass.

- [ ] **Step 7.5: Commit**

```bash
git add scripts/loopvid/capybara_preset.py test_capybara_preset.py
git commit -m "feat(capybara_preset): pick_setting + get_setting_by_key

Deterministic selection given --seed; lookup raises with valid keys
listed for discoverability when key is unknown."
```

---

## Task 8: Capybara preset — `build_motion_prompts`

**Files:**
- Modify: `scripts/loopvid/capybara_preset.py`
- Modify: `test_capybara_preset.py`

- [ ] **Step 8.1: Write failing tests**

Append to `test_capybara_preset.py`:

```python
from scripts.loopvid.capybara_preset import build_motion_prompts


def test_build_motion_prompts_returns_six_strings():
    s = CAPYBARA_SETTINGS[0]
    prompts = build_motion_prompts(s)
    assert len(prompts) == 6
    assert all(isinstance(p, str) for p in prompts)


def test_build_motion_prompts_loop_seam_invariant():
    """Clip 1 and clip 6 must be string-equal so the loop seam is invisible."""
    for s in CAPYBARA_SETTINGS:
        prompts = build_motion_prompts(s)
        assert prompts[0] == prompts[5], f"seam broken for {s['key']}"


def test_build_motion_prompts_share_scene_prefix():
    s = CAPYBARA_SETTINGS[0]
    prompts = build_motion_prompts(s)
    for i, p in enumerate(prompts):
        assert s["scene"] in p, f"clip {i} missing scene"
        assert "Ghibli" in p, f"clip {i} missing style anchor"
        assert "Camera locked" in p, f"clip {i} missing camera-lock"


def test_build_motion_prompts_each_clip_distinct_except_seam():
    s = CAPYBARA_SETTINGS[0]
    prompts = build_motion_prompts(s)
    # clip 1 == clip 6 (seam); clips 2-5 must all be different from each other AND from rest
    assert len(set(prompts[1:5])) == 4, "clips 2-5 must each be distinct"
    assert prompts[0] not in prompts[1:5], "rest state must differ from active clips"


def test_build_motion_prompts_micro_motion_elements_present():
    s = CAPYBARA_SETTINGS[0]
    prompts = build_motion_prompts(s)
    # Each non-seam clip adds exactly one named micro-motion
    assert "leaf" in prompts[1].lower()
    assert "ear" in prompts[2].lower()
    assert "blink" in prompts[3].lower()
    assert "tail" in prompts[4].lower()
```

- [ ] **Step 8.2: Run tests — verify they fail**

```bash
pytest test_capybara_preset.py -v -k "build_motion_prompts"
```
Expected: ImportError.

- [ ] **Step 8.3: Implement `build_motion_prompts`**

Append to `scripts/loopvid/capybara_preset.py`:

```python
def build_motion_prompts(setting: dict) -> list[str]:
    """6 motion-prompt strings. Tiny-arc: clip 1 and clip 6 are the rest
    state (string-equal for invisible loop seam); clips 2-5 each add one
    distinct micro-motion element."""
    base = (
        f"{setting['scene']}. {setting['lighting']}. {setting['palette']}. "
        f"Studio Ghibli soft watercolor anime style. "
        f"Ambient micro-motion only:"
    )
    rest_state = (
        f"{base} thin wisps of warm steam rising slowly from the teacup, "
        f"the capybara's chest barely rises and falls. "
        f"Camera locked. Nothing else moves."
    )
    return [
        rest_state,  # clip 1 — rest (loop seam)
        f"{base} thin wisps of warm steam, capybara breathing softly, "
        f"a single leaf drifts downward slowly at the edge of frame. "
        f"Camera locked.",                                               # clip 2 — leaf drift
        f"{base} thin wisps of warm steam, capybara breathing softly, "
        f"one slow ear twitch, a leaf settles onto the ground. "
        f"Camera locked.",                                               # clip 3 — ear twitch
        f"{base} thin wisps of warm steam, capybara breathing softly, "
        f"one slow eye blink, faint dust motes drift through a sunbeam. "
        f"Camera locked.",                                               # clip 4 — eye blink
        f"{base} thin wisps of warm steam, capybara breathing softly, "
        f"one slow soft tail flick, the light shifts gently. "
        f"Camera locked.",                                               # clip 5 — tail flick
        rest_state,  # clip 6 — rest (must equal clip 1)
    ]
```

- [ ] **Step 8.4: Run tests — verify pass**

```bash
pytest test_capybara_preset.py -v
```
Expected: all pass.

- [ ] **Step 8.5: Commit**

```bash
git add scripts/loopvid/capybara_preset.py test_capybara_preset.py
git commit -m "feat(capybara_preset): build_motion_prompts (tiny-arc 6 clips)

clip 1 ≡ clip 6 (string-equal rest state) for invisible loop seam.
Clips 2-5 each add one distinct micro-motion: leaf drift, ear twitch,
eye blink, tail flick — escalates and settles back to rest."
```

---

## Task 9: Capybara preset — `build_plan_dict`

**Files:**
- Modify: `scripts/loopvid/capybara_preset.py`
- Modify: `test_capybara_preset.py`

- [ ] **Step 9.1: Write failing tests**

Append to `test_capybara_preset.py`:

```python
from scripts.loopvid.capybara_preset import build_plan_dict
from scripts.loopvid.plan_schema import validate_plan_dict


def test_build_plan_dict_validates_with_extra_archetype_keys():
    s = CAPYBARA_SETTINGS[0]
    d = build_plan_dict(s)
    plan = validate_plan_dict(
        d,
        extra_archetype_keys={PRESET_SENTINEL_KEY},
        extra_motion_archetypes={PRESET_SENTINEL_KEY},
    )
    assert plan.music_palette == CAPYBARA_MUSIC_PALETTE
    assert plan.music_bpm == CAPYBARA_MUSIC_BPM
    assert plan.image_archetype_key == PRESET_SENTINEL_KEY
    assert plan.motion_archetype == PRESET_SENTINEL_KEY


def test_build_plan_dict_works_for_all_settings():
    for s in CAPYBARA_SETTINGS:
        d = build_plan_dict(s)
        validate_plan_dict(
            d,
            extra_archetype_keys={PRESET_SENTINEL_KEY},
            extra_motion_archetypes={PRESET_SENTINEL_KEY},
        )


def test_build_plan_dict_seedream_scene_includes_all_three_setting_fields():
    s = CAPYBARA_SETTINGS[0]
    d = build_plan_dict(s)
    assert s["scene"] in d["seedream_scene"]
    assert s["lighting"] in d["seedream_scene"]
    assert s["palette"] in d["seedream_scene"]


def test_build_plan_dict_seedream_style_is_ghibli_anchor():
    s = CAPYBARA_SETTINGS[0]
    d = build_plan_dict(s)
    assert d["seedream_style"] == CAPYBARA_SEEDREAM_STYLE


def test_build_plan_dict_motion_prompts_six_with_loop_seam():
    s = CAPYBARA_SETTINGS[0]
    d = build_plan_dict(s)
    assert len(d["motion_prompts"]) == 6
    assert d["motion_prompts"][0] == d["motion_prompts"][5]


def test_build_seedream_prompt_with_capybara_constraints_concatenates_all_parts():
    """Smoke-test: the rendered Seedream prompt produced by the existing
    build_seedream_prompt() with the capybara constraints contains scene,
    style anchor, AND universal-block phrase from constraints."""
    from scripts.loopvid.image_pipeline import build_seedream_prompt
    s = CAPYBARA_SETTINGS[0]
    d = build_plan_dict(s)
    rendered = build_seedream_prompt(
        d["seedream_scene"], d["seedream_style"],
        constraints=CAPYBARA_SEEDREAM_CONSTRAINTS,
    )
    assert s["scene"] in rendered
    assert "Ghibli" in rendered
    assert "no human figures" in rendered.lower()
```

- [ ] **Step 9.2: Run tests — verify they fail**

```bash
pytest test_capybara_preset.py -v -k "build_plan_dict or build_seedream_prompt_with_capybara"
```
Expected: ImportError on `build_plan_dict`.

- [ ] **Step 9.3: Implement `build_plan_dict`**

Append to `scripts/loopvid/capybara_preset.py`:

```python
def build_plan_dict(setting: dict) -> dict:
    """Assemble a plan dict that satisfies plan_schema.validate_plan_dict
    when called with extra_archetype_keys={PRESET_SENTINEL_KEY} and
    extra_motion_archetypes={PRESET_SENTINEL_KEY}."""
    return {
        "genre": CAPYBARA_GENRE,
        "mood": "cozy capybara tea",
        "music_palette": CAPYBARA_MUSIC_PALETTE,
        "music_segment_descriptors": CAPYBARA_SEGMENT_DESCRIPTORS,
        "music_bpm": CAPYBARA_MUSIC_BPM,
        "seedream_scene": f"{setting['scene']}. {setting['lighting']}. {setting['palette']}",
        "seedream_style": CAPYBARA_SEEDREAM_STYLE,
        "motion_prompts": build_motion_prompts(setting),
        "motion_archetype": PRESET_SENTINEL_KEY,
        "image_archetype_key": PRESET_SENTINEL_KEY,
    }
```

- [ ] **Step 9.4: Run tests — verify pass**

```bash
pytest test_capybara_preset.py -v
```
Expected: all pass.

- [ ] **Step 9.5: Commit**

```bash
git add scripts/loopvid/capybara_preset.py test_capybara_preset.py
git commit -m "feat(capybara_preset): build_plan_dict factory

Produces a schema-valid plan dict for any setting. Validates against
extra_archetype_keys={'capybara_tea'} (added in plan_schema). All 10
settings round-trip through validate_plan_dict cleanly."
```

---

## Task 10: Entry script — `scripts/capybara_tea_loop.py`

**Files:**
- Create: `scripts/capybara_tea_loop.py`

(No unit tests for the CLI script itself — exercised via the live smoke test in Task 11. We test the library functions, not argparse plumbing.)

- [ ] **Step 10.1: Create the entry script**

```python
#!/usr/bin/env python3
"""Capybara + Tea Loop — CLI entry.

Generates a 60-min 1280x704 looping music video with a locked Studio Ghibli
capybara+tea aesthetic. Setting varies per run (random pick or --setting).

No LLM call. Pure template via scripts/loopvid/capybara_preset.py.

Usage:
  python3 scripts/capybara_tea_loop.py
  python3 scripts/capybara_tea_loop.py --setting forest_hot_spring
  python3 scripts/capybara_tea_loop.py --duration 300 --seed 42
  python3 scripts/capybara_tea_loop.py --resume capybara-20260428T120000Z
  python3 scripts/capybara_tea_loop.py --rollback capybara-20260428T120000Z

Spec: docs/superpowers/specs/2026-04-28-capybara-tea-loop-design.md
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from pathlib import Path

# Ensure the loopvid package is importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loopvid.capybara_preset import (  # type: ignore
    CAPYBARA_GENRE,
    CAPYBARA_LTX_NEGATIVE,
    CAPYBARA_SEEDREAM_CONSTRAINTS,
    CAPYBARA_SETTINGS,
    PRESET_SENTINEL_KEY,
    build_plan_dict,
    get_setting_by_key,
    pick_setting,
)
from loopvid.cost import cost_breakdown_lines, estimate_run_cost  # type: ignore
from loopvid.orchestrator import OrchestratorConfig, run_orchestrator  # type: ignore
from loopvid.rollback import (  # type: ignore
    RollbackError, rollback_forensic, rollback_hard, rollback_with_keep,
)


DEFAULT_LTX_ENDPOINT = "1g0pvlx8ar6qns"
DEFAULT_ACE_STEP_ENDPOINT = "nwqnd0duxc6o38"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "out" / "capybara_tea"


def _autogen_run_id() -> str:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"capybara-{ts}"


def _parse_csv(s):
    if s is None:
        return None
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollback", help="Rollback the named run instead of running pipeline")
    p.add_argument("--resume", help="Resume the named run from its last completed step")
    p.add_argument("--run-id", help="Explicit run id (default: autogen capybara-<ts>)")

    valid_keys = sorted(s["key"] for s in CAPYBARA_SETTINGS)
    p.add_argument("--setting", choices=valid_keys, default=None,
                   help=f"Pick a specific scene; default: random from {len(valid_keys)} curated")
    p.add_argument("--seed", type=int, default=None,
                   help="Deterministic seed for setting selection (and downstream RNG)")
    p.add_argument("--duration", type=int, default=3600,
                   help="Target seconds (default 3600 = 60 min)")

    p.add_argument("--ltx-endpoint",
                   default=os.environ.get("RUNPOD_LTX_ENDPOINT_ID", DEFAULT_LTX_ENDPOINT))
    p.add_argument("--ace-step-endpoint",
                   default=os.environ.get("RUNPOD_ACE_STEP_ENDPOINT_ID", DEFAULT_ACE_STEP_ENDPOINT))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))

    p.add_argument("--only", help="Comma-separated steps to run only")
    p.add_argument("--skip", help="Comma-separated steps to skip")
    p.add_argument("--force", action="store_true", help="Re-run completed steps")
    p.add_argument("--dry-run", action="store_true", help="Show plan + cost, no API calls")

    p.add_argument("--max-cost", type=float, help="Abort if estimated cost exceeds this USD value")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompts")

    p.add_argument("--keep", help="--rollback: comma-separated keepers (music,image,video)")
    p.add_argument("--hard", action="store_true", help="--rollback: hard delete (with confirm)")

    return p


def cmd_rollback(args, out_dir: Path) -> int:
    run_dir = out_dir / args.rollback
    try:
        if args.hard:
            confirmed = args.yes or _confirm(f"Permanently delete {run_dir}? [y/N] ")
            rollback_hard(run_dir, confirm=confirmed)
            print(f"✓ deleted {run_dir}")
        elif args.keep:
            keep = _parse_csv(args.keep)
            failed = rollback_with_keep(run_dir, keep=keep)
            print(f"✓ kept {keep} in {run_dir}; rest moved to {failed}")
        else:
            failed = rollback_forensic(run_dir)
            print(f"✓ moved {run_dir} to {failed}")
        return 0
    except RollbackError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


def cmd_run(args, out_dir: Path) -> int:
    setting = (get_setting_by_key(args.setting) if args.setting
               else pick_setting(seed=args.seed))
    print(f"▸ setting: {setting['key']}")

    run_id = args.resume or args.run_id or _autogen_run_id()
    plan_dict = build_plan_dict(setting)

    cfg = OrchestratorConfig(
        run_id=run_id,
        run_dir=out_dir / run_id,
        genre=CAPYBARA_GENRE,
        mood=plan_dict["mood"],
        duration_sec=args.duration,
        ace_step_endpoint=args.ace_step_endpoint,
        ltx_endpoint=args.ltx_endpoint,
        runpod_api_key=os.environ.get("RUNPOD_API_KEY", ""),
        openrouter_api_key="",  # not used — preset_plan_dict skips the LLM
        replicate_api_token=os.environ.get("REPLICATE_API_TOKEN", ""),
        only=_parse_csv(args.only),
        skip=_parse_csv(args.skip),
        force=args.force,
        dry_run=args.dry_run,
        max_cost=args.max_cost,
        seedream_constraints=CAPYBARA_SEEDREAM_CONSTRAINTS,
        ltx_negative=CAPYBARA_LTX_NEGATIVE,
        extra_archetype_keys={PRESET_SENTINEL_KEY},
        extra_motion_archetypes={PRESET_SENTINEL_KEY},
        preset_plan_dict=plan_dict,
    )

    if not args.dry_run and not args.yes:
        skip = set(cfg.skip or ())
        cost = estimate_run_cost(duration_sec=cfg.duration_sec, skip=skip)
        print(f"Estimated cost: ${cost:.2f}")
        for line in cost_breakdown_lines(duration_sec=cfg.duration_sec, skip=skip):
            print(line)
        if not _confirm("Continue? [y/N] "):
            print("aborted.")
            return 1

    final = run_orchestrator(cfg)
    print(f"Done: {final}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.rollback:
        return cmd_rollback(args, out_dir)
    return cmd_run(args, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 10.2: Smoke-test the script in dry-run mode (no API calls)**

```bash
cd /root/ace-step-music-xl
python3 scripts/capybara_tea_loop.py --dry-run --setting forest_hot_spring --yes
```
Expected output (verbatim):
- `▸ setting: forest_hot_spring`
- `[DRY RUN] genre=lofi mood=cozy capybara tea duration=3600s`
- followed by cost-breakdown lines

If it raises ImportError or argparse error, fix before committing.

- [ ] **Step 10.3: Smoke-test --setting validation**

```bash
python3 scripts/capybara_tea_loop.py --setting not_a_real_key --dry-run --yes
```
Expected: argparse error listing valid choices, exit code 2.

- [ ] **Step 10.4: Smoke-test --seed reproducibility**

```bash
python3 scripts/capybara_tea_loop.py --dry-run --seed 42 --yes 2>&1 | grep "▸ setting"
python3 scripts/capybara_tea_loop.py --dry-run --seed 42 --yes 2>&1 | grep "▸ setting"
```
Expected: both invocations print the same setting key.

- [ ] **Step 10.5: Commit**

```bash
git add scripts/capybara_tea_loop.py
git commit -m "feat: scripts/capybara_tea_loop.py CLI entry

Forks loop_music_video.py into a parallel script that locks the
Ghibli capybara+tea aesthetic and varies only by curated setting.
No LLM call (preset_plan_dict short-circuits the plan step).

Spec: docs/superpowers/specs/2026-04-28-capybara-tea-loop-design.md"
```

---

## Task 11: Live 5-min smoke test (opt-in)

**Files:**
- Create: `test_capybara_tea_loop_smoke.py`

- [ ] **Step 11.1: Create the opt-in smoke test**

```python
"""Opt-in 5-min live E2E smoke test for capybara_tea_loop.py.

Skipped unless RUN_LIVE_SMOKE=1. Costs real $ on Replicate + RunPod
(~$1-2) and takes ~6-8 minutes wall-clock. Use to validate pipeline
end-to-end before unattended 60-min runs.

Usage:
    RUN_LIVE_SMOKE=1 pytest test_capybara_tea_loop_smoke.py -v -s
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

LIVE = os.environ.get("RUN_LIVE_SMOKE", "0") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="set RUN_LIVE_SMOKE=1 to run")


def test_capybara_tea_loop_5min_e2e(tmp_path):
    """5-min smoke: locked seed, locked setting → final.mp4 exists, ≈ 5 min."""
    out_dir = tmp_path / "out"
    cmd = [
        "python3", "scripts/capybara_tea_loop.py",
        "--setting", "forest_hot_spring",
        "--duration", "300",
        "--seed", "42",
        "--out-dir", str(out_dir),
        "--yes",
    ]
    result = subprocess.run(
        cmd, cwd="/root/ace-step-music-xl",
        capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, (
        f"capybara_tea_loop.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    runs = list(out_dir.glob("capybara-*"))
    assert len(runs) == 1, f"expected one run dir, got {runs}"
    run_dir = runs[0]

    final = run_dir / "final.mp4"
    assert final.exists(), f"no final.mp4 at {final}"

    # ffprobe duration check — within ±10s of 300
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(final)],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip())
    assert 290 <= duration <= 310, f"duration {duration}s out of bounds"

    # Manifest sanity check
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["steps"]["plan"]["status"] == "done"
    assert manifest["steps"]["video"]["status"] == "done"
    assert manifest["steps"]["mux"]["status"] == "done"

    # Plan.json used preset values
    plan = json.loads((run_dir / "plan.json").read_text())
    assert "Nujabes" in plan["music_palette"]
    assert plan["image_archetype_key"] == "capybara_tea"
    assert "capybara" in plan["seedream_scene"].lower()
```

- [ ] **Step 11.2: Verify the test is correctly skipped without the env var**

```bash
pytest test_capybara_tea_loop_smoke.py -v
```
Expected: `1 skipped` (the test is gated by RUN_LIVE_SMOKE).

- [ ] **Step 11.3: Final test sweep — verify no regressions across the whole loopvid suite**

```bash
pytest test_loopvid_*.py test_capybara_preset.py test_capybara_tea_loop_smoke.py -v
```
Expected: all green except smoke tests skipped.

- [ ] **Step 11.4: Commit**

```bash
git add test_capybara_tea_loop_smoke.py
git commit -m "test: opt-in 5-min E2E smoke for capybara_tea_loop

Gated by RUN_LIVE_SMOKE=1. Asserts exit 0, final.mp4 exists at
~5 min ±10s, manifest steps marked done, plan.json contains the
locked Nujabes palette + capybara_tea archetype sentinel."
```

---

## Self-review (engineer's check before declaring done)

Before submitting the final task, walk this list:

- [ ] **Spec coverage check.** Open the spec (`docs/superpowers/specs/2026-04-28-capybara-tea-loop-design.md`). For each numbered section (§3.1 through §8.4), point to a task that implements it. Gaps mean a missing task.
- [ ] **All tests passing.** Run `pytest test_loopvid_*.py test_capybara_preset.py test_capybara_tea_loop_smoke.py -v` from repo root. Green except RUN_LIVE_SMOKE-gated.
- [ ] **`loop_music_video.py` still works.** Run `python3 scripts/loop_music_video.py --dry-run --genre ambient --yes` — exit 0, prints cost breakdown.
- [ ] **No edits to `scripts/loopvid/constants.py`.** It must be byte-identical to the start of this work (the four originals — `SEEDREAM_HARD_CONSTRAINTS`, `LTX_NEGATIVE_PROMPT`, `GENRE_ARCHETYPES`, `ALLOWED_MOTION_ARCHETYPES` — all preserved).
- [ ] **`git log --oneline` shows ~11 commits**, one per task. Each commit is independently revertible.
- [ ] **Run the live smoke at least once before merging.** `RUN_LIVE_SMOKE=1 pytest test_capybara_tea_loop_smoke.py -v -s` — costs ~$1-2, takes ~6-8 minutes. If it passes, the pipeline is validated end-to-end. Note manifest path + final.mp4 location in the PR.

---

## Notes

**Hardening backlog still applies.** The 2 HIGH items from `loopvid_hardening_backlog` (non-atomic `plan.json` write, missing try/except wrap) affect this fork too. They're tracked separately and addressed in their own PR — out of scope here. Don't fix them as part of this work; confining the scope keeps the diff reviewable.

**No new dependencies.** Everything reuses the existing pinned pip stack (`requests`, `pytest`, `responses`, `runpod`, `replicate`, `ffmpeg`).

**Cost expectations per 60-min run.** Same as the existing loopvid pipeline: roughly the cost of 11 ACE-Step segments + 6 LTX clips + 1 Seedream still + ffmpeg time. Subtract the Gemini 3 Flash plan cost (~$0.01) — net slightly cheaper.

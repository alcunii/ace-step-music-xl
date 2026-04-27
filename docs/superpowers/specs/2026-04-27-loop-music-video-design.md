# Loop Music Video Generator — Design

**Date:** 2026-04-27
**Status:** Approved (brainstorm complete, ready for implementation plan)
**Owner:** Orchestrator script in `/root/ace-step-music-xl`

---

## 1. Overview

A CLI orchestrator that, given a genre and mood, produces a **1-hour 1280×704 looping music video** with naturally-paced ambient motion that does not visibly read as AI-generated.

The pipeline composes three external services:

- **OpenRouter / Gemini 3 Flash** — generates a structured plan (music prompt, still prompt, six motion prompts).
- **ACE-Step 1.5 XL serverless** (`nwqnd0duxc6o38`) — generates ~61 minutes of ambient/instrumental music as 11 stitched segments.
- **LTX-2.3 ComfyUI serverless** (`1g0pvlx8ar6qns`) — generates 6 × 7-second image-to-video clips, each conditioned on a 7-second slice of the produced music.

The 6 clips are assembled into a ~40.5-second seamlessly looping base, looped ~89 times to fill 60 minutes, and muxed with the full ACE-Step master via ffmpeg. (See §8.4 for the exact loop math.)

## 2. Goals

- Produce a 1280×704, 60:00, 24 fps `final.mp4` with H.264 video and AAC audio.
- Visual output must read as a single continuous cinematic, not as a tight repeating clip.
- Music output must read as a single continuous ambient/instrumental piece, not as 11 stitched segments.
- Pipeline must be **resumable** from any failed step without re-paying for completed expensive steps.
- Pipeline must be **rollbackable** at three levels: forensic preservation (default), selective preservation of expensive intermediates, hard delete with confirmation.
- LTX handler change must preserve the existing 480×832 portrait default exactly so the avatar-video integration cannot break.

## 3. Non-goals

- We do not target visual styles requiring complex physics, faces, hands, fast motion, or text rendering. The "looks natural" goal is achieved by avoiding what video diffusion models demonstrably get wrong.
- We do not provide a web UI in this iteration. CLI only.
- We do not implement deterministic output. Seedream 4.5 on Replicate does not accept `seed`, so reruns produce different stills. Reproducibility is via prompt + Replicate prediction ID, not seed.
- We do not redeploy or modify the LTX endpoint to support text-to-video. Image-to-video conditioning is preserved (we generate a still first).
- We do not implement parallel music + video generation. Video generation depends on the produced music (see §6 for rationale).

## 4. User-facing CLI

```bash
# Default ambient run, 60-min, auto-generated run-id
python3 scripts/loop_music_video.py --genre ambient

# Specific genre + mood + duration + run-id
python3 scripts/loop_music_video.py \
    --genre lofi \
    --mood "rainy autumn evening, cozy" \
    --duration 3600 \
    --run-id lofi-rainy-001

# Resume from any failed step (idempotent)
python3 scripts/loop_music_video.py --resume lofi-rainy-001

# Run only specific steps (prereqs must already be done)
python3 scripts/loop_music_video.py --resume lofi-rainy-001 --only video,mux

# Skip steps already complete from external sources
python3 scripts/loop_music_video.py --resume lofi-rainy-001 --skip music

# Dry run — show plan, no API calls
python3 scripts/loop_music_video.py --genre jazz --dry-run

# Cost guard
python3 scripts/loop_music_video.py --genre ambient --max-cost 5.00

# Force re-run of completed steps (with cost confirmation)
python3 scripts/loop_music_video.py --resume lofi-rainy-001 --force

# Rollback
python3 scripts/loop_music_video.py --rollback lofi-rainy-001
python3 scripts/loop_music_video.py --rollback lofi-rainy-001 --keep music
python3 scripts/loop_music_video.py --rollback lofi-rainy-001 --keep music,image
python3 scripts/loop_music_video.py --rollback lofi-rainy-001 --hard
```

## 5. Architecture

### 5.1 Top-level pipeline

```
┌─────────────────┐
│ User CLI input  │  --genre lofi --mood "..." --duration 3600 --run-id ID
└────────┬────────┘
         ▼
┌──────────────────┐    OpenRouter (gemini-3-flash-preview)
│ 1. LLM Planner   │ ──► returns plan.json (music + still + 6 motion prompts)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 2. Music gen     │  ACE-Step 1.5 XL, 11 × 360s segments,
│                  │  acrossfade qsin 30s → master.mp3 (~61 min, trimmed to 60:00)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 3. Still gen     │  Replicate Seedream 4.5, aspect_ratio="16:9" → still.png
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 4. Video gen     │  LTX-2.3 × 6, sequential, image+audio→video,
│                  │  audio = 7s slices of master.mp3, num_frames=169, 1280×704
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 5. Loop builder  │  ffmpeg: concat 6 clips with 0.25s xfades → 42s,
│                  │  loop seam fade → ~41.5s seamless base
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 6. Final mux     │  stream_loop video to 60:00, mux with master.mp3,
│                  │  -c:v copy, -c:a aac, +faststart
└────────┬─────────┘
         ▼
   final.mp4 (1280×704, 60:00, music)
```

### 5.2 Repo split

| Repo | Changes |
|---|---|
| `/root/ace-step-music-xl` | All new orchestrator code (Python). Reuses the runpod-client patterns from `ambient_eno_45min.py`. |
| `/root/ltx23-pro6000` | Handler adds `width`/`height` input params (default 480/832 for back-compat) + workflow injection into nodes 30 + 33. Docker rebuild + endpoint redeploy. |
| `/root/avatar-video` | No code changes. We port the OpenRouter + Replicate call patterns to Python. |

### 5.3 File layout

```
/root/ace-step-music-xl/
├── scripts/
│   ├── loop_music_video.py            # CLI entry — argparse + top-level orchestrator
│   ├── loopvid/
│   │   ├── __init__.py
│   │   ├── constants.py               # SEEDREAM_HARD_CONSTRAINTS, LTX_NEGATIVE_PROMPT,
│   │   │                              #   genre archetypes, ACE-Step PRESET
│   │   ├── llm_planner.py             # OpenRouter call → plan.json
│   │   ├── music_pipeline.py          # 11 segments × 360s + stitch (shared with
│   │   │                              #   ambient_eno_45min via shared modules)
│   │   ├── image_pipeline.py          # Replicate Seedream call (~80 LOC, ported from TS)
│   │   ├── video_pipeline.py          # LTX 6-clip runner (sequential, music-conditioned)
│   │   ├── mux.py                     # ffmpeg: clip concat, loop seam, final mux
│   │   ├── runpod_client.py           # shared async /run + /status (extracted from
│   │   │                              #   ambient_eno_45min — DRY)
│   │   ├── manifest.py                # run manifest dataclass + atomic JSON I/O
│   │   ├── cost.py                    # cost estimator + budget guard
│   │   └── preflight.py               # env vars, endpoint workersMax checks
│   └── ambient_eno_45min.py           # untouched at the call-site; some internals
│                                      # extracted into loopvid/ shared modules
├── scripts/smoke/
│   ├── 01_ltx_handler_back_compat.py
│   ├── 02_ltx_landscape_1280x704.py
│   └── 03_loop_music_video_5min.py
├── out/loop_video/<run-id>/
│   ├── manifest.json
│   ├── plan.json                      # validated LLM output
│   ├── plan_raw.json                  # raw LLM response cache
│   ├── run.log                        # JSONL events
│   ├── api/<step>/<call-id>.json      # request/response forensics (no base64 blobs)
│   ├── music/
│   │   ├── seg_01.mp3 … seg_11.mp3
│   │   ├── seg_NN.json                # sidecars: prompt, seed, runpod job id
│   │   └── master.mp3                 # stitched, ~61 min
│   ├── still.png                      # Seedream output
│   ├── video/
│   │   ├── audio_chunks/clip_N.mp3    # 7s slices of master.mp3
│   │   ├── clip_01.mp4 … clip_06.mp4  # raw LTX outputs (with throwaway audio)
│   │   ├── concat_42s.mp4             # adjacent xfades, no loop seam yet
│   │   ├── loop_seamed.mp4            # ~41.5s seamless loopable base
│   │   └── video_60min.mp4            # stream-looped, no audio
│   └── final.mp4                      # final 1280×704 60-min artifact
└── (tests at repo root, matching existing convention — test_handler.py,
    test_endpoint.py, test_ambient_eno_45min.py)
    ├── test_loopvid_constants.py
    ├── test_loopvid_manifest.py
    ├── test_loopvid_cost.py
    ├── test_loopvid_plan_schema.py
    ├── test_loopvid_llm_planner.py
    ├── test_loopvid_image_pipeline.py
    ├── test_loopvid_video_pipeline.py
    ├── test_loopvid_music_pipeline.py
    ├── test_loopvid_mux.py
    ├── test_loopvid_loop_build.py
    ├── test_loopvid_orchestrator_e2e.py
    └── test_loopvid_rollback.py
```

All test files use the `test_loopvid_` prefix so they group together via `pytest test_loopvid_*.py` while not colliding with existing `test_handler.py` / `test_endpoint.py` / `test_workflow.py`.

## 6. Why music-first ordering (not parallel)

LTX-2.3 is an audio-VAE i2v model — the audio is **a generation input**, not just a frame-count source. We use this on purpose: by feeding each LTX clip its corresponding 7-second music slice as `audio_base64`, the resulting motion subtly tracks the music's amplitude and harmonic envelope. This is the single biggest contributor to the "doesn't look AI-generated" goal.

Consequence: video generation cannot start until music is stitched. Wall-clock is `music_time + video_time`, not `max(music, video)`. We accept this — the naturalness win is large.

## 7. Phase 0 — LTX handler change

### 7.1 Handler diff (~15 LOC in `/root/ltx23-pro6000/handler.py`)

```python
# New defaults (preserve current portrait behavior)
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 832

# New node IDs (in addition to existing constants)
NODE_RESIZE_IMAGE = "33"             # ResizeImageMaskNode
NODE_RESIZE_LONGER_EDGE = "30"       # ResizeImagesByLongerEdge

def inject_inputs(workflow, ..., width, height):
    # ...existing injections...
    if NODE_RESIZE_IMAGE in wf:
        wf[NODE_RESIZE_IMAGE]["inputs"]["resize_type.width"] = width
        wf[NODE_RESIZE_IMAGE]["inputs"]["resize_type.height"] = height
    if NODE_RESIZE_LONGER_EDGE in wf:
        wf[NODE_RESIZE_LONGER_EDGE]["inputs"]["longer_edge"] = max(width, height)

def handler(job):
    # ...existing parsing...
    width = int(job_input.get("width", DEFAULT_WIDTH))
    height = int(job_input.get("height", DEFAULT_HEIGHT))
    if width % 32 != 0 or height % 32 != 0:
        return {"error": f"width and height must be multiples of 32 (got {width}x{height})"}
```

### 7.2 Back-compat guarantee

Callers that omit `width` and `height` get **byte-identical workflow JSON** to the current v4 deployment. The avatar-video integration never sets these fields, so its behavior is unchanged.

This is enforced by `test_handler_workflow_json_byte_identical_when_no_width_height` — the strongest sentinel test in the suite.

### 7.3 Deploy + rollback strategy

Using RunPod template versions (the endpoint is currently on template `qo92k71b0g` v4):

```
1. Build new Docker image:    ltx23-pro6000:v5
2. Create template v5
3. Pre-flip smoke tests:
   - Spin up a one-off pod with v5
   - Test 1: payload without width/height → must produce 480×832 video
   - Test 2: payload with width=1280 height=704 → must produce 1280×704 video
4. Flip endpoint 1g0pvlx8ar6qns to template v5
5. Post-flip smoke tests against the live endpoint (same two payloads)
6. Run /root/avatar-video/src/scripts/test-episode-avatar.ts against the new endpoint
7. ROLLBACK trigger: any of the post-flip tests fail
   → rp endpoint update 1g0pvlx8ar6qns --template-version 4
   → ~10 sec recovery, no in-flight job loss
```

## 8. Component contracts

### 8.1 LLM Planner (`loopvid/llm_planner.py`)

**Function:** `plan(genre: str, mood: str, llm_seed: int) -> Plan`

**Output schema:**

```python
@dataclass(frozen=True)
class Plan:
    genre: str                              # echoed input
    mood: str                               # echoed input

    # Music side
    music_palette: str                      # locked palette, ~400 chars
    music_segment_descriptors: list[dict]   # 11 entries: [{"phase": str, "descriptors": str}]
    music_bpm: int                          # for sidecar metadata

    # Image side
    seedream_scene: str                     # scene description (no constraints — appended later)
    seedream_style: str                     # film stock, lighting, color grade

    # Video side (6 motion prompts following breathing-arc pattern)
    motion_prompts: list[str]               # exactly 6 entries; clip[0] and clip[5] = rest-state
    motion_archetype: str                   # one of: rain, candle, mist, smoke, dust, snow
```

**Call configuration:**

- Model: `google/gemini-3-flash-preview`
- `response_format`: JSON Schema matching `Plan`
- Temperature: 0.7
- Retries: up to 3 on schema-validation failure (validation error appended to system prompt for retry); up to 3 on 5xx with exponential backoff (1s, 4s, 16s).
- Caches raw response at `plan_raw.json` so schema-failure retries never re-bill the LLM.

**System prompt** lives in `loopvid/constants.py` (~600 words). It enforces:
- 11-segment music plan with breathing arc (3 settle → 1 hold → 5 deepen-and-release → 2 dissolve)
- Image archetype selection from a closed list (`rainy_window_desk`, `mountain_ridge_dusk`, `candle_dark_wood`, `dim_bar_booth`, `study_window_book`, `observatory_dome`)
- 6 motion prompts where clip 1 and clip 6 depict the still's rest state (loop seam invariant)
- No mention of text, signs, faces, hands, fingers, mirrors, scene cuts, fast motion in any field

### 8.2 Music pipeline (`loopvid/music_pipeline.py`)

`run(plan: Plan, run_dir: Path, force: bool) -> Path` → returns `master.mp3`.

- 11 × 360s segments (vs. 7 × 420s in `ambient_eno_45min.py`). 360s is the safe VRAM ceiling per the operational-gotchas memory.
- Total before crossfades: 11 × 360 = 3960s. After 10 × 30s `acrossfade qsin`: 3660s ≈ 61 min. Final mux trims to exactly 3600s.
- Per-segment payload identical to `ambient_eno_45min.build_payload`:
  - Same PRESET (inference_steps=64, guidance_scale=8.0, shift=3.0, use_adg=True, cfg_interval_start=0.0, cfg_interval_end=1.0, infer_method="ode")
  - `audio_format="mp3"` per the /job-done size-cap mitigation
  - `thinking=False`
  - Prompt = `plan.music_palette + ", " + plan.music_segment_descriptors[n].descriptors`
- Atomic write per segment: `seg_NN.mp3.tmp` → `os.replace` → `seg_NN.mp3`. JSON sidecar same pattern.
- Resume: scans for missing/corrupt segments, regenerates only those.
- Stitch: existing `ffmpeg acrossfade qsin` at 192 kbps libmp3lame.

**Refactor boundary:** the runpod-client logic (async `/run`, poll `/status`, transient 404 tolerance, retry-on-OOM) and the ACE-Step PRESET constants are extracted from `ambient_eno_45min.py` into `loopvid/runpod_client.py` and `loopvid/constants.py`. Genre-specific bits in `ambient_eno_45min.py` (its hardcoded `LOCKED_PALETTE` and `SEGMENT_DESCRIPTORS`) **stay inline** in that script — `ambient_eno_45min.py` remains a single-purpose CLI for the Eno-style ambient piece. The new `loop_music_video.py` differs by getting palette + descriptors from the LLM at run time, not from a constant.

### 8.3 Image pipeline (`loopvid/image_pipeline.py`)

`run(plan: Plan, run_dir: Path) -> Path` → returns `still.png`.

```python
body = {
    "input": {
        "prompt": build_seedream_prompt(plan.seedream_scene, plan.seedream_style),
        "aspect_ratio": "16:9",
    }
}
# Notably absent: seed (422 error), negative_prompt (not in schema), image_input (we want t2i)
```

- Endpoint: `POST https://api.replicate.com/v1/models/bytedance/seedream-4.5/predictions`
- Auth: `Bearer $REPLICATE_API_TOKEN`
- Poll interval 3s, timeout 600s.
- On `succeeded`: download output URL with axios-equivalent (`requests` with stream + arraybuffer), atomic-write to `still.png`.
- Reproducibility unit: prompt string + Replicate `prediction_id` (recorded in `manifest.json`), not seed.
- Native output dimensions are determined by Replicate (typically 1024×576 or 1920×1080 for 16:9). LTX node 33 will resize to 1280×704 on its end, so the exact Seedream output size does not matter.

### 8.4 Video pipeline (`loopvid/video_pipeline.py`)

`run(plan: Plan, run_dir: Path, music_master: Path, still: Path) -> Path` → returns `loop_seamed.mp4`.

```
NUM_FRAMES = 169                                # (169-1) % 8 == 0
FPS        = 24
CLIP_SEC   = NUM_FRAMES / FPS                   # = 7.0417s exact

For i in 1..6:
  1. ffmpeg slice master.mp3 [(i-1)*CLIP_SEC : i*CLIP_SEC]
        → audio_chunks/clip_i.mp3 (7.0417s, atomic write)
  2. payload = {
       image_base64:     base64(still.png),
       audio_base64:     base64(audio_chunks/clip_i.mp3),
       prompt:           plan.motion_prompts[i-1],
       negative_prompt:  LTX_NEGATIVE_PROMPT,           # constant
       num_frames:       NUM_FRAMES,                    # 169 — closest to 7s under (n-1)%8==0
       fps:              FPS,
       seed:             stable_per_clip(run_id, i),    # for resume reproducibility
       width:            1280,
       height:           704,
     }
  3. POST /run → poll /status → save clip_NN.mp4 atomically
```

- Audio chunks are sliced to exactly `NUM_FRAMES / FPS` = 7.0417s so audio length matches video length and the LTX `MathExpression` node 62 result is consistent with our explicit `num_frames` override.
- Sequential, not parallel — keeps one warm worker, no cold-start tax per clip.
- Resumable: skip clips with existing `clip_NN.mp4`.
- After all 6 clips: ffmpeg concat with 0.25s `xfade` between adjacent clips → `concat_42s.mp4`. Total xfade consumption: 5 × 0.25s = 1.25s.
- Loop seam fade: last 0.5s of `concat_42s` mixed with first 0.5s → `loop_seamed.mp4`.
- Output is video-only (`-an`) — LTX's audio output is throwaway.

**Loop seam math (exact):**
- Raw concat: 6 × 7.0417s = 42.25s
- After 5 inter-clip xfades: 42.25 − 5 × 0.25 = 41.00s for `concat_42s.mp4` (filename retained for grep-ability; actual is 41.00s)
- After loop-seam fade: 41.00 − 0.5 = **40.50s for `loop_seamed.mp4`**
- 3600s / 40.50s = **88.89 reps to fill an hour** (`-stream_loop -1 -t 3600` handles the fractional final loop)

### 8.5 Mux pipeline (`loopvid/mux.py`)

`run(loop_seamed: Path, music_master: Path, run_dir: Path, target_sec: int) -> Path` → returns `final.mp4`.

```bash
# Step 1: stream-loop the seamed video to fill exactly 60:00
ffmpeg -stream_loop -1 -i loop_seamed.mp4 -t 3600 \
       -c:v copy -an video_60min.mp4

# Step 2: trim master.mp3 to exactly 60:00
ffmpeg -i master.mp3 -t 3600 -c:a copy music_60min.mp3

# Step 3: final mux (no re-encode of video; re-encode mp3→aac for MP4 universal compat)
ffmpeg -i video_60min.mp4 -i music_60min.mp3 \
       -c:v copy -c:a aac -b:a 192k -shortest \
       -movflags +faststart final.mp4
```

`-c:v copy` avoids re-encoding the H.264 stream (sub-minute mux, no quality loss). The audio is re-encoded MP3→AAC — MP3-in-MP4 is technically allowed but breaks on some hardware/web players, so AAC is worth the one-time re-encode for universal compatibility. `+faststart` puts the moov atom at the head for streaming-friendly output.

## 9. Loop seam strategy & natural-look design rules

The visual goal is "single 60-min unbroken cinematic" rather than "tight loop repeating 89 times." Three concentric strategies achieve this:

1. **Same still + 6 unified motion prompts** (clips share visual identity).
2. **Music-conditioned motion** (LTX audio VAE makes motion track the actual music, so visual rhythm matches musical rhythm subconsciously).
3. **Breathing-arc motion design** (clip 1 and clip 6 depict the still's rest state, so the loop seam joins two stillness moments).
4. **Three concentric crossfades**:
   - 0.25s `xfade` between each adjacent clip in the 42s base
   - 0.5s `xfade` at the loop seam (last 0.5s of base ↔ first 0.5s of base)
   - `stream_loop -1` to fill 60 minutes

Combined, the viewer sees subtle music-reactive ambient motion that returns to rest every 40.5 seconds — invisible because the rest state matches the still's neutral pose.

## 10. Constants & prompt design

### 10.1 Seedream still prompt (positive form, no negative_prompt field)

The LLM produces only `seedream_scene` and `seedream_style`. The orchestrator appends a **constant hard-constraints block** to every Seedream call:

```python
SEEDREAM_HARD_CONSTRAINTS = (
    "Clean composition with absolutely no text, letters, numbers, words, "
    "captions, watermarks, signage, signs, neon signs, screen text, book "
    "text, handwriting, signatures, logos, brand marks, or any visible "
    "writing of any kind. No people, no human figures, no faces, no hands, "
    "no fingers. Single focal point, uncluttered background, no mirrors, "
    "no reflective glass surfaces, no transparent objects. Photographic "
    "realism, no AI-style artifacts, no oversaturation, no fake bokeh, "
    "no HDR look. Static medium-wide shot, fixed camera, no pan, no zoom, "
    "no parallax. 16:9 cinematic widescreen."
)
```

Phrased as positive-form requirements (`"with absolutely no..."`) — empirically more effective than separated negative lists for foundation image models that lack a `negative_prompt` field.

### 10.2 LTX video negative prompt

The handler default is preserved; the orchestrator appends the known-weakness block:

```python
LTX_NEGATIVE_PROMPT = (
    # Handler default
    "blurry, low quality, still frame, frames, watermark, overlay, "
    "titles, has blurbox, has subtitles, "
    # Known LTX-2.3 weaknesses
    "text, letters, numbers, words, captions, watermark, logo, "
    "signage, signs, neon text, screen text, on-screen text, "
    "hands, fingers, face, faces, eyes, mouth, lips, "
    "fast motion, sudden movement, rapid motion, camera shake, "
    "camera pan, camera zoom, dolly, tracking shot, handheld, "
    "hard cut, scene change, scene transition, jump cut, "
    "reflections, mirror reflections, glass refraction, "
    "crowd, multiple subjects, multiple animated objects, "
    "splashing water, pouring liquid, explosion, fireworks, sparks, "
    "animal legs, paws, wings, fur detail, "
    "frame stutter, ghosting, motion smear, double exposure, "
    "morphing, warping, melting, glitch"
)
```

### 10.3 Genre archetype list

Closed set the LLM picks from. Each archetype is a (visual_archetype, anchored_foreground, ambient_motion_sources) triple defined in `loopvid/constants.py`:

| Archetype key | Visual | Anchored foreground | Ambient motion sources |
|---|---|---|---|
| `rainy_window_desk` | Desk-by-rainy-window, dusk | Notebook + lamp + plant | Rain on window, lamp flicker, steam from cup |
| `mountain_ridge_dusk` | Mountain ridge at golden hour | Stone cairn or lone tree | Mist drift, distant cloud cycle, grass sway |
| `candle_dark_wood` | Single candle on dark wood | Candle + brass holder, centered | Flame flicker, smoke wisp, dust motes in beam |
| `dim_bar_booth` | Dim bar booth, bokeh windows | Whiskey glass + brass lamp | Cigar smoke curl, bokeh shimmer, slow ceiling-fan glint |
| `study_window_book` | Window-lit study with sheet music | Open book + quill + window frame | Curtain breath, page edge, sun-shaft dust motes |
| `observatory_dome` | Observatory dome interior at night | Telescope silhouette | Star drift, monitor glow pulse, dust |

The LLM may customize specific objects, lighting, color grade — never invent a new archetype.

## 11. Manifest & data flow

### 11.1 Manifest schema (`out/loop_video/<run-id>/manifest.json`)

```json
{
  "run_id": "lofi-rainy-001",
  "schema_version": 1,
  "created_at": "2026-04-27T05:30:12Z",
  "last_updated": "2026-04-27T06:14:33Z",
  "args": {"genre": "lofi", "mood": "rainy autumn evening, cozy",
           "duration_sec": 3600, "force": false},
  "endpoints": {"ltx": "1g0pvlx8ar6qns", "ace_step": "nwqnd0duxc6o38"},
  "comment_endpoints": "Endpoint IDs are read from env vars (RUNPOD_LTX_ENDPOINT_ID, RUNPOD_ACE_STEP_ENDPOINT_ID) at run time and snapshotted here for forensics — they are not hardcoded in the orchestrator.",
  "steps": {
    "plan":         {"status": "done|in_progress|pending|failed",
                     "committed_at": "...", "attempts": 1},
    "music":        {"status": "...", "segments_done": [1,2,3,4,5],
                     "segments_pending": [6,7,8,9,10,11],
                     "master_committed": false, "attempts": 1},
    "image":        {"status": "...", "committed_at": "...",
                     "prediction_id": "abc123", "attempts": 1},
    "video":        {"status": "...", "audio_chunks_done": [],
                     "clips_done": [], "attempts": 1},
    "loop_build":   {"status": "...", "concat_committed": false,
                     "loop_seamed_committed": false},
    "mux":          {"status": "...", "video_60min_committed": false,
                     "final_committed": false}
  },
  "cost_estimate_usd": 1.42,
  "cost_actual_usd": 0.83,
  "failures": [
    {"step": "music_seg_6", "error": "Insufficient free VRAM",
     "ts": "...", "attempts": 2, "next_action": "retry on different worker"}
  ]
}
```

The manifest is rewritten atomically (`.tmp` → `os.replace`) after every successful step commit. If the orchestrator crashes between two steps, the manifest accurately reflects the world.

## 12. Error handling, rollback, and resume

### 12.1 Atomic writes everywhere

```python
def commit_artifact(final_path: Path, write_fn):
    """Write to .tmp then atomically rename. Crash-safe."""
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    write_fn(tmp)
    os.replace(tmp, final_path)
```

Canonical files are always complete. `.tmp` files are stale-orphans, ignored on resume, regenerated.

### 12.2 Failure-handling matrix

| Step | Failure mode | Cost | Retry policy | Fallback |
|---|---|---|---|---|
| Plan (LLM) | Schema validation fail | ~$0.001 | Up to 3, append validation error to system prompt | Fail run, log raw response |
| Plan (LLM) | OpenRouter 429/5xx | — | 3 with exp backoff (1s, 4s, 16s) | Fail run |
| Music seg | RunPod 404 mid-poll (worker churn) | — | Up to 6 transient 404s (matches `ambient_eno_45min`) | — |
| Music seg | "Insufficient free VRAM" | ~$0.20 | 3 retries (different worker often has space) | Drop seg duration 360→300s, regen, flag in manifest |
| Music seg | `/job-done` 502 (size cap) | ~$0.20 | Should not happen with MP3 — abort with clear error | Investigate handler |
| Music seg | Timeout > 1800s | ~$0.20 wasted | 1 retry | Fail step, allow `--resume --only music` |
| Image | Replicate 5xx | ~$0.01 | 3 retries | Fail step |
| Image | Output URL inaccessible | ~$0.01 | 2 retries on download | Fail step |
| Video clip | LTX timeout > 1200s | ~$0.05 | 1 retry | Fail step, partial clips kept |
| Video clip | Returns `error` field | ~$0.05 | 0 retries (likely deterministic) | Fail step, dump diagnostic |
| ffmpeg | Non-zero exit | — | 0 retries | Fail step, log full stderr |
| Pre-flight | ACE-Step `workersMax=0` | $0 | 0 | Fail fast: "Run: rp endpoint update nwqnd0duxc6o38 --workers-max 1" |
| Pre-flight | Missing API key | $0 | 0 | Fail fast with which env var |

### 12.3 Resume semantics

```bash
python3 scripts/loop_music_video.py --resume <run-id>
```

1. Load `manifest.json`. If absent → error.
2. If `mux.status == "done"` → print "Already complete: out/loop_video/<run-id>/final.mp4" and exit 0.
3. Walk steps in order. For each: skip `done`, resume `in_progress` from per-step state (e.g., music regenerates only missing segments), run `pending` normally.
4. `--force` ignores `done` status (with cost-confirmation prompt).
5. `--only step1,step2` runs only those steps (prereqs must be `done`).
6. `--skip step1` skips that step (must already be `done` or external).

### 12.4 Rollback — three levels

```bash
# Level 1: forensic preservation (default)
python3 scripts/loop_music_video.py --rollback <run-id>
# → mv out/loop_video/<run-id>/ → out/loop_video/<run-id>.failed-<UTC>/
# → run-id is now free for a fresh attempt
# → all artifacts preserved; --resume on the .failed copy still works

# Level 2: selective preservation
# All --keep variants implicitly preserve plan.json + plan_raw.json (logical prereq
# for every downstream step; cost is ~$0.001 but losing it forces a re-plan).
python3 scripts/loop_music_video.py --rollback <run-id> --keep music
# → preserves plan.json + master.mp3 + segment files in a clean <run-id>/ dir
# → rewrites manifest.json to mark plan + music as done, everything downstream pending
# → moves everything else to <run-id>.failed-<UTC>/
# → --resume <run-id> picks up from the image step
python3 scripts/loop_music_video.py --rollback <run-id> --keep music,image
# → preserves plan.json + master.mp3 + still.png; only video+mux re-run

# Level 3: hard delete
python3 scripts/loop_music_video.py --rollback <run-id> --hard
# → confirmation prompt: "Permanently delete <run-id>? [y/N]"
# → only after y: rm -rf out/loop_video/<run-id>/
```

Rollback uses `os.rename` of the directory (atomic on the same filesystem) — either the move happened or it did not. No half-rolled-back state.

### 12.5 LTX deploy rollback (Phase 0)

| Trigger | Action | Recovery time |
|---|---|---|
| Pre-deploy smoke test fails | Don't flip endpoint. Fix locally, rebuild. | N/A — production unaffected |
| Post-flip smoke test fails | `rp endpoint update 1g0pvlx8ar6qns --template-version 4` | ~10 sec |
| avatar-video integration test fails post-flip | Same revert | ~10 sec |
| Live production 5xx within first hour | Same revert + investigate | ~10 sec + investigation |

## 13. Cost guard

```bash
python3 scripts/loop_music_video.py --genre ambient --max-cost 5.00
```

Before any paid API call, the orchestrator prints an estimate and prompts:

```
This run will cost approximately $3.20:
  - LLM (Gemini 3 Flash):           $0.001
  - Image (Seedream 4.5):           $0.03
  - Music (ACE-Step, 11 segments):  $2.50
  - Video (LTX, 6 clips):           $0.66
Continue? [y/N]
```

On `--resume`, only counts cost of remaining steps. If estimate exceeds `--max-cost`, abort without prompt. Per-second pricing constants live in `loopvid/cost.py`:

- RTX 4090 ≈ $0.00031/s (ACE-Step worker)
- RTX Pro 6000 ≈ $0.00076/s (LTX worker)
- Seedream 4.5 ≈ $0.03/image
- Gemini 3 Flash ≈ $0.0015/1k tokens

Cost numbers update in `manifest.json` as actual call durations come back.

## 14. Logging contract

- **stdout**: human-readable progress (`✓ plan committed`, `▸ music seg 3/11 (started, ETA 4 min)`, `✗ music seg 6 retry 2/3 (Insufficient VRAM)`)
- **`<run-id>/run.log`**: structured JSON Lines, one per significant event, `jq`-friendly
- **`<run-id>/api/<step>/<call-id>.json`**: every API request/response (excluding base64 blobs) for forensics

## 15. Testing strategy

### 15.1 Test pyramid

- ~80 unit tests (pure functions, no I/O)
- ~25 integration tests (per-module, mocked HTTP + real fs + real ffmpeg)
- ~10 E2E tests (full orchestrator, all HTTP mocked)
- 3 live smoke tests (manual, paid, never automatic)

Coverage target: 80%+. Estimated runtime: < 60s for the full mocked suite.

### 15.2 TDD order

1. **Phase 0 (LTX handler):** `test_handler_resolution.py` — including `test_handler_workflow_json_byte_identical_when_no_width_height` as the back-compat sentinel.
2. **Phase 1 (pure functions):** `test_constants.py`, `test_manifest.py`, `test_cost.py`, `test_plan_schema.py`.
3. **Phase 2 (pipelines, mocked HTTP):** `test_llm_planner.py`, `test_image_pipeline.py`, `test_video_pipeline.py`, `test_music_pipeline.py`.
4. **Phase 3 (integration, real ffmpeg):** `test_mux.py`, `test_loop_build.py`.
5. **Phase 4 (E2E + rollback):** `test_orchestrator_e2e.py`, `test_rollback.py`.

### 15.3 Mock vs. real

| Thing | Test treatment | Why |
|---|---|---|
| `requests` HTTP calls | Mocked | Speed, determinism, no API spend |
| `subprocess.run` for ffmpeg | Real | ffmpeg is fast; catches broken commands |
| Filesystem | Real (`tmp_path` fixture) | Atomic-write semantics need real `os.replace` |
| Time | Mocked (`freezegun`) for retry-backoff tests | Reproducibility |
| Manifest JSON I/O | Real | Tests our actual code |
| Random seeds | Stable via fixed-seed args | Reproducibility |

### 15.4 Live smoke tests (`scripts/smoke/`, manual only)

```
01_ltx_handler_back_compat.py    — payload omits width/height, expects 480×832, ~$0.10
02_ltx_landscape_1280x704.py     — explicit landscape test, ~$0.10
03_loop_music_video_5min.py      — full pipeline at 5 min duration, ~$0.40, ~10 min wall
```

The avatar-video TS test (`/root/avatar-video/src/scripts/test-episode-avatar.ts`) is reused as a deploy-time canary, not re-implemented.

### 15.5 What we explicitly do NOT test

- ffmpeg's correctness (it's well-trodden)
- ACE-Step or LTX model quality (model-side concerns)
- LLM output quality (would require an LLM judge)
- Performance benchmarks (wall-clock dominated by remote inference)

## 16. Open questions / future work

- **Seedream output dimensions**: not specified by us; we accept whatever Replicate returns for `aspect_ratio: "16:9"`. LTX node 33 resizes regardless. If Seedream's output is significantly less than 1280×704, we may want to revisit. (Verified at first smoke test.)
- **Music BPM in the plan**: included in `Plan` for sidecar metadata but not used for video timing. Could later be used to pick `num_frames` per clip such that 7s contains an integer number of beats — would tighten the music-video sync further.
- **Seam fade duration**: 0.5s is a reasonable default. If the loop seam still reads on visual inspection, parameterize and try 1.0s.
- **Genre archetypes**: closed set of 6 today. Adding new archetypes is a constants-only change.
- **Web UI**: out of scope for this iteration. The CLI shape is stable enough that a future thin wrapper (Next.js + RunPod-style polling) could call into it.

---

## Appendix A — Sources of design decisions

- ACE-Step XL 45-min orchestrator pattern (memory: `ace_step_xl_45min_orchestrator.md`)
- RunPod operational gotchas (memory: `runpod_acestep_operational_gotchas.md`) — `/job-done` 20MB cap, VRAM ceiling, MP3-not-FLAC for segments
- ACE-Step XL-base official preset (memory: `ace_step_xl_base_official_preset.md`) — verbatim INFERENCE.md Example 9
- Long-form prompt design pattern (memory: `prompt_design_pattern_for_long_form.md`) — locked palette + per-segment evolution
- LTX-2.3 handler at `/root/ltx23-pro6000/handler.py` (resolution constraint discovery in workflow_api.json nodes 30 + 33)
- Replicate Seedream 4.5 client at `/root/avatar-video/src/pipeline/lib/replicate-client.ts:63-75` — confirmed no `seed`, no `negative_prompt`, no `image_input` for our use
- Existing ambient orchestrator at `/root/ace-step-music-xl/scripts/ambient_eno_45min.py` — reference implementation we extract shared modules from

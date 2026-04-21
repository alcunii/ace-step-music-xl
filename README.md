# ace-step-music-xl

RunPod serverless endpoint for **ACE-Step 1.5 XL** (`acestep-v15-xl-base`, 4B DiT).
Unified handler supporting six task types: `text2music`, `cover`, `repaint`, `extract`, `lego`, `complete`.

## Deployment

- **GPU:** RTX 4090 24GB, RunPod data center **US-IL-1**
- **Weights:** 35 GB RunPod network volume (`52nyie2f6k`) mounted at `/runpod-volume`
- **Image:** `dmrabh/ace-step-music-xl:latest`
- **Pipeline:** push to `main` → GitHub Actions → Docker Hub → RunPod `saveTemplate` mutation

See `docs/superpowers/specs/2026-04-20-ace-step-xl-serverless-design.md` for the
full design, and `docs/superpowers/plans/2026-04-20-ace-step-xl-serverless.md` for
the step-by-step implementation plan.

## Quick start

```bash
pip install -r requirements-test.txt
pytest test_handler.py test_workflow.py -v   # unit tests, no GPU
export RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=...
python test_endpoint.py --all                 # smoke test all 6 tasks
```

## Input schema

| Field | Type | Default | Notes |
|---|---|---|---|
| `task_type` | string | `text2music` | `text2music` / `cover` / `repaint` / `extract` / `lego` / `complete` |
| `prompt` | string | — | Required for all tasks except `extract` (which uses `instruction`) |
| `src_audio_url` or `src_audio_base64` | string | — | Required for all audio-input tasks; `https://` URL or base64 |
| `instruction` | string | — | Required for `extract` |
| `repainting_start` / `repainting_end` | float | — | Required for `repaint` / `lego` (seconds; `-1` for end of file) |
| `duration` | float | `30` | Clamped 10–600 |
| `inference_steps` | int | `50` | XL recommended |
| `guidance_scale` | float | `7.0` | XL uses CFG |
| `batch_size` | int | `1` | Clamped to `MAX_BATCH_SIZE` (default 2) |
| `seed` | int | `-1` | `-1` = random |
| `audio_format` | string | `mp3` | `mp3` / `wav` / `flac` |
| `instrumental` | bool | `true` | When `true`, forces `lyrics="[Instrumental]"` |
| `lyrics` | string | `""` | Used only when `instrumental=false` |

## Output

```json
{
  "audio_base64": "<base64 audio bytes>",
  "format": "mp3",
  "duration": 30.0,
  "seed": 12345,
  "sample_rate": 48000,
  "task_type": "text2music"
}
```

## Scripts

Standalone client-side orchestrators for generation patterns beyond a single
`/runsync` call.

- `scripts/bruno_mars_style_midnight_gold.py` — 3-minute original funk-pop
  song via `text2music`.
- `scripts/cover_afterlife_armstrong.py` — reinterpret a local MP3 as a
  different style via `task_type=cover`.
- `scripts/ambient_eno_45min.py` — 7-segment × 420 s Eno-style ambient run
  stitched locally with ffmpeg crossfade to a ~46-minute FLAC. Resumable,
  pinnable via `--pin-seeds-from`, idempotent per segment. See
  `docs/superpowers/specs/2026-04-21-ambient-eno-45min-design.md`.

## Environment variables

See `.env.example`.

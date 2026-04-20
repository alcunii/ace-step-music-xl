# ACE-Step 1.5 XL — RunPod Serverless Endpoint Design

**Status:** Approved for implementation
**Date:** 2026-04-20
**Project directory:** `/root/ace-step-music-xl`
**Upstream model:** [ACE-Step/acestep-v15-xl-base](https://huggingface.co/ACE-Step/acestep-v15-xl-base)
**Upstream source:** https://github.com/ace-step/ACE-Step-1.5
**Reference implementation (turbo):** `/root/ace-step-music`

---

## 1. Goals

Deploy ACE-Step 1.5 XL (`acestep-v15-xl-base`, 4B DiT) as a RunPod serverless endpoint
that exposes **all six task types** behind a single unified handler: `text2music`,
`cover`, `repaint`, `extract`, `lego`, `complete`. Deployment pipeline mirrors the
existing turbo deployment: GitHub Actions → Docker Hub → RunPod template update.

## 2. Non-goals

- Multi-region deployment. Endpoint lives in EU-SE-1 only; network volume is
  region-locked so future expansion requires per-region volume duplication.
- Private-auth `src_audio` ingestion. Handler only accepts `https://` URLs; caller
  is responsible for any access control (public URLs, GitHub releases are fine for now).
- Fine-tuning, LoRA merging, or prompt-caption-training workflows.
- Bakeing weights into the image; model lives on a network volume.
- Other XL variants (`xl-sft`, `xl-turbo`) — volume sizing leaves headroom to add
  them later but they are out of scope for initial deployment.

## 3. Infrastructure

### 3.1 Compute

- **GPU:** NVIDIA A40 (48 GB VRAM, Ampere sm_86)
- **Data center:** EU-SE-1 (Stockholm, Sweden)
- **Serverless config:** `workersMin=0`, `workersMax=3`, `idleTimeout=5s`, FlashBoot enabled

### 3.2 Storage

- **Network volume:** `ace-step-xl-weights`, **35 GB**, region EU-SE-1
- **Mount point:** `/runpod-volume`
- **Layout:**
  ```
  /runpod-volume/
    checkpoints/
      acestep-v15-xl-base/   # DiT weights (~15–19 GB)
      main-model/            # shared LM + VAE + embeddings (~4–5 GB)
        acestep-5Hz-lm-1.7B/
        vae/
        embedding/
  ```
- **Headroom:** ~10 GB for HF cache churn and a future second DiT variant
- **Cost:** ~$2.45/mo flat

### 3.3 Container image

- **Base:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Same stack as the turbo deployment** — torch 2.4.0+cu124, SDPA attention,
  no flash-attn, no torchao. A40 sm_86 means no rebuild needed vs. turbo.
- **Image size target:** ~5 GB (weights not baked in)
- **Registry:** Docker Hub, `dmrabh/ace-step-music-xl:{latest,<sha>}`

## 4. Request/Response contract

### 4.1 Common input fields (all task types)

| Field | Type | Default | Notes |
|---|---|---|---|
| `task_type` | string | `"text2music"` | One of: `text2music`, `cover`, `repaint`, `extract`, `lego`, `complete` |
| `audio_format` | string | `"mp3"` | `mp3` / `wav` / `flac` |
| `seed` | int | `-1` | `-1` = random |
| `batch_size` | int | `1` | Clamped to `[1, ACESTEP_MAX_BATCH_SIZE]`, default max 4 |
| `inference_steps` | int | `50` | XL recommended. Clamped `[1, 200]` |
| `guidance_scale` | float | `7.0` | XL uses CFG |
| `shift` | float | `1.0` | Timestep shift |

### 4.2 Task-specific fields

Shorthand: `src_audio` means "exactly one of `src_audio_url` or `src_audio_base64`".

| task_type | Required | Optional | LM behavior |
|---|---|---|---|
| `text2music` | `prompt` | `duration` (default 30, clamped 10–600), `lyrics`, `instrumental` (default true), `bpm`, `key_scale`, `time_signature`, `thinking` (default true), `lm_temperature` (default 0.85) | Used if `thinking=true` |
| `cover` | `src_audio`, `prompt` | `audio_cover_strength` (default 0.3) | Auto-skipped |
| `repaint` | `src_audio`, `prompt`, `repainting_start` (sec), `repainting_end` (sec or `-1`) | `audio_cover_strength`, `lyrics`, `instrumental` | Auto-skipped |
| `extract` | `src_audio`, `instruction` | — | Auto-skipped |
| `lego` | `src_audio`, `prompt`, `repainting_start`, `repainting_end` | `audio_cover_strength`, `thinking` | Used if `thinking=true` |
| `complete` | `src_audio`, `prompt` | `thinking`, `duration` (total target) | Used if `thinking=true` |

**Lyrics/instrumental coupling:** When `instrumental=true`, handler forces
`lyrics="[Instrumental]"` before building `GenerationParams` (matches the turbo
handler). When `instrumental=false`, `lyrics` is passed through verbatim
(empty string is valid and means "auto-generate").

### 4.3 Source-audio delivery

Caller supplies one of (precedence in order):

1. `src_audio_url` — any `https://` URL. Handler `GET`s it.
2. `src_audio_base64` — standard base64 of the audio bytes.

Handler validation:
- Reject `http://`, `file://`, non-https schemes.
- Max size **50 MB** after decode/download.
- Must decode as valid audio (`soundfile.info()` check).
- Max source duration **600 s** to prevent OOM with long references.
- 30-second download timeout for URLs.
- Content-Type should be `audio/*`; final arbiter is `soundfile.info()`.
- If **both** `url` and `base64` supplied, URL wins; warning logged.

### 4.4 Output

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

### 4.5 Error responses

```json
{ "error": "descriptive message" }
```

- Unknown `task_type` → lists valid values.
- Missing required fields → names each missing field.
- `src_audio` invalid (scheme, size, decode) → reason.
- Internal failure → `"Internal error: <exc_type>: <msg>"`, full traceback logged.

## 5. Handler implementation (`handler.py`)

Single flat file mirroring `/root/ace-step-music/handler.py` structure:

```python
# Section A: imports + env constants
#   CHECKPOINT_DIR = env "ACESTEP_CHECKPOINT_DIR" (default /runpod-volume/checkpoints)
#   DIT_CONFIG     = env "ACESTEP_CONFIG_PATH"   (default acestep-v15-xl-base)
#   INFERENCE_STEPS_DEFAULT = env "ACESTEP_INFERENCE_STEPS_DEFAULT" (default 50)
#   GUIDANCE_SCALE_DEFAULT  = env "ACESTEP_GUIDANCE_SCALE_DEFAULT"  (default 7.0)
#   MAX_BATCH_SIZE = env "ACESTEP_MAX_BATCH_SIZE" (default 4)

# Section B: _apply_torch24_compat_patches()
#   Verbatim copy from turbo handler (bool-argsort fix, SDPA enable_gqa strip)

# Section C: load_models()
#   Reads weights from CHECKPOINT_DIR (not project_root/checkpoints)
#   Skips HF download if volume already populated (which it always will be)
#   Loads DiT + LM once at cold start

# Section D: _resolve_src_audio(job_input) -> Optional[str]
#   url path:    requests.get(url, stream=True, timeout=30), 50MB cap,
#                scheme whitelist, write to tempfile
#   base64 path: b64decode, write tempfile
#   returns path (caller responsible for cleanup) or None

# Section E: _validate(job_input, task_type) -> Optional[dict]
#   Uses TASK_REQUIRED map, returns error dict or None

# Section F: handler(job) -> dict
#   - parse + validate
#   - resolve src_audio if required
#   - build GenerationParams with task-specific fields
#   - call generate_music in tempfile save_dir
#   - return response dict
#   - finally: cleanup resolved src_audio tempfile

# Section G: load_models() at import time; runpod.serverless.start({"handler": handler})
```

**LM behavior:** Handler always loads the LM at cold start (so all six tasks are
servable). For cover/repaint/extract, `thinking` is forced to `False` before
building `GenerationParams` so the LM isn't invoked per-request.

**Response shape consistency:** All six task types return the same output schema.

## 6. Dockerfile

Identical structure to `/root/ace-step-music/Dockerfile`. Deltas:

```dockerfile
# Environment deltas
ENV ACESTEP_CONFIG_PATH=acestep-v15-xl-base
ENV ACESTEP_CHECKPOINT_DIR=/runpod-volume/checkpoints
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ENV ACESTEP_LM_BACKEND=vllm
ENV ACESTEP_MAX_BATCH_SIZE=4
ENV ACESTEP_INFERENCE_STEPS_DEFAULT=50
ENV ACESTEP_GUIDANCE_SCALE_DEFAULT=7.0
ENV ACESTEP_OFFLOAD_DIT_TO_CPU=0   # safety valve, default off
```

All other lines (FROM, apt deps, pip deps, ACE-Step install, RunPod SDK install,
torch compat patches at runtime, CMD) are copied verbatim from the turbo
Dockerfile.

## 7. GitHub Actions (`.github/workflows/deploy.yml`)

Three-job pipeline identical to turbo. Deltas:

- `images: dmrabh/ace-step-music-xl` (was `dmrabh/ace-step-music`)
- Deploy job uses secret `RUNPOD_TEMPLATE_ID_XL` (new secret)
- Same GraphQL `saveTemplate` mutation, same disk-space freeing, same PR gating
  (`build-push` + `deploy` only on `push` to `main`).

## 8. Testing strategy

Three files:

1. **`test_handler.py`** — unit tests, no GPU, CI-runnable. Covers:
   - Schema validation for all six task types (missing fields, unknown task, invalid format)
   - URL scheme rejection (`http://`, `file://`)
   - Oversized download rejection (50 MB cap, streamed abort)
   - Both-provided precedence (URL wins)
   - Task routing: each task builds correct `GenerationParams`
   - `thinking` auto-forced `False` for cover/repaint/extract
   - `repainting_end=-1` sentinel preserved
   - Response shape on success for every task type
2. **`test_workflow.py`** — validates `deploy.yml` structure (YAML valid, test job present, required secrets referenced).
3. **`test_endpoint.py`** — manual integration test. Uses `RUNPOD_ENDPOINT_ID` +
   `RUNPOD_API_KEY` from env; posts one job per task type using a short public
   `https://` fixture mp3; decodes `audio_base64`, writes `out/<task>.mp3`, prints timings.

**`requirements-test.txt`:**
```
pytest
pyyaml
responses
soundfile
```

**Coverage target:** 80% on `handler.py`.

## 9. File tree (final)

```
/root/ace-step-music-xl/
├── .dockerignore
├── .env.example
├── .github/workflows/deploy.yml
├── .gitignore
├── Dockerfile
├── README.md                    # short: what this is, how to call, env var reference
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-04-20-ace-step-xl-serverless-design.md   (this file)
├── fixtures/
│   └── short.mp3                # ~2s fixture for deterministic tests
├── handler.py
├── requirements-test.txt
├── test_endpoint.py             # manual integration test
├── test_handler.py              # unit tests, CI
└── test_workflow.py             # deploy.yml structural tests
```

## 10. Deployment bootstrap (one-time, via RunPod MCP)

1. `create-network-volume`: name `ace-step-xl-weights`, 35 GB, EU-SE-1 → `volumeId`.
2. `create-pod`: use the same image we're about to deploy
   (`dmrabh/ace-step-music-xl:latest`), mount `volumeId` at `/runpod-volume`,
   EU-SE-1, A40 or A4000 (any GPU that fits the image). SSH in and run the
   project's built-in downloader so we don't have to hardcode HF repo names:
   ```bash
   python -c "
   from pathlib import Path
   from acestep.model_downloader import ensure_main_model, ensure_dit_model
   ckpt = Path('/runpod-volume/checkpoints')
   ckpt.mkdir(parents=True, exist_ok=True)
   ok, msg = ensure_main_model(checkpoints_dir=ckpt); print('main:', ok, msg)
   ok, msg = ensure_dit_model('acestep-v15-xl-base', checkpoints_dir=ckpt); print('dit:', ok, msg)
   "
   du -sh /runpod-volume/checkpoints/*
   ```
   This uses whatever HF repo the ACE-Step package currently points at, so we
   stay in sync with upstream.
3. `delete-pod`.
4. `create-template`: serverless, `dmrabh/ace-step-music-xl:latest`, env vars from §6 → `templateId`.
5. `create-endpoint`: A40, EU-SE-1, `templateId`, `networkVolumeId`, `workersMin=0`, `workersMax=3`, `idleTimeout=5`, FlashBoot on → `endpointId`.
6. Save `templateId` as GitHub secret `RUNPOD_TEMPLATE_ID_XL`.

## 11. Secrets reference

| Secret | Scope | Usage |
|---|---|---|
| `DOCKERHUB_USERNAME` | GitHub | `docker login` in build job |
| `DOCKERHUB_TOKEN` | GitHub | `docker login` in build job |
| `RUNPOD_API_KEY` | GitHub + local | GraphQL auth, MCP calls |
| `RUNPOD_TEMPLATE_ID_XL` | GitHub | `saveTemplate` mutation target |
| `RUNPOD_ENDPOINT_ID` | local (optional) | `test_endpoint.py` target |

## 12. Cost model (rough)

- **Flat monthly:** ~$2.45 (network volume) + $0 (Docker Hub public) + $0 (GHA free tier)
- **Per-request:** A40 serverless ~$0.00050/s × actual GPU time. Cold start on
  new worker ~10 s (image pull + volume mount + model load to GPU), reuse hot.
- **Scale-to-zero** via `workersMin=0`; no GPU billed when idle.

## 13. Open risks / future work

- **LM always loaded** even when only `extract` is ever called. Acceptable — loading cost is one-time at cold start, GPU memory fits.
- **Network volume region-locked** to EU-SE-1. Multi-region = duplicate volumes.
- **CFG+50 steps on XL** is ~5–10× slower than turbo's 8-step no-CFG. Per-request cost is materially higher; callers who want cheap iteration should choose turbo (existing deployment) and use XL for final renders.
- **Single Docker Hub account** (`dmrabh`). If the account is compromised or rate-limited, redeploy path breaks.

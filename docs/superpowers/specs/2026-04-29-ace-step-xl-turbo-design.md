# ACE-Step XL-turbo handler + capybara cutover

**Date:** 2026-04-29
**Status:** Draft → user review
**Companion repos:** `/root/ace-step-music-xl` (orchestrator + base handler), `/root/ace-step-music-xl-turbo` (new — to be created)

## 1. Goal

Run `scripts/capybara_tea_loop.py` against the **distilled XL-turbo DiT** (`ACE-Step/acestep-v15-xl-turbo`) instead of the **XL-base DiT** it uses today. Single-call music generation should drop from ~4–5 min to ~30–50 s, taking the music step of a 60-min run from ~50 min wall-clock down to ~6–8 min, with no perceptible quality loss for instrumental ambient lofi.

## 2. Scope

In scope:

- New repo `ace-step-music-xl-turbo` at `/root/ace-step-music-xl-turbo` — handler, Dockerfile, GitHub Actions, prefetch script, tests.
- New DockerHub image `dmrabh/ace-step-music-xl-turbo:{latest,sha}`.
- New RunPod template + endpoint, mounting existing volume `xujs4ifsur` (35 GB, EU-RO-1, named "ace-step-music-xl-turbo").
- Orchestrator changes to plumb a `preset_dict` through `OrchestratorConfig` → `run_music_pipeline` → `build_segment_payload`, defaulting to `ACE_STEP_PRESET` (base) but overridable per script.
- New `ACE_STEP_TURBO_PRESET` constant and capybara wired to it.
- Auto-prefetch via a temp RunPod CPU pod driven by the runpod MCP.

Out of scope:

- Changes to `ambient_eno_45min.py` or `lofi_45min.py`. Both keep using the existing XL-base endpoint and `ACE_STEP_PRESET`.
- Changes to the existing `/root/ace-step-music-xl` repo, image, template, or endpoint beyond the orchestrator constants/wiring above.
- Architectural changes to crossfade math (still 11 × 360 s segments, 30 s xfade, mux to 3600 s).

## 3. Decisions resolved during brainstorming

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Capybara-only cutover; base stays untouched | Lowest blast radius; ambient/lofi paths stay validated |
| 2 | Keep 11 × 360 s segment math | Pure preset swap; no plan_schema/motion_prompts churn |
| 3 | Distilled turbo preset (8 steps, cfg=1.0, use_adg=False, shift=3.0, ode) | Matches HF model card and INFERENCE.md |
| 4 | New repo at `/root/ace-step-music-xl-turbo`, image `dmrabh/ace-step-music-xl-turbo` | Mirrors volume name; parallels base naming |
| 5 | RTX 4090 24 GB GPU, same as base endpoint | ~18 GB resident VRAM estimate fits with ~6 GB headroom |
| 6 | Auto-create template + endpoint via runpod MCP | User authorized auto-execution |
| 7 | HF-CLI prefetch + Python tail for `_sync_model_code_files` | User asked for HF CLI; sync step closes the upstream gotcha |
| 8 | Auto-drive temp CPU pod via runpod MCP for prefetch | One-shot ops, fully automated |
| 9 | Plumb `preset_dict` through orchestrator config (option i) | Cleanest; isolates per-script choice |

## 4. Repo layout — `/root/ace-step-music-xl-turbo/`

```
ace-step-music-xl-turbo/
├── .github/workflows/deploy.yml
├── Dockerfile
├── handler.py
├── scripts/
│   └── prefetch_weights.sh
├── docs/
│   └── superpowers/specs/
│       └── 2026-04-29-ace-step-xl-turbo-design.md  # copy of this doc
├── test_handler.py
├── test_workflow.py
├── test_endpoint.py
├── requirements-test.txt
├── conftest.py
└── README.md
```

Bootstrap is `cp -r /root/ace-step-music-xl/ /root/ace-step-music-xl-turbo/`, then trim git history with `rm -rf .git && git init`, then apply the diffs in §5–§8 below.

## 5. Handler diff (vs `/root/ace-step-music-xl/handler.py`)

```diff
- DIT_CONFIG = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-xl-base")
+ DIT_CONFIG = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-xl-turbo")

- INFERENCE_STEPS_DEFAULT = int(os.environ.get("ACESTEP_INFERENCE_STEPS_DEFAULT", "50"))
- GUIDANCE_SCALE_DEFAULT = float(os.environ.get("ACESTEP_GUIDANCE_SCALE_DEFAULT", "7.0"))
+ INFERENCE_STEPS_DEFAULT = int(os.environ.get("ACESTEP_INFERENCE_STEPS_DEFAULT", "8"))
+ GUIDANCE_SCALE_DEFAULT = float(os.environ.get("ACESTEP_GUIDANCE_SCALE_DEFAULT", "1.0"))
```

`download_models()` gets a final block that runs unconditionally (after `ensure_dit_model` returns):

```python
# Sync .py code files even if weights pre-existed on the network volume.
# HF-CLI prefetch doesn't trigger upstream's _sync_model_code_files, so this
# guards against HF-shipped .py files diverging from the package's versions.
from acestep.model_downloader import _sync_model_code_files
synced = _sync_model_code_files("acestep-v15-xl-turbo", ckpt_path)
logger.info(f"Synced model code files: {synced}")
```

Everything else (torch24 compat patches, validation, src-audio resolver, task-type schema, `_build_params`) is identical to base. Task semantics, max single-call duration cap (600 s), and the 6-task-type matrix do not change.

## 6. Dockerfile diff

```diff
- ENV ACESTEP_CONFIG_PATH=acestep-v15-xl-base
+ ENV ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo
- ENV ACESTEP_INFERENCE_STEPS_DEFAULT=50
+ ENV ACESTEP_INFERENCE_STEPS_DEFAULT=8
- ENV ACESTEP_GUIDANCE_SCALE_DEFAULT=7.0
+ ENV ACESTEP_GUIDANCE_SCALE_DEFAULT=1.0
```

Base image, system deps, pip pins, torch 2.4.0+cu124, `ACE-Step-1.5` clone, `nano-vllm` install, `runpod`/`requests` install, and the rest of the `ENV` block (`ACESTEP_PROJECT_ROOT`, `ACESTEP_CHECKPOINTS_DIR`, `ACESTEP_LM_MODEL_PATH`, `ACESTEP_LM_BACKEND`, `ACESTEP_COMPILE_MODEL`, `ACESTEP_MAX_BATCH_SIZE`, `ACESTEP_OFFLOAD_DIT_TO_CPU`, `PYTORCH_CUDA_ALLOC_CONF`, `TORCH_CUDA_ARCH_LIST`) — all unchanged.

## 7. GitHub Actions diff (`.github/workflows/deploy.yml`)

```diff
- images: dmrabh/ace-step-music-xl
+ images: dmrabh/ace-step-music-xl-turbo

- echo "Template ${{ secrets.RUNPOD_TEMPLATE_ID_XL }} points at :latest..."
+ echo "Template ${{ secrets.RUNPOD_TEMPLATE_ID_XL_TURBO }} points at :latest..."
```

DockerHub creds (`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`) reuse the existing org-level secrets. The `RUNPOD_TEMPLATE_ID_XL_TURBO` secret is added at the new repo level after template creation.

Test job, build-push job, and deploy announcement step are otherwise unchanged.

## 8. Prefetch script — `scripts/prefetch_weights.sh`

```bash
#!/usr/bin/env bash
# Prefetch ACE-Step XL-turbo weights into the network volume.
# Idempotent: skips repos whose canonical config.json already exists.
set -euo pipefail

CKPT_DIR="${ACESTEP_CHECKPOINTS_DIR:-/runpod-volume/checkpoints}"
mkdir -p "$CKPT_DIR"

pip install --quiet --upgrade "huggingface_hub[cli]"

if [ ! -f "$CKPT_DIR/acestep-5Hz-lm-1.7B/config.json" ]; then
  echo "▸ main model (vae + text encoder + 1.7B LM + small turbo)"
  huggingface-cli download ACE-Step/Ace-Step1.5 \
    --local-dir "$CKPT_DIR" \
    --local-dir-use-symlinks False
fi

if [ ! -f "$CKPT_DIR/acestep-v15-xl-turbo/config.json" ]; then
  echo "▸ XL-turbo DiT (~18.8 GB)"
  huggingface-cli download ACE-Step/acestep-v15-xl-turbo \
    --local-dir "$CKPT_DIR/acestep-v15-xl-turbo" \
    --local-dir-use-symlinks False
fi

echo "▸ syncing custom .py code files"
python -c "
import os
from pathlib import Path
from acestep.model_downloader import _sync_model_code_files
ckpt = Path(os.environ.get('ACESTEP_CHECKPOINTS_DIR', '/runpod-volume/checkpoints'))
synced = _sync_model_code_files('acestep-v15-xl-turbo', ckpt)
print(f'synced: {synced}')
"

du -sh "$CKPT_DIR"
echo "✓ prefetch complete"
```

Auto-drive flow (runpod MCP):

1. `create-pod` — image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, GPU=cheapest available, network volume `xujs4ifsur` mounted at `/runpod-volume`, datacenter EU-RO-1.
2. Exec via RunPod API: `git clone https://github.com/<owner>/ace-step-music-xl-turbo /root/repo && bash /root/repo/scripts/prefetch_weights.sh`.
3. Verify `acestep-v15-xl-turbo/config.json` exists and total volume usage is in the 27–32 GB band.
4. `delete-pod`.

Run prefetch BEFORE the endpoint goes hot to avoid concurrent volume access.

## 9. Orchestrator changes — `/root/ace-step-music-xl/`

### 9.1 `scripts/loopvid/constants.py` — new constant

```python
# ── ACE-Step XL-turbo distilled preset (capybara) ──
# Source: HF model card (ACE-Step/acestep-v15-xl-turbo) + INFERENCE.md.
# Distilled for 8 steps, CFG disabled (guidance_scale=1.0).
ACE_STEP_TURBO_PRESET = {
    "inference_steps": 8,
    "guidance_scale": 1.0,
    "shift": 3.0,
    "use_adg": False,
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
    "infer_method": "ode",
}
```

`ACE_STEP_PRESET` is unchanged. Both ambient_eno_45min and lofi_45min continue to use it.

### 9.2 `scripts/loopvid/music_pipeline.py` — accept a preset arg

```diff
- def build_segment_payload(*, prompt: str, duration: int, seed: int) -> dict:
+ def build_segment_payload(*, prompt: str, duration: int, seed: int,
+                           preset: dict | None = None) -> dict:
+     preset = preset if preset is not None else ACE_STEP_PRESET
      return {
          "input": {
              "task_type": "text2music",
              ...
-             **ACE_STEP_PRESET,
+             **preset,
          }
      }
```

```diff
  def run_music_pipeline(
      *, prompts: list[str], duration_sec: int, seeds: list[int],
      out_dir: Path, endpoint_id: str, api_key: str,
+     preset: dict | None = None,
      on_segment_done: Optional[Callable[[int, Path], None]] = None,
  ) -> list[Path]:
      ...
-         payload = build_segment_payload(prompt=prompt, duration=duration_sec, seed=seed)
+         payload = build_segment_payload(
+             prompt=prompt, duration=duration_sec, seed=seed, preset=preset)
```

### 9.3 `scripts/loopvid/orchestrator.py` — plumb preset

```diff
  @dataclass
  class OrchestratorConfig:
      ...
      preset_plan_dict: Optional[dict] = None
+     ace_step_preset: Optional[dict] = None  # default = ACE_STEP_PRESET (base)
```

```diff
          seg_paths = run_music_pipeline(
              prompts=prompts, duration_sec=SEGMENT_DURATION_SEC, seeds=seeds,
              out_dir=music_dir,
              endpoint_id=cfg.ace_step_endpoint, api_key=cfg.runpod_api_key,
+             preset=cfg.ace_step_preset,
              on_segment_done=lambda i, p: _print(...),
          )
```

### 9.4 `scripts/capybara_tea_loop.py` — pass turbo preset + new endpoint

```diff
- DEFAULT_ACE_STEP_ENDPOINT = "nwqnd0duxc6o38"
+ DEFAULT_ACE_STEP_ENDPOINT = "<NEW_TURBO_ENDPOINT_ID>"  # filled by §12 step 6 after MCP create-endpoint returns
```

```diff
  from loopvid.capybara_preset import (...)
+ from loopvid.constants import ACE_STEP_TURBO_PRESET
  ...
  cfg = OrchestratorConfig(
      ...
+     ace_step_preset=ACE_STEP_TURBO_PRESET,
  )
```

ambient_eno_45min.py and lofi_45min.py do NOT pass `ace_step_preset`, so they fall through to the base preset.

## 10. RunPod resources

### 10.1 Network volume (existing, no action)

- ID: `xujs4ifsur`
- Name: `ace-step-music-xl-turbo`
- Size: 35 GB
- Datacenter: EU-RO-1
- Expected fill: ~28 GB (main ~9 GB + XL-turbo DiT ~18.8 GB)
- Headroom: ~7 GB (covers tokenizer/config drift, future LM swaps)

### 10.2 Template (to be created via MCP)

- Name: `ace-step-music-xl-turbo`
- Image: `dmrabh/ace-step-music-xl-turbo:latest`
- Container disk: 20 GB
- Volume mount path: `/runpod-volume`
- Env:
  - `ACESTEP_CHECKPOINTS_DIR=/runpod-volume/checkpoints`
  - `ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo`
  - `ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B`
  - `ACESTEP_LM_BACKEND=vllm`
  - `ACESTEP_COMPILE_MODEL=1`
  - `ACESTEP_OFFLOAD_DIT_TO_CPU=0`
  - `ACESTEP_INFERENCE_STEPS_DEFAULT=8`
  - `ACESTEP_GUIDANCE_SCALE_DEFAULT=1.0`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - `TORCH_CUDA_ARCH_LIST=8.9`

### 10.3 Endpoint (to be created via MCP)

- Name: `ace-step-music-xl-turbo`
- GPU type: `NVIDIA GeForce RTX 4090`
- Workers: min=0, max=2, standby=2
- Idle timeout: 5 s
- Execution timeout: 600 000 ms (10 min — matches handler max duration)
- Network volume: `xujs4ifsur` (datacenter is implicitly EU-RO-1 — RunPod requires endpoint to colocate with the volume)
- Scaler: `QUEUE_DELAY=4`
- Flashboot: enabled

The resulting endpoint id is captured and patched into `scripts/capybara_tea_loop.py:DEFAULT_ACE_STEP_ENDPOINT` automatically.

## 11. Testing

### 11.1 Handler unit tests (new repo)

`test_handler.py` cloned from base; assertions for env defaults change:

```python
assert handler.INFERENCE_STEPS_DEFAULT == 8     # was 50
assert handler.GUIDANCE_SCALE_DEFAULT == 1.0    # was 7.0
assert handler.DIT_CONFIG == "acestep-v15-xl-turbo"
```

`test_workflow.py` and `test_endpoint.py` cloned, with `test_endpoint.py` retargeted to the new endpoint id.

### 11.2 Orchestrator tests (existing repo)

Extend `test_loopvid_constants.py` with a snapshot for `ACE_STEP_TURBO_PRESET`.

Add `test_loopvid_music_pipeline.py` cases:

- `build_segment_payload(preset=None)` → uses base `ACE_STEP_PRESET`
- `build_segment_payload(preset=ACE_STEP_TURBO_PRESET)` → 8 steps, cfg=1.0, use_adg=False
- `run_music_pipeline(preset=...)` → forwards preset to `build_segment_payload`

These confirm ambient_eno/lofi behavior is frozen (no preset arg → base preset).

### 11.3 Smoke test (opt-in)

Extend `test_capybara_tea_loop_smoke.py` with a turbo-only smoke gated on `ACESTEP_TURBO_SMOKE=1`. One short request (`--duration 60`), assert round-trip < 90 s and a non-empty mp3.

## 12. Rollout sequence (post-approval)

1. Commit this design doc.
2. Hand off to writing-plans skill for an implementation plan.
3. Plan execution scaffolds the new repo from a copy of base, applies §5/§6/§7/§8 diffs.
4. `gh repo create dmrabh/ace-step-music-xl-turbo --public --source=. --push` (org and visibility per user).
5. GitHub Actions builds and pushes `dmrabh/ace-step-music-xl-turbo:latest` (~10 min).
6. Runpod MCP `create-template` + `create-endpoint`. Capture the endpoint id.
7. Runpod MCP: spin temp CPU pod, run prefetch, tear down (~15–20 min).
8. Apply orchestrator diffs (§9) and patch capybara `DEFAULT_ACE_STEP_ENDPOINT`.
9. Run unit tests in both repos.
10. Run opt-in 60-s turbo smoke against the new endpoint.
11. Save memory entry: turbo endpoint id, repo name, image, template id, volume.

## 13. Risks and mitigations

| # | Risk | Mitigation |
|---|------|-----------|
| 1 | RTX 4090 24 GB OOM at cold start | Set `ACESTEP_OFFLOAD_DIT_TO_CPU=1` and re-deploy template; or upgrade to A40 48 GB |
| 2 | Turbo audio quality regression on ambient lofi | Easy revert: change capybara `DEFAULT_ACE_STEP_ENDPOINT` back to base, drop `ace_step_preset=` arg |
| 3 | Volume bloat (~3 GB unused small turbo DiT) | Tolerable at 35 GB allocation; phase-2 pruning if needed |
| 4 | Concurrent prefetch + cold start corrupts download | Prefetch BEFORE endpoint goes hot (workersMin=0 starts cold) |
| 5 | HF-CLI download stalls or partial-file artifacts | `huggingface-cli` retries; idempotent re-run skips completed repos |
| 6 | `_sync_model_code_files` import path drift between upstream releases | Tail step is wrapped to log+warn but not fail prefetch; handler boot also runs sync |
| 7 | New GH repo collides with org policy | If `dmrabh/ace-step-music-xl-turbo` is taken/restricted, fall back to user namespace |

## 14. Non-goals (deferred)

- Orchestration-level retry policy for failed turbo segments (existing `runpod_client.run_segment` retries are sufficient).
- A single shared "long-form preset registry" abstraction across all three scripts. Wait until a 4th script lands.
- Dropping the unused small `acestep-v15-turbo` DiT during prefetch.
- Promoting turbo to ambient_eno_45min or lofi_45min after capybara validation. Separate decision.

## 15. Open questions for implementation phase

- Exact GitHub owner/org for the new repo. The DockerHub image namespace is `dmrabh/`, so we assume `dmrabh/ace-step-music-xl-turbo` unless told otherwise during plan execution.

## 16. Deliberate non-features

- **No `--preset {turbo,base}` flag on capybara_tea_loop.py.** The preset is a code-level constant; flipping it requires a code change. This prevents accidental flag drift and forces deliberate base-preset use if we ever need to fall back.
- **No promotion of turbo to ambient_eno_45min or lofi_45min.** Those scripts continue to use base. A future decision after capybara validation can revisit this.

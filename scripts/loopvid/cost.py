"""Cost estimator for the loop_music_video pipeline.

Per-second pricing constants are best-effort; actual cost is recorded into
manifest.json as call durations come back."""
from __future__ import annotations

from typing import Iterable

from scripts.loopvid.constants import (
    SEGMENT_COUNT_60MIN,
    SEGMENT_DURATION_SEC,
    CLIP_COUNT,
    CLIP_DURATION_SEC,
)

# Per-second pricing (USD)
RTX_4090_PER_SEC = 0.00031     # ACE-Step worker
RTX_PRO_6000_PER_SEC = 0.00076  # LTX worker

# Per-call pricing (USD)
SEEDREAM_PER_IMAGE = 0.03
GEMINI_3_FLASH_PER_RUN = 0.001  # ~700-1500 tokens output, generous estimate

# Heuristics for inference time per second of output
ACE_STEP_INFERENCE_SEC_PER_OUTPUT_SEC = 1.6   # 360s output ≈ 575s on RTX 4090
LTX_INFERENCE_SEC_PER_OUTPUT_SEC = 5.0        # 7s output ≈ 35s on RTX Pro 6000


class BudgetExceededError(RuntimeError):
    pass


def _step_costs(duration_sec: int) -> dict:
    music_segments = max(1, SEGMENT_COUNT_60MIN if duration_sec >= 3000 else
                          (duration_sec + SEGMENT_DURATION_SEC - 1) // SEGMENT_DURATION_SEC)
    music_inference_sec = music_segments * SEGMENT_DURATION_SEC * ACE_STEP_INFERENCE_SEC_PER_OUTPUT_SEC
    music_cost = music_inference_sec * RTX_4090_PER_SEC

    video_inference_sec = CLIP_COUNT * CLIP_DURATION_SEC * LTX_INFERENCE_SEC_PER_OUTPUT_SEC
    video_cost = video_inference_sec * RTX_PRO_6000_PER_SEC

    return {
        "plan":  GEMINI_3_FLASH_PER_RUN,
        "image": SEEDREAM_PER_IMAGE,
        "music": music_cost,
        "video": video_cost,
        "loop_build": 0.0,
        "mux": 0.0,
    }


def estimate_run_cost(duration_sec: int = 3600, skip: Iterable[str] = ()) -> float:
    skipset = set(skip)
    costs = _step_costs(duration_sec)
    return sum(c for step, c in costs.items() if step not in skipset)


def cost_breakdown_lines(duration_sec: int = 3600, skip: Iterable[str] = ()) -> list[str]:
    skipset = set(skip)
    costs = _step_costs(duration_sec)
    labels = {
        "plan":  "LLM (Gemini 3 Flash)",
        "image": "Image (Seedream 4.5)",
        "music": f"Music (ACE-Step, {SEGMENT_COUNT_60MIN} segments)",
        "video": f"Video (LTX, {CLIP_COUNT} clips)",
    }
    out = []
    for step, label in labels.items():
        if step in skipset:
            continue
        out.append(f"  - {label:<35} ${costs[step]:.3f}")
    out.append(f"  TOTAL: ${estimate_run_cost(duration_sec, skip):.2f}")
    return out


def enforce_budget(estimated: float, max_cost: float) -> None:
    if estimated > max_cost:
        raise BudgetExceededError(
            f"Estimated ${estimated:.2f} exceeds --max-cost ${max_cost:.2f}"
        )

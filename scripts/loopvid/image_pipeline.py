"""Replicate Seedream 4.5 caller — text-to-image, atomic write to disk."""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from scripts.loopvid.constants import SEEDREAM_HARD_CONSTRAINTS

REPLICATE_PREDICTIONS_URL = "https://api.replicate.com/v1/models/{model}/predictions"
REPLICATE_PREDICTION_STATUS_URL = "https://api.replicate.com/v1/predictions/{pred_id}"
MODEL = "bytedance/seedream-4.5"
ASPECT_RATIO = "16:9"
POLL_INTERVAL_SEC = 3
POLL_TIMEOUT_SEC = 600
DOWNLOAD_TIMEOUT_SEC = 60


def build_seedream_prompt(scene: str, style: str, *, constraints: str = SEEDREAM_HARD_CONSTRAINTS) -> str:
    return f"{scene}. {style}. {constraints}"


def generate_still(
    *, prompt: str, api_token: str, out_path: Path,
    poll_interval: int = POLL_INTERVAL_SEC,
    timeout_sec: int = POLL_TIMEOUT_SEC,
) -> str:
    """Generate one still image via Replicate. Returns the prediction_id (for
    manifest tracking) and atomically writes the PNG bytes to out_path."""
    if not api_token:
        raise ValueError("api_token is required (set REPLICATE_API_TOKEN)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    submit_url = REPLICATE_PREDICTIONS_URL.format(model=MODEL)
    body = {"input": {"prompt": prompt, "aspect_ratio": ASPECT_RATIO}}
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

    resp = requests.post(submit_url, json=body, headers=headers, timeout=60)
    resp.raise_for_status()
    pred_id = resp.json()["id"]

    status_url = REPLICATE_PREDICTION_STATUS_URL.format(pred_id=pred_id)
    start = time.time()
    while time.time() - start < timeout_sec:
        s = requests.get(status_url, headers=headers, timeout=30).json()
        status = s.get("status", "")
        if status == "succeeded":
            output = s.get("output")
            url = output if isinstance(output, str) else (output[0] if output else None)
            if not url:
                raise RuntimeError(f"Replicate succeeded but no output URL: {s}")
            img = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SEC).content
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp.write_bytes(img)
            os.replace(tmp, out_path)
            return pred_id
        if status in ("failed", "canceled"):
            raise RuntimeError(
                f"Replicate prediction {pred_id} {status}: {s.get('error', s)}"
            )
        if poll_interval > 0:
            time.sleep(poll_interval)
    raise RuntimeError(f"Replicate prediction {pred_id} timed out after {timeout_sec}s")

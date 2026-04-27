"""Shared RunPod /v2 async-job client.

Extracted from scripts/ambient_eno_45min.py — see that script's docstrings
for the rationale on /run vs /runsync, transient-404 tolerance, and the
whole-segment retry wrapper.
"""
from __future__ import annotations

import time
from typing import Optional

import requests

REQUEST_TIMEOUT_SEC = 1800
POLL_INTERVAL_SEC = 5
MAX_TRANSIENT_404 = 6
MAX_SEGMENT_RETRIES = 3


def submit_job(endpoint_id: str, api_key: str, payload: dict) -> dict:
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    resp = requests.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    return resp.json()


def poll_job(
    endpoint_id: str,
    api_key: str,
    job_id: str,
    *,
    poll_interval: int = POLL_INTERVAL_SEC,
) -> dict:
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    consecutive_404s = 0
    while True:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 404:
            consecutive_404s += 1
            if consecutive_404s > MAX_TRANSIENT_404:
                raise RuntimeError(f"Too many consecutive 404s polling job {job_id}")
            if poll_interval > 0:
                time.sleep(poll_interval)
            continue
        resp.raise_for_status()
        consecutive_404s = 0
        body = resp.json()
        status = body.get("status", "")
        if status == "COMPLETED":
            return body
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(
                f"Job {job_id} terminal status {status}: {body.get('error', body)}"
            )
        if poll_interval > 0:
            time.sleep(poll_interval)


def run_segment(
    *,
    endpoint_id: str,
    api_key: str,
    payload: dict,
    label: str,
    poll_interval: int = POLL_INTERVAL_SEC,
    retry_sleep: int = 5,
    max_retries: int = MAX_SEGMENT_RETRIES,
) -> dict:
    """Submit one job and poll to completion, with up to max_retries retries
    on transient failure. `label` is for error messages only (e.g., 'music seg 3')."""
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            body = submit_job(endpoint_id, api_key, payload)
            status = body.get("status", "")
            if status == "COMPLETED":
                return body
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(
                    f"{label} submit returned status={status}: {body.get('error', body)}"
                )
            job_id = body.get("id", "")
            if not job_id:
                raise RuntimeError(f"{label} submit response missing id: {body}")
            return poll_job(endpoint_id, api_key, job_id, poll_interval=poll_interval)
        except (requests.RequestException, RuntimeError) as e:
            last_err = e
            if attempt < max_retries:
                if retry_sleep > 0:
                    time.sleep(retry_sleep)
                continue
    raise RuntimeError(f"{label} failed after {max_retries} attempts: {last_err}")

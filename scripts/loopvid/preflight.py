"""Pre-flight checks — fail fast before any paid API call."""
from __future__ import annotations

import os
import shutil

import requests


class PreflightError(RuntimeError):
    pass


def check_env_vars(names: tuple[str, ...]) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise PreflightError(
            f"Missing required env vars: {', '.join(missing)}. "
            f"Source from /root/avatar-video/.env or export explicitly."
        )


def check_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise PreflightError("ffmpeg not found on $PATH — install ffmpeg before running")


def _get_endpoint(endpoint_id: str, api_key: str) -> dict:
    """Fetch endpoint metadata via RunPod REST API. Wrapped for test mocking."""
    url = f"https://rest.runpod.io/v1/endpoints/{endpoint_id}"
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    if resp.status_code == 404:
        raise PreflightError(f"Endpoint {endpoint_id} not found or inaccessible")
    resp.raise_for_status()
    return resp.json()


def check_endpoint_workers(endpoint_id: str, api_key: str) -> None:
    info = _get_endpoint(endpoint_id, api_key)
    if info.get("workersMax", 0) < 1:
        raise PreflightError(
            f"Endpoint {endpoint_id} has workersMax={info.get('workersMax')}. "
            f"Run: rp endpoint update {endpoint_id} --workers-max 1"
        )


def run_preflight(
    *,
    runpod_api_key: str,
    ace_step_endpoint: str,
    ltx_endpoint: str,
    require_ace_step: bool = True,
    require_ltx: bool = True,
) -> None:
    check_env_vars(("OPENROUTER_API_KEY", "REPLICATE_API_TOKEN", "RUNPOD_API_KEY"))
    check_ffmpeg_available()
    if require_ace_step:
        check_endpoint_workers(ace_step_endpoint, runpod_api_key)
    if require_ltx:
        check_endpoint_workers(ltx_endpoint, runpod_api_key)

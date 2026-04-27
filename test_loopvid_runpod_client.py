import json
import pytest
import responses
from scripts.loopvid.runpod_client import submit_job, poll_job, run_segment


EP = "test-endpoint"
KEY = "test-key"


@responses.activate
def test_submit_job_posts_to_run_endpoint():
    responses.add(
        responses.POST,
        f"https://api.runpod.ai/v2/{EP}/run",
        json={"id": "job-1", "status": "IN_QUEUE"},
        status=200,
    )
    body = submit_job(EP, KEY, {"input": {"x": 1}})
    assert body["id"] == "job-1"
    assert body["status"] == "IN_QUEUE"
    assert responses.calls[0].request.headers["Authorization"] == f"Bearer {KEY}"


@responses.activate
def test_poll_job_returns_completed_body():
    responses.add(
        responses.GET,
        f"https://api.runpod.ai/v2/{EP}/status/job-1",
        json={"status": "COMPLETED", "output": {"audio_base64": "abc"}},
        status=200,
    )
    body = poll_job(EP, KEY, "job-1", poll_interval=0)
    assert body["output"]["audio_base64"] == "abc"


@responses.activate
def test_poll_job_tolerates_transient_404s():
    for _ in range(3):
        responses.add(
            responses.GET,
            f"https://api.runpod.ai/v2/{EP}/status/job-1",
            status=404,
        )
    responses.add(
        responses.GET,
        f"https://api.runpod.ai/v2/{EP}/status/job-1",
        json={"status": "COMPLETED", "output": {}},
        status=200,
    )
    body = poll_job(EP, KEY, "job-1", poll_interval=0)
    assert body["status"] == "COMPLETED"


@responses.activate
def test_poll_job_raises_after_too_many_404s():
    for _ in range(10):
        responses.add(
            responses.GET,
            f"https://api.runpod.ai/v2/{EP}/status/job-1",
            status=404,
        )
    with pytest.raises(RuntimeError, match="404"):
        poll_job(EP, KEY, "job-1", poll_interval=0)


@responses.activate
def test_poll_job_raises_on_terminal_failure():
    responses.add(
        responses.GET,
        f"https://api.runpod.ai/v2/{EP}/status/job-1",
        json={"status": "FAILED", "error": "OOM"},
        status=200,
    )
    with pytest.raises(RuntimeError, match="OOM"):
        poll_job(EP, KEY, "job-1", poll_interval=0)

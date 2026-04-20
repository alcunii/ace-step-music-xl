#!/usr/bin/env python3
"""Manual integration test for the ACE-Step XL RunPod endpoint.

Usage:
  export RUNPOD_API_KEY=...
  export RUNPOD_ENDPOINT_ID=...
  python test_endpoint.py --task text2music --prompt "jazz piano"
  python test_endpoint.py --task cover --src-url https://.../track.mp3 --prompt "lo-fi cover"
  python test_endpoint.py --task repaint --src-url https://.../a.mp3 --prompt "fix" --start 5 --end 15
  python test_endpoint.py --task extract --src-url https://.../a.mp3 --instruction "drums"
  python test_endpoint.py --task lego --src-url https://.../a.mp3 --prompt "remix" --start 0 --end 20
  python test_endpoint.py --task complete --src-url https://.../a.mp3 --prompt "outro"
  python test_endpoint.py --all   # run all 6 tasks in sequence against a preset fixture
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

TASKS = ["text2music", "cover", "repaint", "extract", "lego", "complete"]

# Default public fixture for --all; override with --src-url if needed.
DEFAULT_FIXTURE_URL = "https://github.com/alcunii/ace-step-music-xl/raw/main/fixtures/short.mp3"


def build_payload(args) -> dict:
    inp = {
        "task_type": args.task,
        "audio_format": args.format,
        "inference_steps": args.steps,
        "seed": args.seed,
        "batch_size": args.batch_size,
    }
    if args.prompt:
        inp["prompt"] = args.prompt
    if args.src_url:
        inp["src_audio_url"] = args.src_url
    if args.duration:
        inp["duration"] = args.duration
    if args.instruction:
        inp["instruction"] = args.instruction
    if args.start is not None:
        inp["repainting_start"] = args.start
    if args.end is not None:
        inp["repainting_end"] = args.end
    if args.lyrics:
        inp["lyrics"] = args.lyrics
        inp["instrumental"] = False
    return {"input": inp}


def call_endpoint(endpoint_id: str, api_key: str, payload: dict, timeout: int) -> dict:
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")
        sys.exit(1)

    # Poll if still pending
    while result.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
        job_id = result["id"]
        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
        time.sleep(5)
        status_req = urllib.request.Request(
            status_url, headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(status_req, timeout=30) as resp:
            result = json.loads(resp.read())
        print(f"  status: {result.get('status')}")
        if result.get("status") in ("FAILED", "CANCELLED", "TIMED_OUT"):
            print(json.dumps(result, indent=2))
            sys.exit(1)
    return result


def save_output(result: dict, task: str, fmt: str, out_dir: str) -> None:
    output = result.get("output", {})
    if "error" in output:
        print(f"Error: {output['error']}")
        sys.exit(1)
    audio_b64 = output.get("audio_base64")
    if not audio_b64:
        print("No audio in response")
        print(json.dumps(output, indent=2))
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{task}.{fmt}")
    with open(path, "wb") as f:
        f.write(base64.b64decode(audio_b64))

    size = os.path.getsize(path)
    print(f"  saved {path} ({size:,} bytes, duration={output.get('duration')}s, seed={output.get('seed')})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, default="text2music")
    p.add_argument("--prompt", default="upbeat electronic dance music")
    p.add_argument("--src-url", default="")
    p.add_argument("--instruction", default="")
    p.add_argument("--lyrics", default="")
    p.add_argument("--start", type=float, default=None)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--duration", type=float, default=0)
    p.add_argument("--format", choices=["mp3", "wav", "flac"], default="mp3")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY", ""))
    p.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID", ""))
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--out", default="out")
    p.add_argument("--all", action="store_true", help="Run all 6 tasks in sequence")
    args = p.parse_args()

    if not args.api_key or not args.endpoint_id:
        print("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID env vars")
        sys.exit(1)

    if args.all:
        fixture = args.src_url or DEFAULT_FIXTURE_URL
        task_presets = [
            ("text2music", {"prompt": "energetic synthwave"}),
            ("cover",      {"prompt": "lo-fi jazz cover", "src_url": fixture}),
            ("repaint",    {"prompt": "calmer chorus", "src_url": fixture,
                            "start": 5, "end": 15}),
            ("extract",    {"instruction": "isolate drums", "src_url": fixture}),
            ("lego",       {"prompt": "extended intro", "src_url": fixture,
                            "start": 0, "end": 20}),
            ("complete",   {"prompt": "cinematic outro", "src_url": fixture}),
        ]
        for task, preset in task_presets:
            print(f"\n=== {task} ===")
            args.task = task
            args.prompt = preset.get("prompt", "")
            args.src_url = preset.get("src_url", "")
            args.instruction = preset.get("instruction", "")
            args.start = preset.get("start")
            args.end = preset.get("end")

            payload = build_payload(args)
            print(f"  payload: {json.dumps(payload['input'])[:120]}...")
            t0 = time.time()
            result = call_endpoint(args.endpoint_id, args.api_key, payload, args.timeout)
            dt = time.time() - t0
            print(f"  elapsed: {dt:.1f}s")
            save_output(result, task, args.format, args.out)
        return

    payload = build_payload(args)
    print(f"Payload: {json.dumps(payload['input'])[:200]}")
    t0 = time.time()
    result = call_endpoint(args.endpoint_id, args.api_key, payload, args.timeout)
    print(f"Elapsed: {time.time() - t0:.1f}s")
    save_output(result, args.task, args.format, args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate an original 3-minute Bruno-Mars-style funk-pop song from scratch
via the ACE-Step 1.5 XL serverless endpoint.

task_type: text2music (no source audio, vocal track with lyrics)

Usage:

  export RUNPOD_API_KEY=<your-key>
  export RUNPOD_ENDPOINT_ID=nwqnd0duxc6o38   # or your endpoint
  python3 scripts/bruno_mars_style_midnight_gold.py

Config knobs (at top of file) use the officially recommended XL-base settings
from https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md
("High-Quality Generation (Base Model)").
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_PATH = "/root/ace-step-music-xl/out/midnight-gold.mp3"

# ---------------------------------------------------------------------------
# Original lyrics — Bruno-Mars-inspired funk-pop love song, structured with
# section tags that ACE-Step understands. ~3 minutes at 112 BPM.
# ---------------------------------------------------------------------------
LYRICS = """[Intro]
Oh yeah
Hey, hey
Come on

[Verse 1]
Clock striking seven, city lights are flashing low
Got my fresh shoes on and the rhythm in my soul
I picked up the phone, baby, told you meet me downtown
Turn the radio up 'cause I'm ready to get down

[Pre-Chorus]
Oh, the saxophone's calling, yeah the bassline's swinging
Every star in the sky, baby, look how they're gleaming
Lock eyes across the floor
Tell me what you waiting for

[Chorus]
We got midnight gold running through our veins
Hips swaying slow like a sweet refrain
Baby take my hand, won't you hold it tight
Midnight gold, gonna shine all night
Oh, oh, oh
Midnight gold, gonna shine all night

[Verse 2]
Velvet dress glowing like a supernova star
Ice in my glass but the feeling's taking over hard
The band hits a groove and I can feel the floor shake
One look from you, girl, and my heart's gonna break

[Pre-Chorus]
Hear the saxophone calling, feel the bassline swinging
Every diamond you're wearing's got the moonlight singing
Lock eyes across the floor
Tell me what you waiting for

[Chorus]
We got midnight gold running through our veins
Hips swaying slow like a sweet refrain
Baby take my hand, won't you hold it tight
Midnight gold, gonna shine all night
Oh, oh, oh
Midnight gold, gonna shine all night

[Bridge]
Turn the lights down low
Let the horns blow
I don't need nobody but you, you, you
Turn the lights down low
Let the horns blow
Everything I'm doing, I'm doing it for you

[Chorus]
We got midnight gold running through our veins
Hips swaying slow like a sweet refrain
Baby take my hand, won't you hold it tight
Midnight gold, gonna shine all night

[Outro]
Shine all night, shine all night
Yeah we shine all night
Oh baby we shine all night
Mmm, midnight gold
"""

# ---------------------------------------------------------------------------
# Style prompt — crafted for modern Motown-revival funk-pop, Bruno-Mars /
# Silk Sonic / 24K Magic territory. Dense descriptor set to steer the model
# hard toward that specific sonic signature.
# ---------------------------------------------------------------------------
PROMPT = (
    "Feel-good retro funk-pop in the style of modern Motown revival and "
    "silk-soul pop, silky male lead tenor vocal with smooth falsetto "
    "ad-libs and lush stacked background harmonies, tight groove-locked "
    "rhythm section, slap electric bass playing syncopated 16th-note "
    "lines, clean Fender Rhodes electric piano with vintage chorus, "
    "bright stereo horn section featuring trumpet trombone and tenor "
    "saxophone stabs, crisp live-room drum kit with snappy backbeat on "
    "two and four, sparkly muted single-coil guitar chicken-picking, "
    "handclaps and finger snaps on the backbeat, glossy 80s-meets-2020s "
    "radio production, warm analog tape compression and plate reverb on "
    "vocals, uplifting romantic late-night dance-floor atmosphere, "
    "112 BPM in 4/4 time, key of B-flat major, danceable mid-tempo "
    "groove, world-class pop production quality, pristine stereo mix"
)

# ---------------------------------------------------------------------------
# Generation knobs — ACE-Step 1.5 XL-base "High-Quality Generation" preset
# (source: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md)
# ---------------------------------------------------------------------------
INFERENCE_STEPS = 64       # docs: 64 for high quality on XL base
GUIDANCE_SCALE = 8.0       # docs: 8.0 high-quality preset
SHIFT = 3.0                # docs: 3.0 recommended timestep shift for XL base
USE_ADG = True             # Adaptive Dual Guidance — XL-base-only quality booster
CFG_INTERVAL_START = 0.0   # full CFG interval
CFG_INTERVAL_END = 1.0
INFER_METHOD = "ode"       # faster/deterministic Euler sampler
SEED = -1                  # -1 = random. Pin once you find a take you like.

# Song length. ACE-Step clamps to [10, 600]. 180s = 3 minutes.
DURATION = 180

# THINKING=False keeps our crafted caption verbatim. With THINKING=True the
# LM paraphrases the prompt before DiT sees it — in a prior run that turned
# "funk-pop/Motown-revival, 112 BPM" into "nu-disco, 107 BPM" and dropped
# the tenor/Rhodes/chicken-picking details. Our prompt already declares
# BPM, key and time signature, so we don't need CoT to infer them.
THINKING = False
LM_TEMPERATURE = 0.85

# mp3 for fast iteration; flip to "wav" for lossless critical listening.
AUDIO_FORMAT = "mp3"

TIMEOUT = 1800
POLL_INTERVAL = 10


def main():
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "nwqnd0duxc6o38").strip()
    if not api_key:
        print("ERROR: export RUNPOD_API_KEY=<your-key> first", file=sys.stderr)
        sys.exit(1)

    lyrics_clean = LYRICS.strip()
    print(f"Lyrics: {len(lyrics_clean.splitlines())} lines, {len(lyrics_clean)} chars")
    print(f"Prompt: {len(PROMPT)} chars")
    print(f"Duration: {DURATION}s")

    payload_input = {
        "task_type": "text2music",
        "prompt": PROMPT,
        "lyrics": lyrics_clean,
        "instrumental": False,
        "duration": DURATION,
        "inference_steps": INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "shift": SHIFT,
        "use_adg": USE_ADG,
        "cfg_interval_start": CFG_INTERVAL_START,
        "cfg_interval_end": CFG_INTERVAL_END,
        "infer_method": INFER_METHOD,
        "seed": SEED,
        "audio_format": AUDIO_FORMAT,
        "thinking": THINKING,
        "lm_temperature": LM_TEMPERATURE,
    }
    payload = {"input": payload_input}

    redacted = {**payload_input, "lyrics": f"<{len(lyrics_clean)} chars>"}
    print(f"\nPayload summary:\n  {json.dumps(redacted, indent=2)}")

    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    print(f"\nSubmitting text2music job to endpoint {endpoint_id}...")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        sys.exit(1)

    job_id = result.get("id", "")
    print(f"Job submitted: {job_id}  status={result.get('status')}")

    consecutive_404s = 0
    max_404s = 6
    while result.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
        time.sleep(POLL_INTERVAL)
        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
        status_req = urllib.request.Request(
            status_url, headers={"Authorization": f"Bearer {api_key}"}
        )
        try:
            with urllib.request.urlopen(status_req, timeout=30) as resp:
                result = json.loads(resp.read())
            consecutive_404s = 0
        except urllib.error.HTTPError as e:
            if e.code == 404 and consecutive_404s < max_404s:
                consecutive_404s += 1
                print(f"  status: (transient 404 {consecutive_404s}/{max_404s})")
                continue
            print(f"Poll error HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
            sys.exit(1)
        elapsed = int(time.time() - t0)
        print(f"  status: {result.get('status')} ({elapsed}s elapsed)")
        if result.get("status") in ("FAILED", "CANCELLED", "TIMED_OUT"):
            print(json.dumps(result, indent=2), file=sys.stderr)
            sys.exit(1)

    total = time.time() - t0
    output = result.get("output", {})
    if "error" in output:
        print(f"Generation error: {output['error']}", file=sys.stderr)
        sys.exit(1)

    audio_b64 = output.get("audio_base64", "")
    if not audio_b64:
        print("No audio in response:")
        print(json.dumps(result, indent=2), file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    audio_bytes = base64.b64decode(audio_b64)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(audio_bytes)

    print("\n" + "=" * 60)
    print(f"SAVED: {OUTPUT_PATH}")
    print(f"  size: {len(audio_bytes):,} bytes")
    print(f"  duration: {output.get('duration')}s")
    print(f"  sample_rate: {output.get('sample_rate')} Hz")
    print(f"  seed: {output.get('seed')}  (set SEED in script to reuse)")
    print(f"  total time: {total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

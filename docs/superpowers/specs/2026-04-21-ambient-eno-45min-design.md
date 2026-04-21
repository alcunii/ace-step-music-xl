# 45-minute Eno-style ambient via ACE-Step 1.5 XL — design

**Status:** Draft (approved in brainstorming, awaiting written-spec review)
**Date:** 2026-04-21
**Target endpoint:** `nwqnd0duxc6o38` (already deployed; no redeploy required)
**Output:** single `eno_45min_final.flac`, ~46 minutes, lossless

## 1. Goal

Generate a single ~45-minute Eno-style tonal ambient piece ("Music for Airports" lineage) using the existing ACE-Step 1.5 XL serverless endpoint, producing a listenable, harmonically cohesive track that follows a "two-phase breath" arc (settle-in → widen → dissolve).

Constraints the code must respect:

- Each serverless call is capped at `duration ≤ 600 s` (handler.py:466) and a 10 min job timeout — a 45 min piece cannot be generated in one shot.
- Quality must match the ACE-Step team's official "High-Quality Generation (Base Model)" preset verbatim — no sampling-knob compromises.
- No changes to the Docker image, handler, GitHub Actions workflow, or RunPod template.

## 2. Strategy — client-side orchestration of 7 independent text2music segments

Seven `text2music` calls to the existing endpoint, each ~420 s, stitched locally with equal-power crossfades.

Rejected alternatives and why:

- **`complete`-chain continuation** (feed the tail of segment N as `src_audio` into segment N+1 via `task_type=complete`). Musically continuous in principle, but quality drift over 4–6 hops is uncharacterised; each hop re-encodes through the VAE (mild loss); a mid-chain failure loses everything downstream. Crossfading over sustained Eno-style pads is already close to inaudible, so the continuity upside is small.
- **Single-seed loop with variation.** Too homogeneous for the "breath" arc (Section 3).
- **Server-side stitching via a new task type.** Would require a redeploy, would blow past the 10 min job timeout, and the client-side stitch is already trivial. Not worth it.
- **Parallel submission to N workers.** Wall-clock 5–6 min instead of ~28–35 min, but each worker pays its own ~60–90 s cold start loading the 35 GB model from the network volume, multiplying GPU-seconds by 7×. Sequential submission rides the warm worker and is the right tradeoff for a one-shot generation.

## 3. Musical structure — "two-phase breath" (7 segments)

Locked sonic palette, shared by every segment (this is the identity that holds 45 min together):

```
Eno-inspired tonal ambient, slowly evolving pads, soft felted piano notes
scattered over sustained synthesizer bed, warm analog tape bloom, long
plate-reverb tails, harmonic major key, 50 BPM in 4/4 (barely perceptible
pulse), no percussion, no vocals, spacious stereo field, gentle overtone
shimmer, meditative calm atmosphere, pristine recording
```

Per-segment evolution — only one or two descriptors shift per step:

| # | Phase      | Added descriptors (appended to locked palette) |
|---|-----------|-----------------------------------------------|
| 1 | Inhale-1   | "sparse piano notes, soft entry, first unfolding" |
| 2 | Inhale-2   | "settling pads, slightly lower register" |
| 3 | Inhale-3   | "deepest stillness, fewest events, suspended" |
| 4 | Turn       | "widest reverb, slowest harmonic change, held breath" |
| 5 | Exhale-1   | "overtones emerging, air widening" |
| 6 | Exhale-2   | "sparser piano, more air, upper register glow" |
| 7 | Dissolve   | "dissolving pads, long diminuendo, fade into silence" |

Duration math: 7 × 420 s − 6 × 30 s crossfade = 2760 s ≈ **46:00** total.

## 4. Generation config — official XL-base preset, verbatim

From `docs/en/INFERENCE.md` Example 9, "High-Quality Generation (Base Model)":

| Parameter              | Value  | Note |
|------------------------|--------|------|
| `inference_steps`      | 64     | Official high-quality value |
| `guidance_scale`       | 8.0    | Official |
| `shift`                | 3.0    | Official |
| `use_adg`              | true   | XL-base-only quality booster |
| `cfg_interval_start`   | 0.0    | CFG from step 1 |
| `cfg_interval_end`     | 1.0    | CFG through last step (no truncation) |
| `infer_method`         | `ode`  | Deterministic Euler sampler |
| `batch_size`           | 1      | One take per segment |
| `duration`             | 420    | Per segment |
| `audio_format`         | `flac` | Lossless per-segment + final |
| `instrumental`         | true   | Forces `lyrics="[Instrumental]"` server-side |
| `thinking`             | false  | Keep our prompt verbatim (no LM paraphrase) |
| `seed`                 | −1     | Random per segment on first run; captured in sidecar for replay |

Every DiT sampling knob matches the published preset exactly. No quality compromises; no speed tricks inside the denoising loop.

## 5. Speedups that preserve quality (all outside the DiT sampling loop)

| Optimization | Saving per call | Mechanism |
|---|---|---|
| `thinking=false`               | ~5–15 s      | Skips the LM CoT paraphrase pass; our prompt is already self-complete. |
| `batch_size=1`                 | avoids ×N     | No duplicate candidates. |
| Sequential submission, one endpoint | ~60–90 s / seg after #1 | Rides the warm worker; avoids reloading the 35 GB DiT+LM each call. |
| `POLL_INTERVAL = 5 s` (was 10 s in prior scripts) | ~2–5 s / seg | Client-side only. |

Expected wall-clock for a full 7-segment run:
- Segment 1: ~60–90 s cold start + 4–5 min inference ≈ **5–6 min**
- Segments 2–7: warm worker ≈ **4–5 min each**
- **Total: ~28–35 min** for 46 min of finished ambient.

## 6. Stitching — ffmpeg `acrossfade` with equal-power curves

Single ffmpeg invocation, run locally after all segments exist on disk:

```bash
ffmpeg \
  -i segment_01.flac -i segment_02.flac -i segment_03.flac \
  -i segment_04.flac -i segment_05.flac -i segment_06.flac \
  -i segment_07.flac \
  -filter_complex "\
    [0][1]acrossfade=d=30:c1=qsin:c2=qsin[a01]; \
    [a01][2]acrossfade=d=30:c1=qsin:c2=qsin[a02]; \
    [a02][3]acrossfade=d=30:c1=qsin:c2=qsin[a03]; \
    [a03][4]acrossfade=d=30:c1=qsin:c2=qsin[a04]; \
    [a04][5]acrossfade=d=30:c1=qsin:c2=qsin[a05]; \
    [a05][6]acrossfade=d=30:c1=qsin:c2=qsin[out]" \
  -map "[out]" -c:a flac eno_45min_final.flac
```

- `d=30` — 30 s crossfade region, long enough to be inaudible under sustained pads.
- `c1=qsin, c2=qsin` — quarter-sine curves, the standard equal-power approximation; perceived loudness is stable through the overlap.

## 7. File layout

```
scripts/ambient_eno_45min.py                          # new orchestrator
out/ambient/eno_45min_<run-id>/                       # one directory per run
├── segment_01.flac    + segment_01.json              # audio + {seed, prompt, params, duration}
├── segment_02.flac    + segment_02.json
├── ...
├── segment_07.flac    + segment_07.json
├── manifest.json                                     # {run-id, date, endpoint, seeds[], prompt_core}
└── eno_45min_final.flac                              # stitched output
```

`<run-id>` defaults to `YYYY-MM-DD-HHMM`. Each `.json` sidecar captures the exact seed and full prompt for that segment, so a run that sounds good can be re-created deterministically.

## 8. Orchestrator behaviour

`scripts/ambient_eno_45min.py` — a single self-contained Python script following the submit/poll/save pattern from `scripts/bruno_mars_style_midnight_gold.py` (including transient-404 tolerance in the status poll loop).

### Flags

| Flag | Behaviour |
|---|---|
| *(default)* | Skip any segment whose `.flac` already exists on disk. A failed run resumes on the next invocation. |
| `--run-id ID` | Reuse an existing run directory; defaults to a fresh `YYYY-MM-DD-HHMM`. |
| `--force` | Regenerate all segments, overwriting existing. |
| `--segment N` | Regenerate only segment N. Rerun stitching afterwards. |
| `--stitch-only` | Skip all API calls; run ffmpeg on existing segments. |
| `--pin-seeds-from RUN-ID` | Reuse exact seeds from a prior run's sidecars — deterministic rebuild. |
| `--dry-run` | Print the 7 payloads without submitting. |
| `--duration SEC` | Override per-segment duration (for smoke tests only; production runs always use 420 s). |

### Pre-flight (fail fast before any GPU spend)

- `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` set
- `ffmpeg` present on `$PATH`
- `out/ambient/` writable
- Endpoint reachable via a no-op health probe

### Per-segment retry

Two layers of tolerance, distinct:

1. **In-poll 404 tolerance** — inherited from `scripts/bruno_mars_style_midnight_gold.py:213–236`. Up to 6 consecutive transient 404s on `/status/{job_id}` are swallowed without failing the job (RunPod status endpoint is briefly eventually-consistent after submission).
2. **Whole-segment retry** — if a segment ultimately returns `FAILED`, `TIMED_OUT`, `CANCELLED`, empty audio, or an `"error"` field, the orchestrator retries the whole segment up to 3 times with the same seed before giving up.

On hard failure after 3 retries the script exits non-zero; existing successful segments remain on disk so a rerun resumes from where it stopped.

## 9. Testing plan

Before committing to a full 7-segment run:

1. **`--dry-run`** — prints 7 payloads; verify prompts, params, seeds shape.
2. **`--segment 1 --duration 60`** (≈ 2 min end-to-end) — verifies the endpoint is warm, the preset is accepted, FLAC output is valid, JSON sidecar is written.

Only after those pass do we run the full 7-segment batch. Total GPU cost for the preflight smoke: ~2 min × 1 segment; negligible vs the full run.

## 10. Out of scope

- Automatic tempo / key detection across segments.
- Any form of cross-segment `complete`-chain conditioning (rejected in §2).
- Server-side stitching (no redeploy).
- Duration beyond ~46 min (further length would need either more segments with the same pattern, or a different architecture).
- LUFS/loudness normalization across segments (each segment comes from the same preset so levels should already be consistent; if audible level drift shows up post-stitch, we add a `loudnorm` pass in a follow-up).

## 11. Success criteria

- `eno_45min_final.flac` exists, duration 45:00–46:30, mono-compatible, no clicks at crossfade boundaries audible on headphones.
- Tonal palette is recognisably stable across all 7 segments (same key, same texture family).
- The "breath" arc is perceivable on an attentive listen (phase 1–3 quieter/sparser, phase 4 most reverberant, phase 5–7 widening then dissolving) without breaking ambient listenability.
- A rerun with `--pin-seeds-from <id>` reproduces the same final file bit-for-bit (modulo ffmpeg determinism).
- No changes landed in `handler.py`, `Dockerfile`, or `.github/workflows/*`.

## 12. References

- ACE-Step-1.5 INFERENCE.md — Example 9 "High-Quality Generation (Base Model)": https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md
- ACE-Step-1.5 Tutorial: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md
- acestep-v15-xl-base model card: https://huggingface.co/ACE-Step/acestep-v15-xl-base
- Existing orchestrator pattern: `scripts/bruno_mars_style_midnight_gold.py`
- Handler duration clamp: `handler.py:466` (`duration = max(10.0, min(600.0, duration))`)
- Handler src_audio constants: `handler.py:55–57`

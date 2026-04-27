"""Shared constants — both ambient_eno_45min.py and loop_music_video.py
import from here. Snapshot-tested in test_loopvid_constants.py."""
from __future__ import annotations

# ── ACE-Step XL-base official "High-Quality Generation" preset ──
# Source: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md
ACE_STEP_PRESET = {
    "inference_steps": 64,
    "guidance_scale": 8.0,
    "shift": 3.0,
    "use_adg": True,
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
    "infer_method": "ode",
}

# ── Music pipeline (60 min target) ──
# 11 × 360 − 10 × 30 = 3660s ≈ 61 min. Final mux trims to exactly 3600s.
SEGMENT_COUNT_60MIN = 11
SEGMENT_DURATION_SEC = 360
CROSSFADE_SEC = 30

# ── Video pipeline ──
CLIP_COUNT = 6
CLIP_NUM_FRAMES = 169                   # 168+1 → (n-1) % 8 == 0 for LTX-2.3
CLIP_FPS = 24
CLIP_DURATION_SEC = CLIP_NUM_FRAMES / CLIP_FPS   # = 7.0417s exact
CLIP_WIDTH = 1280
CLIP_HEIGHT = 704
INTER_CLIP_XFADE_SEC = 0.25
LOOP_SEAM_XFADE_SEC = 0.5

# ── Seedream still — hard constraints appended to every prompt ──
# Phrased as positive-form requirements because Seedream 4.5 has no
# negative_prompt field on the Replicate API.
SEEDREAM_HARD_CONSTRAINTS = (
    "Clean composition with absolutely no text, letters, numbers, words, "
    "captions, watermarks, signage, signs, neon signs, screen text, book "
    "text, handwriting, signatures, logos, brand marks, or any visible "
    "writing of any kind. No people, no human figures, no faces, no hands, "
    "no fingers. Single focal point, uncluttered background, no mirrors, "
    "no reflective glass surfaces, no transparent objects. Photographic "
    "realism, no AI-style artifacts, no oversaturation, no fake bokeh, "
    "no HDR look. Static medium-wide shot, fixed camera, no pan, no zoom, "
    "no parallax. 16:9 cinematic widescreen."
)

# ── LTX video — full negative prompt (handler default + known weaknesses) ──
LTX_NEGATIVE_PROMPT = (
    "blurry, low quality, still frame, frames, watermark, overlay, "
    "titles, has blurbox, has subtitles, "
    "text, letters, numbers, words, captions, logo, "
    "signage, signs, neon text, screen text, on-screen text, "
    "hands, fingers, face, faces, eyes, mouth, lips, "
    "fast motion, sudden movement, rapid motion, camera shake, "
    "camera pan, camera zoom, dolly, tracking shot, handheld, "
    "hard cut, scene change, scene transition, jump cut, "
    "reflections, mirror reflections, glass refraction, "
    "crowd, multiple subjects, multiple animated objects, "
    "splashing water, pouring liquid, explosion, fireworks, sparks, "
    "animal legs, paws, wings, fur detail, "
    "frame stutter, ghosting, motion smear, double exposure, "
    "morphing, warping, melting, glitch"
)

# ── Genre archetypes — closed set; LLM customizes within these ──
GENRE_ARCHETYPES = {
    "rainy_window_desk": {
        "visual": "Desk-by-rainy-window, dusk",
        "anchored": "Notebook + lamp + plant, fixed",
        "ambient_motion": "Rain on window, lamp-light flicker, steam from cup",
    },
    "mountain_ridge_dusk": {
        "visual": "Mountain ridge at golden hour",
        "anchored": "Stone cairn or lone tree mid-frame",
        "ambient_motion": "Mist drift, distant cloud cycle, grass sway",
    },
    "candle_dark_wood": {
        "visual": "Single candle on dark wood surface",
        "anchored": "Candle + brass holder, centered",
        "ambient_motion": "Flame flicker, smoke wisp, dust motes in beam",
    },
    "dim_bar_booth": {
        "visual": "Dim bar booth, bokeh windows in background",
        "anchored": "Whiskey glass + brass lamp",
        "ambient_motion": "Cigar smoke curl, bokeh shimmer, slow ceiling-fan glint",
    },
    "study_window_book": {
        "visual": "Window-lit study with sheet music",
        "anchored": "Open book + quill + window frame",
        "ambient_motion": "Curtain breath, page edge, sun-shaft dust motes",
    },
    "observatory_dome": {
        "visual": "Observatory dome interior at night",
        "anchored": "Telescope silhouette, fixed",
        "ambient_motion": "Star drift through aperture, monitor glow pulse, dust",
    },
}

"""Unit tests for scripts/ambient_eno_45min.py orchestrator."""
import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent / "scripts" / "ambient_eno_45min.py"


def _load():
    """Import the script as a module. Does NOT execute main()."""
    spec = importlib.util.spec_from_file_location("ambient_eno_45min", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ambient_eno_45min"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestConstants:
    def test_locked_palette_is_non_empty_string(self):
        m = _load()
        assert isinstance(m.LOCKED_PALETTE, str)
        assert "Eno" in m.LOCKED_PALETTE or "tonal ambient" in m.LOCKED_PALETTE
        assert len(m.LOCKED_PALETTE) > 200

    def test_segment_descriptors_has_seven_entries(self):
        m = _load()
        assert len(m.SEGMENT_DESCRIPTORS) == 7
        phases = [s["phase"] for s in m.SEGMENT_DESCRIPTORS]
        assert phases == [
            "Inhale-1", "Inhale-2", "Inhale-3", "Turn",
            "Exhale-1", "Exhale-2", "Dissolve",
        ]
        for s in m.SEGMENT_DESCRIPTORS:
            assert isinstance(s["descriptors"], str) and s["descriptors"]

    def test_preset_matches_official_xl_base_values(self):
        m = _load()
        # Verbatim from docs/en/INFERENCE.md Example 9.
        assert m.PRESET["inference_steps"] == 64
        assert m.PRESET["guidance_scale"] == 8.0
        assert m.PRESET["shift"] == 3.0
        assert m.PRESET["use_adg"] is True
        assert m.PRESET["cfg_interval_start"] == 0.0
        assert m.PRESET["cfg_interval_end"] == 1.0
        assert m.PRESET["infer_method"] == "ode"

    def test_per_segment_duration_is_420(self):
        m = _load()
        assert m.SEGMENT_DURATION_SEC == 420

    def test_crossfade_is_30s(self):
        m = _load()
        assert m.CROSSFADE_SEC == 30

    def test_segment_count_is_seven(self):
        m = _load()
        assert m.SEGMENT_COUNT == 7

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


class TestPromptBuilder:
    def test_prompt_contains_locked_palette(self):
        m = _load()
        for n in range(1, 8):
            p = m.build_segment_prompt(n)
            assert m.LOCKED_PALETTE in p

    def test_prompt_contains_correct_descriptor(self):
        m = _load()
        for n in range(1, 8):
            p = m.build_segment_prompt(n)
            expected = m.SEGMENT_DESCRIPTORS[n - 1]["descriptors"]
            assert expected in p

    def test_prompt_rejects_out_of_range_segment(self):
        m = _load()
        import pytest
        with pytest.raises(ValueError):
            m.build_segment_prompt(0)
        with pytest.raises(ValueError):
            m.build_segment_prompt(8)


class TestPayloadBuilder:
    def test_payload_has_all_preset_keys(self):
        m = _load()
        payload = m.build_payload(segment_num=1, duration=420, seed=-1)
        for k, v in m.PRESET.items():
            assert payload[k] == v, f"{k}: expected {v}, got {payload[k]}"

    def test_payload_task_type_is_text2music(self):
        m = _load()
        assert m.build_payload(1, 420, -1)["task_type"] == "text2music"

    def test_payload_is_instrumental(self):
        m = _load()
        assert m.build_payload(1, 420, -1)["instrumental"] is True

    def test_payload_thinking_is_false(self):
        m = _load()
        assert m.build_payload(1, 420, -1)["thinking"] is False

    def test_payload_audio_format_is_flac(self):
        m = _load()
        assert m.build_payload(1, 420, -1)["audio_format"] == "flac"

    def test_payload_batch_size_is_one(self):
        m = _load()
        assert m.build_payload(1, 420, -1)["batch_size"] == 1

    def test_payload_duration_and_seed_respected(self):
        m = _load()
        p = m.build_payload(3, duration=300, seed=12345)
        assert p["duration"] == 300
        assert p["seed"] == 12345

    def test_payload_prompt_matches_builder(self):
        m = _load()
        p = m.build_payload(4, 420, -1)
        assert p["prompt"] == m.build_segment_prompt(4)

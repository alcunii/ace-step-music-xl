"""Unit tests for scripts/ambient_eno_45min.py orchestrator."""
import base64
import importlib.util
import json
import sys
from pathlib import Path

import requests
import responses

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


class TestSidecar:
    def test_sidecar_roundtrip(self, tmp_path):
        m = _load()
        path = tmp_path / "segment_03.json"
        data = {
            "segment_num": 3,
            "phase": "Inhale-3",
            "seed": 42,
            "prompt": "test prompt",
            "duration_requested": 420,
            "duration_actual": 420.5,
            "sample_rate": 48000,
            "endpoint_id": "nwqnd0duxc6o38",
            "run_id": "2026-04-21-1234",
        }
        m.write_sidecar(path, data)
        got = m.read_sidecar(path)
        assert got == data

    def test_sidecar_is_human_readable(self, tmp_path):
        m = _load()
        path = tmp_path / "s.json"
        m.write_sidecar(path, {"seed": 7, "phase": "Turn"})
        text = path.read_text()
        # pretty-printed (indented) for diff-friendliness
        assert "\n" in text
        assert '"seed": 7' in text


class TestManifest:
    def test_manifest_contains_required_fields(self, tmp_path):
        m = _load()
        manifest_path = tmp_path / "manifest.json"
        m.write_manifest(
            path=manifest_path,
            run_id="2026-04-21-1234",
            endpoint_id="nwqnd0duxc6o38",
            seeds=[1, 2, 3, 4, 5, 6, 7],
            segment_duration=420,
            crossfade_sec=30,
            locked_palette="x",
        )
        got = json.loads(manifest_path.read_text())
        assert got["run_id"] == "2026-04-21-1234"
        assert got["endpoint_id"] == "nwqnd0duxc6o38"
        assert got["seeds"] == [1, 2, 3, 4, 5, 6, 7]
        assert got["segment_count"] == 7
        assert got["segment_duration_sec"] == 420
        assert got["crossfade_sec"] == 30
        assert got["locked_palette"] == "x"
        assert "written_at" in got  # ISO8601 timestamp


class TestRunPodClient:
    def test_submit_job_posts_to_runsync(self):
        m = _load()
        import responses
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "https://api.runpod.ai/v2/EP/runsync",
                json={"id": "abc123", "status": "IN_QUEUE"},
                status=200,
            )
            body = m.submit_job("EP", "key", {"input": {}})
            assert body["id"] == "abc123"
            assert body["status"] == "IN_QUEUE"
            assert len(rsps.calls) == 1
            assert "Bearer key" in rsps.calls[0].request.headers["Authorization"]

    def test_submit_job_returns_output_on_sync_complete(self):
        m = _load()
        import responses
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "https://api.runpod.ai/v2/EP/runsync",
                json={"id": "j1", "status": "COMPLETED",
                      "output": {"audio_base64": "AAA"}},
                status=200,
            )
            body = m.submit_job("EP", "key", {"input": {}})
            assert body["status"] == "COMPLETED"
            assert body["output"]["audio_base64"] == "AAA"

    def test_poll_job_returns_completed_output(self):
        m = _load()
        import responses
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "https://api.runpod.ai/v2/EP/status/job1",
                json={"status": "COMPLETED", "output": {"audio_base64": "AAA"}},
                status=200,
            )
            result = m.poll_job("EP", "key", "job1", poll_interval=0)
            assert result["status"] == "COMPLETED"
            assert result["output"]["audio_base64"] == "AAA"

    def test_poll_job_tolerates_transient_404s(self):
        m = _load()
        import responses
        with responses.RequestsMock() as rsps:
            rsps.add(responses.GET,
                     "https://api.runpod.ai/v2/EP/status/j", status=404)
            rsps.add(responses.GET,
                     "https://api.runpod.ai/v2/EP/status/j", status=404)
            rsps.add(responses.GET,
                     "https://api.runpod.ai/v2/EP/status/j",
                     json={"status": "COMPLETED", "output": {"x": 1}},
                     status=200)
            result = m.poll_job("EP", "key", "j", poll_interval=0)
            assert result["status"] == "COMPLETED"
            assert len(rsps.calls) == 3

    def test_poll_job_gives_up_after_too_many_404s(self):
        m = _load()
        import responses
        import pytest
        with responses.RequestsMock() as rsps:
            # MAX_TRANSIENT_404 + 1 consecutive 404s
            for _ in range(m.MAX_TRANSIENT_404 + 1):
                rsps.add(responses.GET,
                         "https://api.runpod.ai/v2/EP/status/j",
                         status=404)
            with pytest.raises(RuntimeError, match="404"):
                m.poll_job("EP", "key", "j", poll_interval=0)

    def test_poll_job_raises_on_terminal_failure(self):
        m = _load()
        import responses
        import pytest
        with responses.RequestsMock() as rsps:
            rsps.add(responses.GET,
                     "https://api.runpod.ai/v2/EP/status/j",
                     json={"status": "FAILED", "error": "oom"},
                     status=200)
            with pytest.raises(RuntimeError, match="FAILED"):
                m.poll_job("EP", "key", "j", poll_interval=0)


class TestRunSegment:
    def _mock_runsync_completed(self, rsps, endpoint_id, audio_b64):
        rsps.add(
            responses.POST,
            f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
            json={"id": "j1", "status": "COMPLETED",
                  "output": {"audio_base64": audio_b64, "duration": 420.0,
                             "sample_rate": 48000, "seed": 777}},
            status=200,
        )

    def test_run_segment_success_first_try(self):
        m = _load()
        import pytest
        b64 = "ZmxhYw=="  # "flac"
        with responses.RequestsMock() as rsps:
            self._mock_runsync_completed(rsps, "EP", b64)
            result = m.run_segment(
                endpoint_id="EP", api_key="key", segment_num=2,
                duration=420, seed=-1, poll_interval=0,
            )
        assert result["output"]["audio_base64"] == b64
        assert result["output"]["seed"] == 777

    def test_run_segment_retries_on_transient_failure(self, monkeypatch):
        m = _load()
        import pytest
        # Track retries using monkeypatch + a counter
        attempt_count = [0]
        original_submit = m.submit_job

        def submit_with_failures(endpoint_id, api_key, payload):
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise requests.exceptions.ConnectionError("boom")
            return {"id": "j1", "status": "COMPLETED",
                    "output": {"audio_base64": "AAA", "duration": 420}}

        monkeypatch.setattr(m, "submit_job", submit_with_failures)
        result = m.run_segment(
            endpoint_id="EP", api_key="key", segment_num=1,
            duration=420, seed=-1, poll_interval=0, retry_sleep=0,
        )
        assert result["output"]["audio_base64"] == "AAA"
        assert attempt_count[0] == 3

    def test_run_segment_gives_up_after_max_retries(self):
        m = _load()
        import pytest
        with responses.RequestsMock() as rsps:
            for _ in range(m.MAX_SEGMENT_RETRIES):
                rsps.add(responses.POST,
                         "https://api.runpod.ai/v2/EP/runsync",
                         body=requests.exceptions.ConnectionError("boom"))
            with pytest.raises(RuntimeError, match="segment 4"):
                m.run_segment(
                    endpoint_id="EP", api_key="key", segment_num=4,
                    duration=420, seed=-1, poll_interval=0, retry_sleep=0,
                )


class TestSaveFlac:
    def test_save_writes_decoded_bytes(self, tmp_path):
        m = _load()
        raw = b"notactualflacbutfine"
        output = {"audio_base64": base64.b64encode(raw).decode()}
        path = tmp_path / "s01.flac"
        m.save_flac_from_output(output, path)
        assert path.read_bytes() == raw

    def test_save_creates_parent_dir(self, tmp_path):
        m = _load()
        output = {"audio_base64": base64.b64encode(b"x").decode()}
        path = tmp_path / "deep" / "nested" / "s.flac"
        m.save_flac_from_output(output, path)
        assert path.exists()

    def test_save_raises_on_missing_audio(self, tmp_path):
        m = _load()
        import pytest
        with pytest.raises(ValueError, match="audio_base64"):
            m.save_flac_from_output({}, tmp_path / "x.flac")

    def test_save_raises_on_empty_audio(self, tmp_path):
        m = _load()
        import pytest
        with pytest.raises(ValueError, match="empty"):
            m.save_flac_from_output({"audio_base64": ""}, tmp_path / "x.flac")


class TestFFmpegCommand:
    def test_command_lists_all_inputs(self, tmp_path):
        m = _load()
        paths = [tmp_path / f"s{i:02d}.flac" for i in range(1, 8)]
        cmd = m.build_ffmpeg_command(paths, tmp_path / "out.flac",
                                      crossfade_sec=30)
        # -i appears once per input
        assert cmd.count("-i") == 7
        for p in paths:
            assert str(p) in cmd

    def test_command_uses_qsin_equal_power_curves(self, tmp_path):
        m = _load()
        paths = [tmp_path / f"s{i:02d}.flac" for i in range(1, 8)]
        cmd = m.build_ffmpeg_command(paths, tmp_path / "out.flac", 30)
        filter_str = cmd[cmd.index("-filter_complex") + 1]
        assert "c1=qsin" in filter_str
        assert "c2=qsin" in filter_str
        assert "d=30" in filter_str

    def test_command_chains_six_crossfades_for_seven_inputs(self, tmp_path):
        m = _load()
        paths = [tmp_path / f"s{i:02d}.flac" for i in range(1, 8)]
        cmd = m.build_ffmpeg_command(paths, tmp_path / "out.flac", 30)
        filter_str = cmd[cmd.index("-filter_complex") + 1]
        assert filter_str.count("acrossfade=") == 6

    def test_command_outputs_flac_codec(self, tmp_path):
        m = _load()
        paths = [tmp_path / f"s{i:02d}.flac" for i in range(1, 8)]
        cmd = m.build_ffmpeg_command(paths, tmp_path / "out.flac", 30)
        # -c:a flac must appear
        i = cmd.index("-c:a")
        assert cmd[i + 1] == "flac"

    def test_command_rejects_fewer_than_two_inputs(self, tmp_path):
        m = _load()
        import pytest
        with pytest.raises(ValueError):
            m.build_ffmpeg_command([tmp_path / "s.flac"], tmp_path / "o.flac", 30)

    def test_stitch_segments_invokes_ffmpeg(self, tmp_path, monkeypatch):
        m = _load()
        calls = []
        def fake_run(cmd, check, capture_output):
            calls.append(cmd)
            # Create an empty output file so post-checks succeed.
            Path(cmd[-1]).write_bytes(b"")
            class R:
                returncode = 0
                stdout = b""
                stderr = b""
            return R()
        monkeypatch.setattr(m.subprocess, "run", fake_run)
        paths = [tmp_path / f"s{i:02d}.flac" for i in range(1, 8)]
        for p in paths:
            p.write_bytes(b"x")
        out = tmp_path / "final.flac"
        m.stitch_segments(paths, out, crossfade_sec=30)
        assert len(calls) == 1
        assert "ffmpeg" in calls[0][0]


class TestPreflight:
    def test_passes_when_all_deps_present(self, tmp_path, monkeypatch):
        m = _load()
        monkeypatch.setattr(m.shutil, "which",
                            lambda x: "/usr/bin/ffmpeg" if x == "ffmpeg" else None)
        # should not raise
        m.preflight_checks(api_key="k", endpoint_id="EP", out_dir=tmp_path)

    def test_fails_when_api_key_missing(self, tmp_path):
        m = _load()
        import pytest
        with pytest.raises(RuntimeError, match="RUNPOD_API_KEY"):
            m.preflight_checks(api_key="", endpoint_id="EP", out_dir=tmp_path)

    def test_fails_when_endpoint_missing(self, tmp_path):
        m = _load()
        import pytest
        with pytest.raises(RuntimeError, match="endpoint"):
            m.preflight_checks(api_key="k", endpoint_id="", out_dir=tmp_path)

    def test_fails_when_ffmpeg_missing(self, tmp_path, monkeypatch):
        m = _load()
        import pytest
        monkeypatch.setattr(m.shutil, "which", lambda x: None)
        with pytest.raises(RuntimeError, match="ffmpeg"):
            m.preflight_checks(api_key="k", endpoint_id="EP", out_dir=tmp_path)


class TestCLI:
    def test_defaults(self):
        m = _load()
        args = m.parse_args([])
        assert args.run_id is None  # resolved to YYYY-MM-DD-HHMM later
        assert args.force is False
        assert args.segment is None
        assert args.stitch_only is False
        assert args.dry_run is False
        assert args.pin_seeds_from is None
        assert args.duration == m.SEGMENT_DURATION_SEC

    def test_all_flags(self):
        m = _load()
        args = m.parse_args([
            "--run-id", "run1", "--force", "--segment", "3",
            "--stitch-only", "--dry-run",
            "--pin-seeds-from", "run0", "--duration", "60",
        ])
        assert args.run_id == "run1"
        assert args.force is True
        assert args.segment == 3
        assert args.stitch_only is True
        assert args.dry_run is True
        assert args.pin_seeds_from == "run0"
        assert args.duration == 60

    def test_segment_flag_rejects_out_of_range(self):
        m = _load()
        import pytest
        with pytest.raises(SystemExit):
            m.parse_args(["--segment", "8"])
        with pytest.raises(SystemExit):
            m.parse_args(["--segment", "0"])


class TestRunDir:
    def test_resolve_run_dir_new_format(self, tmp_path):
        m = _load()
        rd = m.resolve_run_dir(base=tmp_path, run_id=None)
        assert rd.parent == tmp_path
        # Format: YYYY-MM-DD-HHMM (15 chars)
        assert len(rd.name) == 15
        assert rd.name[4] == "-" and rd.name[7] == "-" and rd.name[10] == "-"

    def test_resolve_run_dir_reuses_supplied_id(self, tmp_path):
        m = _load()
        rd = m.resolve_run_dir(base=tmp_path, run_id="my-run-42")
        assert rd == tmp_path / "my-run-42"

    def test_segment_paths_for_returns_seven(self, tmp_path):
        m = _load()
        paths = m.segment_paths_for(tmp_path)
        assert len(paths) == 7
        assert paths[0].name == "segment_01.flac"
        assert paths[6].name == "segment_07.flac"

    def test_load_pinned_seeds_from_prior_run(self, tmp_path):
        m = _load()
        prior = tmp_path / "old"
        prior.mkdir()
        for i, seed in enumerate([1, 2, 3, 4, 5, 6, 7], start=1):
            m.write_sidecar(prior / f"segment_{i:02d}.json", {"seed": seed})
        seeds = m.load_pinned_seeds(prior)
        assert seeds == [1, 2, 3, 4, 5, 6, 7]

    def test_load_pinned_seeds_missing_file_errors(self, tmp_path):
        m = _load()
        import pytest
        with pytest.raises(FileNotFoundError):
            m.load_pinned_seeds(tmp_path / "does-not-exist")

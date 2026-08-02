"""The `session.yaml` contract, and the resolved-configuration hash (INV-08, INV-11)."""

from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path
from typing import Any

import pytest
import yaml

from dnd_audio.config import (
    CONFIG_SCHEMA_VERSION,
    SessionConfig,
    config_hash,
    load_session_config,
    resolved_config,
)
from dnd_audio.errors import ConfigError


@pytest.fixture
def raw(valid_session_yaml: Path) -> dict[str, Any]:
    """The valid session as a mutable dict, for building rejection cases from."""
    loaded = yaml.safe_load(valid_session_yaml.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _reject(payload: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SessionConfig.model_validate(payload)


class TestValidSession:
    def test_loads_every_field(self, valid_session_yaml: Path) -> None:
        config = load_session_config(valid_session_yaml)

        assert config.schema_version == 1
        assert config.session_id == "2026-08-15"
        assert config.title == "Session 01"
        assert config.language == "English"
        assert config.active_tracks == "auto"
        assert len(config.tracks) == 6

        assert config.timecode.frame_rate == "30F"
        assert config.timecode.origin_date == dt.date(2026, 8, 15)
        assert config.timecode.origin_timecode is None
        assert config.timecode.rollover_policy == "infer_forward"

        assert config.asr.model == "Qwen/Qwen3-ASR-1.7B"
        assert config.asr.aligner == "Qwen/Qwen3-ForcedAligner-0.6B"
        assert config.asr.context_file == "glossary.txt"
        assert config.asr.device == "auto"
        assert config.asr.dtype == "auto"
        assert config.asr.max_segment_s == 120
        assert config.asr.max_new_tokens == 1024

        assert config.activity.correlation_max_lag_ms == 30
        assert config.mix.integrated_lufs == -16.0
        assert config.mix.true_peak_dbtp == -1.5
        assert config.mix.mp3_bitrate_kbps == 128

        assert config.recovery.allow_processed_audio is False
        assert config.recovery.source_time_overrides == {}

    def test_track_identity_comes_from_the_configured_directory(self, raw: dict[str, Any]) -> None:
        """INV-11: receiver fields validate the physical setup, never define identity."""
        config = SessionConfig.model_validate(raw)
        by_id = {track.track_id: track for track in config.tracks}
        assert by_id["tx-a"].input == "raw/tx-a"
        assert by_id["tx-a"].receiver_id == "rx-a"
        assert by_id["tx-a"].receiver_channel == 1

    def test_defaults_apply_when_sections_are_omitted(self, minimal_session_yaml: Path) -> None:
        config = load_session_config(minimal_session_yaml)
        assert config.language == "English"
        assert config.asr.max_segment_s == 120
        assert config.mix.mp3_bitrate_kbps == 128
        assert config.activity.correlation_max_lag_ms == 30


class TestRejections:
    def test_unknown_key_is_an_error(self, raw: dict[str, Any]) -> None:
        """A typo must not be silently ignored — the operator believes it took effect."""
        raw["langauge"] = "English"
        _reject(raw, "Extra inputs are not permitted")

    def test_unknown_nested_key_is_an_error(self, raw: dict[str, Any]) -> None:
        raw["asr"]["max_segment_seconds"] = 90
        _reject(raw, "Extra inputs are not permitted")

    def test_duplicate_track_id_is_rejected(self, raw: dict[str, Any]) -> None:
        raw["tracks"][1]["track_id"] = "tx-a"
        raw["tracks"][1]["input"] = "raw/tx-a"
        _reject(raw, "duplicate track_id: tx-a")

    def test_two_tracks_cannot_share_a_directory(self, raw: dict[str, Any]) -> None:
        """Which now follows from the identity rule rather than a separate check.

        Two tracks reading the same directory would have to share its name, and a
        shared name is a duplicate `track_id`. Kept as a test because the property is
        what matters, not the route the validator takes to it.
        """
        raw["tracks"][1]["input"] = "raw/tx-a"
        _reject(raw, "is the track's identity")

    def test_duplicate_receiver_channel_is_rejected(self, raw: dict[str, Any]) -> None:
        """Two transmitters cannot be channel 1 of the same receiver."""
        raw["tracks"][1]["receiver_channel"] = 1
        _reject(raw, "duplicate receiver_id/receiver_channel")

    def test_receiver_channel_out_of_range_is_rejected(self, raw: dict[str, Any]) -> None:
        raw["tracks"][0]["receiver_channel"] = 3
        _reject(raw, "less than or equal to 2")

    def test_empty_roster_is_rejected(self, raw: dict[str, Any]) -> None:
        raw["tracks"] = []
        _reject(raw, "at least 1 item")

    @pytest.mark.parametrize(
        "value", ["/etc/passwd", "../elsewhere/tx-a", "raw/../../escape", " raw/tx-a"]
    )
    def test_input_must_stay_inside_the_session(self, raw: dict[str, Any], value: str) -> None:
        """INV-01 depends on paths being provably inside the session tree."""
        raw["tracks"][0]["input"] = value
        _reject(raw, "relative|escape|whitespace")

    def test_a_track_may_not_read_another_tracks_directory(self, raw: dict[str, Any]) -> None:
        """INV-11: one transposed letter would mis-attribute a whole session.

        `track_id: tx-a` reading `raw/tx-f` is unique, well formed, and silently wrong
        — every word Frank says becomes Alice's, and nothing downstream can notice.
        """
        raw["tracks"][0]["input"] = "raw/tx-f"
        raw["tracks"][5]["input"] = "raw/tx-a"
        _reject(raw, "is the track's identity")

    def test_swapping_two_tracks_directories_is_rejected_even_though_both_are_unique(
        self, raw: dict[str, Any]
    ) -> None:
        """Uniqueness checks alone cannot see a swap: both sides stay unique."""
        raw["tracks"][0]["input"], raw["tracks"][1]["input"] = (
            raw["tracks"][1]["input"],
            raw["tracks"][0]["input"],
        )
        _reject(raw, "INV-11")

    def test_the_directory_may_live_anywhere_as_long_as_it_is_named_for_the_track(
        self, raw: dict[str, Any]
    ) -> None:
        raw["tracks"][0]["input"] = "raw/2026-08-15/kit-a/tx-a"
        assert SessionConfig.model_validate(raw).tracks[0].input == "raw/2026-08-15/kit-a/tx-a"

    def test_active_tracks_must_name_configured_tracks(self, raw: dict[str, Any]) -> None:
        """An unconfigured directory must never be attributed to a speaker."""
        raw["active_tracks"] = ["tx-a", "tx-z"]
        _reject(raw, "not in the roster")

    def test_active_tracks_rejects_duplicates(self, raw: dict[str, Any]) -> None:
        raw["active_tracks"] = ["tx-a", "tx-a"]
        _reject(raw, "duplicate active_tracks")

    def test_active_tracks_rejects_an_empty_list(self, raw: dict[str, Any]) -> None:
        raw["active_tracks"] = []
        _reject(raw, "non-empty list")

    def test_explicit_active_tracks_is_accepted(self, raw: dict[str, Any]) -> None:
        raw["active_tracks"] = ["tx-a", "tx-b"]
        assert SessionConfig.model_validate(raw).active_tracks == ["tx-a", "tx-b"]

    def test_max_segment_s_is_capped(self, raw: dict[str, Any]) -> None:
        """The package's timestamp path chunks at 180 s; 120 is the documented ceiling.

        Assumes OQ-009 — if the real chunking point differs, this cap moves with it.
        """
        raw["asr"]["max_segment_s"] = 180
        _reject(raw, "less than or equal to 120")

    def test_max_new_tokens_must_be_positive(self, raw: dict[str, Any]) -> None:
        raw["asr"]["max_new_tokens"] = 0
        _reject(raw, "greater than 0")

    def test_model_and_aligner_revisions_are_accepted(self, raw: dict[str, Any]) -> None:
        """The spec allows pinning revisions in configuration rather than in the lock."""
        raw["asr"]["model_revision"] = "0f1e2d3c"
        raw["asr"]["aligner_revision"] = "4b5a6978"
        config = SessionConfig.model_validate(raw)
        assert config.asr.model_revision == "0f1e2d3c"
        assert config.asr.aligner_revision == "4b5a6978"

    def test_unknown_device_is_rejected(self, raw: dict[str, Any]) -> None:
        raw["asr"]["device"] = "rocm"
        _reject(raw, "'auto', 'cpu' or 'cuda'")

    def test_invalid_mp3_bitrate_is_rejected(self, raw: dict[str, Any]) -> None:
        raw["mix"]["mp3_bitrate_kbps"] = 130
        _reject(raw, "not an MPEG-1 Layer III bitrate")

    def test_positive_loudness_target_is_rejected(self, raw: dict[str, Any]) -> None:
        raw["mix"]["integrated_lufs"] = 3.0
        _reject(raw, "less than or equal to 0")

    def test_unknown_frame_rate_label_is_rejected(self, raw: dict[str, Any]) -> None:
        raw["timecode"]["frame_rate"] = "29.97"
        _reject(raw, "Input should be")

    def test_origin_timecode_requires_a_date(self, raw: dict[str, Any]) -> None:
        raw["timecode"]["origin_date"] = None
        raw["timecode"]["origin_timecode"] = "19:00:00:00"
        _reject(raw, "requires timecode.origin_date")

    def test_origin_timecode_is_validated_against_the_rate(self, raw: dict[str, Any]) -> None:
        raw["timecode"]["origin_timecode"] = "19:00:00;00"
        _reject(raw, "drop-frame notation")

    def test_origin_timecode_is_accepted_when_consistent(self, raw: dict[str, Any]) -> None:
        raw["timecode"]["origin_timecode"] = "19:00:00:00"
        assert SessionConfig.model_validate(raw).timecode.origin_timecode == "19:00:00:00"

    def test_unknown_rollover_policy_is_rejected(self, raw: dict[str, Any]) -> None:
        raw["timecode"]["rollover_policy"] = "guess"
        _reject(raw, "Input should be")


class TestRecoveryOverrides:
    @staticmethod
    def _with_override(raw: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        raw["recovery"]["source_time_overrides"] = {
            "raw/tx-a/TX01_MIC002_20260815_190000_orig.wav": override
        }
        return raw

    def test_a_timecode_override_is_accepted(self, raw: dict[str, Any]) -> None:
        payload = self._with_override(
            raw,
            {
                "sha256": "a" * 64,
                "recording_date": "2026-08-15",
                "start_timecode": "19:00:00:00",
                "reason": "BWF time reference was damaged; value from the field log",
            },
        )
        config = SessionConfig.model_validate(payload)
        override = next(iter(config.recovery.source_time_overrides.values()))
        assert override.start_timecode == "19:00:00:00"
        assert override.start_offset_samples is None

    def test_a_sample_offset_override_is_accepted_and_may_be_negative(
        self, raw: dict[str, Any]
    ) -> None:
        payload = self._with_override(
            raw, {"start_offset_samples": -48_000, "reason": "measured against the clap"}
        )
        override = next(
            iter(SessionConfig.model_validate(payload).recovery.source_time_overrides.values())
        )
        assert override.start_offset_samples == -48_000

    def test_both_timing_values_are_rejected(self, raw: dict[str, Any]) -> None:
        payload = self._with_override(
            raw,
            {"start_timecode": "19:00:00:00", "start_offset_samples": 0, "reason": "both"},
        )
        _reject(payload, "not both")

    def test_an_override_with_no_timing_information_is_rejected(self, raw: dict[str, Any]) -> None:
        """A reason and a hash override nothing; accepting it would hide a mistake."""
        payload = self._with_override(raw, {"sha256": "b" * 64, "reason": "nothing to apply"})
        _reject(payload, "at least one of")

    def test_a_date_alone_is_enough(self, raw: dict[str, Any]) -> None:
        """Supplying the recording date resolves a rollover ambiguity by itself."""
        payload = self._with_override(
            raw, {"recording_date": "2026-08-16", "reason": "session crossed midnight"}
        )
        override = next(
            iter(SessionConfig.model_validate(payload).recovery.source_time_overrides.values())
        )
        assert override.recording_date == dt.date(2026, 8, 16)

    def test_reason_is_required(self, raw: dict[str, Any]) -> None:
        payload = self._with_override(raw, {"start_offset_samples": 10})
        _reject(payload, "reason")

    def test_malformed_hash_is_rejected(self, raw: dict[str, Any]) -> None:
        payload = self._with_override(
            raw, {"sha256": "NOTAHASH", "start_offset_samples": 0, "reason": "x"}
        )
        _reject(payload, "should match pattern")

    def test_override_timecode_is_validated_against_the_session_rate(
        self, raw: dict[str, Any]
    ) -> None:
        payload = self._with_override(
            raw, {"start_timecode": "19:00:00;00", "reason": "drop-frame at 30F"}
        )
        _reject(payload, "drop-frame notation")

    def test_override_key_must_be_a_session_relative_path(self, raw: dict[str, Any]) -> None:
        raw["recovery"]["source_time_overrides"] = {
            "/tmp/elsewhere.wav": {"start_offset_samples": 0, "reason": "x"}
        }
        _reject(raw, "relative")

    def test_override_keys_are_normalized(self, raw: dict[str, Any]) -> None:
        """An un-normalized key hashes differently and is never found by a lookup.

        `raw//tx-a/./f.wav` and `raw/tx-a/f.wav` name the same file. Keeping both
        spellings would give two configs that describe identical sessions different
        cache identities (INV-08), and an override written the long way would silently
        never apply.
        """
        raw["recovery"]["source_time_overrides"] = {
            "raw//tx-a/./TX01_MIC002_20260815_190000_orig.wav": {
                "start_offset_samples": 0,
                "reason": "written with a redundant path",
            }
        }
        config = SessionConfig.model_validate(raw)
        assert list(config.recovery.source_time_overrides) == [
            "raw/tx-a/TX01_MIC002_20260815_190000_orig.wav"
        ]

    def test_two_spellings_of_the_same_override_are_rejected(self, raw: dict[str, Any]) -> None:
        raw["recovery"]["source_time_overrides"] = {
            "raw/tx-a/f.wav": {"start_offset_samples": 0, "reason": "one"},
            "raw//tx-a/./f.wav": {"start_offset_samples": 100, "reason": "two"},
        }
        _reject(raw, "same file")

    def test_normalized_overrides_hash_identically(self, raw: dict[str, Any]) -> None:
        plain = copy.deepcopy(raw)
        plain["recovery"]["source_time_overrides"] = {
            "raw/tx-a/f.wav": {"start_offset_samples": 0, "reason": "same override"}
        }
        redundant = copy.deepcopy(raw)
        redundant["recovery"]["source_time_overrides"] = {
            "raw/./tx-a//f.wav": {"start_offset_samples": 0, "reason": "same override"}
        }
        assert config_hash(SessionConfig.model_validate(plain)) == config_hash(
            SessionConfig.model_validate(redundant)
        )


class TestLoader:
    def test_missing_file_is_a_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="cannot read"):
            load_session_config(tmp_path / "absent.yaml")

    def test_invalid_yaml_is_a_config_error(self, tmp_path: Path) -> None:
        path = tmp_path / "session.yaml"
        path.write_text("tracks: [\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_session_config(path)

    def test_non_mapping_is_a_config_error(self, tmp_path: Path) -> None:
        path = tmp_path / "session.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="must contain a YAML mapping"):
            load_session_config(path)

    def test_validation_failure_names_the_field(self, tmp_path: Path, raw: dict[str, Any]) -> None:
        raw["mix"]["mp3_bitrate_kbps"] = 130
        path = tmp_path / "session.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ConfigError, match="mp3_bitrate_kbps"):
            load_session_config(path)


class TestResolvedConfigHash:
    """INV-08: what a cache key means by "the configuration"."""

    def test_omitted_defaults_hash_the_same_as_stated_ones(
        self, valid_session_yaml: Path, minimal_session_yaml: Path
    ) -> None:
        stated = load_session_config(valid_session_yaml)
        omitted = load_session_config(minimal_session_yaml)
        assert config_hash(stated) == config_hash(omitted)

    def test_key_order_in_the_file_does_not_matter(self, raw: dict[str, Any]) -> None:
        reordered = dict(reversed(list(raw.items())))
        assert config_hash(SessionConfig.model_validate(raw)) == config_hash(
            SessionConfig.model_validate(reordered)
        )

    @pytest.mark.parametrize(
        ("section", "field", "value"),
        [
            ("asr", "max_new_tokens", 512),
            ("asr", "max_segment_s", 90),
            ("asr", "model", "Qwen/Qwen3-ASR-Flash"),
            ("asr", "model_revision", "deadbeef"),
            ("asr", "dtype", "float32"),
            ("activity", "correlation_max_lag_ms", 45),
            ("mix", "integrated_lufs", -18.0),
            ("mix", "mp3_bitrate_kbps", 192),
            ("timecode", "frame_rate", "29.97DF"),
        ],
    )
    def test_every_output_affecting_field_changes_the_hash(
        self, raw: dict[str, Any], section: str, field: str, value: object
    ) -> None:
        baseline = config_hash(SessionConfig.model_validate(raw))
        changed = copy.deepcopy(raw)
        changed[section][field] = value
        assert config_hash(SessionConfig.model_validate(changed)) != baseline

    def test_the_projection_carries_its_schema_version(self, raw: dict[str, Any]) -> None:
        """A field whose meaning changed must invalidate caches even at the same value."""
        projection = resolved_config(SessionConfig.model_validate(raw))
        assert projection["config_schema_version"] == CONFIG_SCHEMA_VERSION

    def test_the_projection_contains_no_host_specific_paths(self, raw: dict[str, Any]) -> None:
        """Two machines with the same session must agree on the hash."""
        projection = resolved_config(SessionConfig.model_validate(raw))
        session = projection["session"]
        assert all(not track["input"].startswith("/") for track in session["tracks"])

    def test_reordering_the_roster_does_not_change_the_hash(self, raw: dict[str, Any]) -> None:
        """The roster is a set keyed by track_id; its order in the file means nothing.

        Without normalization, moving a track up the list would invalidate every cached
        result for a session that is identical in every way that affects output.
        """
        reordered = copy.deepcopy(raw)
        reordered["tracks"] = list(reversed(reordered["tracks"]))
        assert config_hash(SessionConfig.model_validate(raw)) == config_hash(
            SessionConfig.model_validate(reordered)
        )

    def test_reordering_active_tracks_does_not_change_the_hash(self, raw: dict[str, Any]) -> None:
        first = copy.deepcopy(raw)
        first["active_tracks"] = ["tx-a", "tx-b", "tx-c"]
        second = copy.deepcopy(raw)
        second["active_tracks"] = ["tx-c", "tx-a", "tx-b"]
        assert config_hash(SessionConfig.model_validate(first)) == config_hash(
            SessionConfig.model_validate(second)
        )

    def test_changing_which_tracks_are_active_does_change_the_hash(
        self, raw: dict[str, Any]
    ) -> None:
        """Sorting must not go so far as to erase a real difference."""
        first = copy.deepcopy(raw)
        first["active_tracks"] = ["tx-a", "tx-b"]
        second = copy.deepcopy(raw)
        second["active_tracks"] = ["tx-a", "tx-c"]
        assert config_hash(SessionConfig.model_validate(first)) != config_hash(
            SessionConfig.model_validate(second)
        )

    def test_the_hash_is_stable_across_runs(self, valid_session_yaml: Path) -> None:
        first = config_hash(load_session_config(valid_session_yaml))
        second = config_hash(load_session_config(valid_session_yaml))
        assert first == second

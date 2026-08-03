"""The `session.yaml` contract, and the resolved-configuration hash (INV-08, INV-11)."""

from __future__ import annotations

import copy
import datetime as dt
import math
from pathlib import Path
from typing import Any, Final, get_args

import pytest
import yaml
from pydantic import BaseModel

from dnd_audio.config import (
    _FIELD_SCOPES,
    CONFIG_SCHEMA_VERSION,
    BleedConfig,
    MixConfig,
    ScoringConfig,
    SessionConfig,
    StageScope,
    VadConfig,
    config_hash,
    load_session_config,
    resolved_config,
    stage_config,
    stage_config_hash,
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


class TestTheEnvelopeGridIsExact:
    """ADR-0022: three separate properties, and the second does not follow from the first."""

    def test_a_control_rate_that_does_not_divide_the_session_grid_is_rejected(
        self, raw: dict[str, Any]
    ) -> None:
        raw["mix"] = {"envelope": {"control_rate_hz": 7000}}
        _reject(raw, "does not divide the 48000 Hz session grid")

    @pytest.mark.parametrize("field", ["attack_ms", "release_ms"])
    def test_a_time_that_is_not_a_whole_number_of_control_frames_is_rejected(
        self, raw: dict[str, Any], field: str
    ) -> None:
        """800 Hz divides 48000 and an 11 ms attack is 8.8 frames of it.

        The plan review's finding, kept as a test rather than as a comment: stating that the
        rate divides the sample rate accounts for only half of what the grid needs.
        """
        raw["mix"] = {"envelope": {"control_rate_hz": 800, field: 11}}
        _reject(raw, "not a whole number")

    def test_the_same_times_are_accepted_at_a_rate_that_divides_them(
        self, raw: dict[str, Any]
    ) -> None:
        """The contrast: 11 ms is exact at 1000 Hz, so the refusal above is about the pair."""
        raw["mix"] = {"envelope": {"control_rate_hz": 1000, "attack_ms": 11, "release_ms": 11}}
        assert SessionConfig.model_validate(raw).mix.envelope.attack_ms == 11

    def test_an_unachievable_dominance_margin_is_rejected(self, raw: dict[str, Any]) -> None:
        """The first draft of M5's plan proposed exactly these numbers.

        `20*log10(0.5/0.02) = 27.96 dB` of separation, eroded by twice a 12 dB clamp, leaves
        3.96 dB against a promised 20 — a mix that would fail its own gate on the first
        session where a correction was needed, silently.
        """
        raw["mix"] = {
            "envelope": {
                "room_tone_share": 0.02,
                "min_active_share": 0.5,
                "max_level_correction_db": 12.0,
                "solo_attenuation_margin_db": 20.0,
            }
        }
        _reject(raw, "is not achievable")

    def test_the_shipped_defaults_are_achievable_with_room_to_spare(self) -> None:
        """40 dB of separation, 12 of it erodible, against a 20 dB promise."""
        envelope = MixConfig().envelope
        separation = 20.0 * math.log10(envelope.min_active_share / envelope.room_tone_share)
        guaranteed = separation - 2.0 * envelope.max_level_correction_db
        assert separation == pytest.approx(40.0)
        assert guaranteed >= envelope.solo_attenuation_margin_db + 5.0

    def test_the_default_attack_finishes_inside_the_default_vad_pad(self) -> None:
        """ADR-0022's stated reason for `attack_ms = 10`, pinned rather than left in prose.

        The candidate the envelope opens on is already padded by `activity.vad.pad_ms`, so a
        ramp shorter than the pad has the channel open before the word starts. Raising the
        attack past the pad is how a first phoneme is lost, and nothing else would notice.
        """
        assert MixConfig().envelope.attack_ms < VadConfig().pad_ms

    def test_an_unachievable_overlap_floor_is_rejected(self, raw: dict[str, Any]) -> None:
        """The twin of the dominance validator, and the one that was missing.

        The gate's overlap criterion was a property of the score combinations the tests used
        rather than of the rule: a speaker scoring zero beside one scoring 1000, cut by the
        permitted 6 dB, lands at -15.66 dB against the -15 that shipped. Found by M5's code
        review; the promise now has to be one the share rule can keep.
        """
        raw["mix"] = {"envelope": {"overlap_min_gain_db": -15.0}}
        _reject(raw, "is not achievable across")

    def test_the_shipped_overlap_floor_is_what_the_rule_guarantees(self) -> None:
        """Derived rather than estimated, and stated to two decimals so it cannot drift."""
        envelope = MixConfig().envelope
        assert envelope.guaranteed_overlap_gain_db(6) == pytest.approx(-15.66, abs=0.01)
        assert envelope.guaranteed_overlap_gain_db(6) >= envelope.overlap_min_gain_db

    def test_more_tracks_divide_the_overlap_guarantee_further(self) -> None:
        """Why this validator lives on `SessionConfig`: the bound depends on the roster."""
        envelope = MixConfig().envelope
        assert envelope.guaranteed_overlap_gain_db(12) < envelope.guaranteed_overlap_gain_db(6)

    def test_release_is_longer_than_attack(self) -> None:
        """The spec's "short attack and longer release", as a property of the defaults."""
        envelope = MixConfig().envelope
        assert envelope.release_ms > envelope.attack_ms


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


_STAGES: Final[tuple[StageScope, ...]] = get_args(StageScope)

#: Where in a session payload to write a value, as a path of mapping keys and list indices.
_Path = tuple[str | int, ...]

#: A different but legal value for every field :data:`_FIELD_SCOPES` classifies, keyed by the
#: same paths. ``None`` means the field has exactly one legal value and therefore cannot move
#: a hash at all. A test asserts this table covers the projection table, so a new
#: configuration section cannot be classified without also being exercised in both directions.
_ALTERNATIVES: Final[dict[str, tuple[_Path, Any] | None]] = {
    "schema_version": None,  # Literal[1]: there is no other value to try.
    "session_id": (("session_id",), "2026-09-19"),
    "tracks": (("tracks", 0, "speaker_name"), "Alice Liddell"),
    "active_tracks": (("active_tracks",), ["tx-a", "tx-b"]),
    "timecode": (("timecode", "frame_rate"), "25F"),
    "recovery": (("recovery", "allow_processed_audio"), True),
    "activity.vad": (("activity", "vad", "speech_threshold"), 0.6),
    "activity.bleed": (("activity", "bleed", "min_correlation"), 0.75),
    "activity.scoring": (("activity", "scoring", "level_weight"), 0.5),
    "activity.correlation_max_lag_ms": (("activity", "correlation_max_lag_ms"), 45),
    "title": (("title",), "Session 02"),
    "language": (("language",), "German"),
    "asr": (("asr", "max_new_tokens"), 512),
    "transcript": (("transcript", "pad_ms"), 250),
    "sync_qa": (("sync_qa", "enabled"), True),
    "mix.envelope": (("mix", "envelope", "release_ms"), 250),
    "mix.encode": (("mix", "encode", "max_retries"), 2),
    "mix.integrated_lufs": (("mix", "integrated_lufs"), -18.0),
    "mix.true_peak_dbtp": (("mix", "true_peak_dbtp"), -2.0),
    "mix.mp3_bitrate_kbps": (("mix", "mp3_bitrate_kbps"), 192),
}

#: ADR-0016's table, transcribed from the decision rather than derived from the code. The
#: parameterized tests below are generated *from* `_FIELD_SCOPES`, which proves the
#: projections behave as the table says — and would happily follow the table anywhere,
#: including into a stage that must not depend on a section. This is the fixed point that
#: makes widening a projection a decision somebody has to record here as well as there.
_ADR_0016_PROJECTIONS: Final[dict[StageScope, frozenset[str]]] = {
    "inspection": frozenset(
        {"schema_version", "session_id", "tracks", "active_tracks", "timecode", "recovery"}
    ),
    "derivative": frozenset(
        {"schema_version", "session_id", "tracks", "active_tracks", "timecode", "recovery"}
    ),
    "detection": frozenset({"schema_version", "session_id", "tracks", "activity.vad"}),
    "attribution": frozenset(
        {
            "schema_version",
            "session_id",
            "tracks",
            "activity.vad",
            "activity.bleed",
            "activity.scoring",
            "activity.correlation_max_lag_ms",
        }
    ),
    # ADR-0023's render boundary. Placement is deliberately absent: the render identity
    # carries the timeline's sha256 and the graph's attribution key, each already downstream
    # of every placement section, and the encode settings are absent because they reach only
    # the MP3, which is never cached.
    "mix": frozenset({"schema_version", "session_id", "tracks", "mix.envelope"}),
}

_MOVABLE: Final[tuple[str, ...]] = tuple(
    sorted(path for path, alternative in _ALTERNATIVES.items() if alternative is not None)
)

_INCLUDED: Final[list[tuple[StageScope, str]]] = [
    (stage, path) for path in _MOVABLE for stage in _STAGES if stage in _FIELD_SCOPES[path]
]

_EXCLUDED: Final[list[tuple[StageScope, str]]] = [
    (stage, path) for path in _MOVABLE for stage in _STAGES if stage not in _FIELD_SCOPES[path]
]


def _stage_hashes(config: SessionConfig) -> dict[StageScope, str]:
    """Every stage's cache identity at once, so a test can compare them side by side."""
    return {stage: stage_config_hash(config, stage) for stage in _STAGES}


def _with_alternative(raw: dict[str, Any], path: str) -> SessionConfig:
    """The same session with the field at ``path`` set to a different legal value."""
    alternative = _ALTERNATIVES[path]
    assert alternative is not None, f"{path} has no alternative value to write"
    keys, value = alternative
    payload = copy.deepcopy(raw)
    target: Any = payload
    for key in keys[:-1]:
        target = target[key] if isinstance(key, int) else target.setdefault(key, {})
    target[keys[-1]] = value
    return SessionConfig.model_validate(payload)


class TestActivityConfigAndStageScopes:
    """M3's three new sections, and the per-stage cache projections of ADR-0016."""

    def test_equal_vad_thresholds_are_rejected(self, raw: dict[str, Any]) -> None:
        """Two identical thresholds are one threshold: no hysteresis at all.

        A probability wobbling across a single threshold mid-syllable chops a word in half,
        which is the whole reason there are two of them.
        """
        raw["activity"]["vad"] = {"speech_threshold": 0.5, "silence_threshold": 0.5}
        _reject(raw, "must be below")

    def test_inverted_vad_thresholds_are_rejected(self, raw: dict[str, Any]) -> None:
        """Silence above speech would start a region where it ends."""
        raw["activity"]["vad"] = {"speech_threshold": 0.4, "silence_threshold": 0.6}
        _reject(raw, "must be below")

    def test_scoring_weights_summing_to_zero_are_rejected(self, raw: dict[str, Any]) -> None:
        """Every candidate would score the same, and the bleed gate could prefer nothing.

        Rejected rather than defaulted, because a normalization by the sum would also
        divide by zero — and a gate that silently never suppresses looks like a gate that
        found no bleed.
        """
        raw["activity"]["scoring"] = {
            "level_weight": 0.0,
            "confidence_weight": 0.0,
            "dominance_weight": 0.0,
            "correlation_weight": 0.0,
        }
        _reject(raw, "sum to zero")

    @pytest.mark.parametrize(
        ("section", "field", "value", "match"),
        [
            ("vad", "speech_threshold", 1.0, "less than 1"),
            ("vad", "pad_ms", 2000, "less than or equal to 1000"),
            ("bleed", "min_correlation", 0.0, "greater than 0"),
            ("bleed", "correlation_window_ms", 60_000, "less than or equal to 30000"),
            ("scoring", "level_span_db", 0.0, "greater than 0"),
            ("scoring", "correlation_weight", 1.5, "less than or equal to 1"),
        ],
    )
    def test_out_of_range_activity_values_are_rejected(
        self, raw: dict[str, Any], section: str, field: str, value: object, match: str
    ) -> None:
        """A representative bound per section, including the one holding an array (INV-07)."""
        raw["activity"][section] = {field: value}
        _reject(raw, match)

    def test_an_omitted_activity_section_hashes_like_a_fully_stated_one(
        self, raw: dict[str, Any]
    ) -> None:
        """INV-08, extended to M3's fields — and a pin on what the defaults are.

        The stated values below are written out independently of the model. If a default
        moves without this file moving with it, the two payloads stop agreeing and this
        fails, which is the point: an operator who writes out the current defaults must not
        thereby invalidate every cached artifact in the session.
        """
        stated_vad = {
            "speech_threshold": 0.5,
            "silence_threshold": 0.35,
            "min_speech_ms": 250,
            "min_silence_ms": 100,
            "merge_gap_ms": 200,
            "pad_ms": 30,
        }
        stated_bleed = {
            "min_score_margin": 0.15,
            "min_correlation": 0.5,
            "veto_db": 12.0,
            "correlation_window_ms": 2000,
            "min_reference_candidates": 3,
        }
        stated_scoring = {
            "level_weight": 0.35,
            "confidence_weight": 0.25,
            "dominance_weight": 0.25,
            "correlation_weight": 0.15,
            "level_span_db": 30.0,
            "dominance_span_db": 20.0,
        }
        # Stating *every* default, not merely some: a field missing here would make the
        # comparison below prove nothing about it.
        assert set(stated_vad) == set(VadConfig.model_fields)
        assert set(stated_bleed) == set(BleedConfig.model_fields)
        assert set(stated_scoring) == set(ScoringConfig.model_fields)

        omitted = copy.deepcopy(raw)
        del omitted["activity"]
        stated = copy.deepcopy(raw)
        stated["activity"] = {
            "correlation_max_lag_ms": 30,
            "vad": stated_vad,
            "bleed": stated_bleed,
            "scoring": stated_scoring,
        }

        assert config_hash(SessionConfig.model_validate(omitted)) == config_hash(
            SessionConfig.model_validate(stated)
        )

    def test_every_classified_field_has_an_alternative_value_to_try(self) -> None:
        """Both projection tests below are parameterized from this table, not by hand.

        A new section added to `_FIELD_SCOPES` and forgotten here would silently be tested
        in neither direction.
        """
        assert set(_ALTERNATIVES) == set(_FIELD_SCOPES)

    def test_every_configuration_field_is_classified(self) -> None:
        """ADR-0016: a new section must be classified deliberately, not default to nothing.

        Derived from the model rather than a hand-written list, so adding a field to
        `SessionConfig` without deciding which stages it can change fails here.
        """
        for name, field in SessionConfig.model_fields.items():
            if name in _FIELD_SCOPES:
                continue
            nested = field.annotation
            unclassified = f"{name} is classified neither as a whole nor field by field"
            assert isinstance(nested, type), unclassified
            assert issubclass(nested, BaseModel), unclassified
            classified = {
                path.partition(".")[2] for path in _FIELD_SCOPES if path.startswith(f"{name}.")
            }
            assert classified == set(nested.model_fields), (
                f"{name} is classified field by field, but {classified} does not cover it"
            )

    def test_the_table_classifies_nothing_that_does_not_exist(self) -> None:
        for path in _FIELD_SCOPES:
            head, _, tail = path.partition(".")
            assert head in SessionConfig.model_fields, f"{path} names no field of SessionConfig"
            if not tail:
                continue
            nested = SessionConfig.model_fields[head].annotation
            assert isinstance(nested, type)
            assert issubclass(nested, BaseModel)
            assert tail in nested.model_fields, f"{path} names no field of {head}"

    @pytest.mark.parametrize("stage", _STAGES)
    def test_each_projection_is_the_one_the_decision_records(self, stage: StageScope) -> None:
        """The tests below follow `_FIELD_SCOPES`; this one holds `_FIELD_SCOPES` to ADR-0016.

        Everything else here is generated from the table, so a section quietly *added* to a
        projection makes the generated tests agree with it and stay green. Widening is the
        cheap direction to get wrong — it costs only recomputation — but it is still a
        change to a recorded decision, and it should not be possible to make one without
        this failing.
        """
        recorded = {path for path, stages in _FIELD_SCOPES.items() if stage in stages}

        assert recorded == set(_ADR_0016_PROJECTIONS[stage])

    @pytest.mark.parametrize(("stage", "path"), _INCLUDED)
    def test_changing_an_included_section_changes_that_stages_hash(
        self, raw: dict[str, Any], stage: StageScope, path: str
    ) -> None:
        baseline = stage_config_hash(SessionConfig.model_validate(raw), stage)

        assert stage_config_hash(_with_alternative(raw, path), stage) != baseline

    @pytest.mark.parametrize(("stage", "path"), _EXCLUDED)
    def test_changing_an_excluded_section_leaves_that_stages_hash_alone(
        self, raw: dict[str, Any], stage: StageScope, path: str
    ) -> None:
        """The half that usually goes untested, and the half a projection dies of.

        Testing only that included sections move the hash lets a projection quietly narrow:
        the key still changes for every reason somebody thought to assert, and stops
        changing for one nobody did — which serves a stale artifact as current, silently.
        Asserted here for every excluded section rather than for the few that seemed
        interesting.
        """
        baseline = stage_config_hash(SessionConfig.model_validate(raw), stage)

        assert stage_config_hash(_with_alternative(raw, path), stage) == baseline

    def test_tuning_the_bleed_gate_rebuilds_neither_pcm_nor_inference(
        self, raw: dict[str, Any]
    ) -> None:
        """ADR-0016's reason for existing, stated as the operator experiences it.

        OQ-017 guarantees `min_correlation` gets tuned repeatedly against real sessions.
        Under whole-configuration hashing — which `config_hash` still does, as asserted
        here — every hundredth of a change would rebuild gigabytes of 16 kHz PCM that
        provably cannot depend on it, and re-run every VAD pass.
        """
        baseline = SessionConfig.model_validate(raw)
        tuned = _with_alternative(raw, "activity.bleed")
        before, after = _stage_hashes(baseline), _stage_hashes(tuned)

        assert after["derivative"] == before["derivative"]
        assert after["detection"] == before["detection"]
        assert after["attribution"] != before["attribution"]
        assert config_hash(tuned) != config_hash(baseline)

    def test_tuning_the_vad_rebuilds_detection_but_not_the_derivative(
        self, raw: dict[str, Any]
    ) -> None:
        """Detection is inference over the derivative, and attribution consumes detections.

        So a VAD change must reach both of those and neither of the placement stages.
        """
        baseline = SessionConfig.model_validate(raw)
        tuned = _with_alternative(raw, "activity.vad")
        before, after = _stage_hashes(baseline), _stage_hashes(tuned)

        assert after["detection"] != before["detection"]
        assert after["attribution"] != before["attribution"]
        assert after["derivative"] == before["derivative"]
        assert after["inspection"] == before["inspection"]

    @pytest.mark.parametrize("stage", _STAGES)
    def test_a_projection_carries_no_key_the_table_does_not_grant_it(
        self, raw: dict[str, Any], stage: StageScope
    ) -> None:
        """What the hash covers is exactly what the table says, in both directions."""
        granted = {path for path, stages in _FIELD_SCOPES.items() if stage in stages}
        projection = stage_config(SessionConfig.model_validate(raw), stage)

        assert projection["stage"] == stage
        assert projection["config_schema_version"] == CONFIG_SCHEMA_VERSION
        for head, value in projection["session"].items():
            if head in granted:
                continue
            assert isinstance(value, dict), f"{stage} carries ungranted {head}"
            for tail in value:
                assert f"{head}.{tail}" in granted, f"{stage} carries ungranted {head}.{tail}"

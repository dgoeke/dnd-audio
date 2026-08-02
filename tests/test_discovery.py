"""Selection and roster rules, one test per rule the spec states.

The tests here mostly build small purpose-made sessions rather than using the canonical
fixture, because each rule is about a specific malformed session and a fixture that had
every defect at once would prove nothing about any of them.

Every assertion checks the *reason code* as well as the outcome. A rule that produces
the right answer with the wrong explanation is a rule that will be "fixed" incorrectly
the first time someone reads the report.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from dnd_audio.config import SessionConfig, load_session_config
from dnd_audio.errors import DiscoveryError, RecoveryError
from dnd_audio.fixtures import (
    FixtureChunk,
    FixtureSession,
    FixtureTrack,
    FixtureTruth,
    build_session,
)
from dnd_audio.fixtures.wav import write_wav
from dnd_audio.inspection.discovery import Discovery, discover


def small_session(
    tmp_path: Path,
    *,
    chunks: tuple[FixtureChunk, ...],
    tracks: int = 1,
    active_tracks: object = "auto",
    allow_processed_audio: bool = False,
    overrides: dict[str, dict[str, object]] | None = None,
) -> tuple[Path, SessionConfig]:
    """One or two short tracks, with whichever chunk shapes a test needs."""
    letters = ["a", "b"][:tracks]
    spec = FixtureSession(
        session_id="2026-08-15",
        title="Session 01",
        tracks=tuple(
            FixtureTrack(
                track_id=f"tx-{letter}",
                speaker_id={"a": "alice", "b": "bob"}[letter],
                speaker_name={"a": "Alice", "b": "Bob"}[letter],
                receiver_id="rx-a",
                receiver_channel=index + 1,
                tx_label=f"TX0{index + 1}",
                chunks=chunks if index == 0 else (FixtureChunk(0, 4800, sequence=1),),
            )
            for index, letter in enumerate(letters)
        ),
        active_tracks=active_tracks,  # type: ignore[arg-type]
        allow_processed_audio=allow_processed_audio,
        source_time_overrides=overrides or {},
    )
    build_session(spec, tmp_path)
    return tmp_path, load_session_config(tmp_path / "session.yaml")


def roles(discovery: Discovery) -> dict[str, str]:
    return {source.file.relative_path: source.role for source in discovery.all_sources()}


def codes(discovery: Discovery) -> dict[str, str]:
    return {source.file.relative_path: source.reason_code for source in discovery.all_sources()}


class TestSelection:
    def test_originals_are_selected(self, canonical_fixture: FixtureTruth) -> None:
        config = load_session_config(canonical_fixture.session_dir / "session.yaml")
        discovery = discover(canonical_fixture.session_dir, config)

        assert len(discovery.all_sources()) == len(canonical_fixture.chunks)
        assert set(roles(discovery).values()) == {"selected"}
        assert set(codes(discovery).values()) == {"original_selected"}

    def test_an_edit_is_associated_and_ignored(self, tmp_path: Path) -> None:
        """The spec: associate them in the manifest and ignore the processed one."""
        session, config = small_session(
            tmp_path,
            chunks=(
                FixtureChunk(0, 4800, sequence=1),
                FixtureChunk(0, 4800, sequence=1, variant="edit"),
            ),
        )
        discovery = discover(session, config)

        assert roles(discovery) == {
            "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav": "selected",
            "raw/tx-a/TX01_MIC001_20260815_190000_edit.wav": "associated_edit",
        }
        edit = next(s for s in discovery.all_sources() if s.role == "associated_edit")
        assert edit.associated_with == "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav"
        assert edit.reason_code == "processed_variant_ignored"
        assert any(d.code == "orig_selected" for d in discovery.decisions)

    def test_processed_only_is_fatal_by_default(self, tmp_path: Path) -> None:
        """Consuming the edit loses the 32-bit float master, so it takes a deliberate
        act rather than a silent fallback."""
        session, config = small_session(
            tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1, variant="edit"),)
        )
        with pytest.raises(DiscoveryError, match="only processed audio") as raised:
            discover(session, config)
        assert raised.value.code == "processed_audio_only"
        assert "allow_processed_audio" in str(raised.value), "the message must name the fix"

    def test_processed_only_is_permitted_when_recovery_says_so(self, tmp_path: Path) -> None:
        session, config = small_session(
            tmp_path,
            chunks=(FixtureChunk(0, 4800, sequence=1, variant="edit"),),
            allow_processed_audio=True,
        )
        discovery = discover(session, config)

        assert set(roles(discovery).values()) == {"selected"}
        assert codes(discovery)["raw/tx-a/TX01_MIC001_20260815_190000_edit.wav"] == (
            "processed_audio_selected"
        )
        assert any(d.code == "processed_audio_allowed" for d in discovery.decisions)

    def test_duplicates_are_detected_by_content_not_by_name(self, tmp_path: Path) -> None:
        session, config = small_session(tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1),))
        original = session / "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav"
        copy = session / "raw/tx-a/TX01_MIC009_20260815_195959_orig.wav"
        shutil.copy(original, copy)

        discovery = discover(session, config)
        assert roles(discovery)[original.relative_to(session).as_posix()] == "selected"
        assert roles(discovery)[copy.relative_to(session).as_posix()] == "duplicate"
        duplicate = next(s for s in discovery.all_sources() if s.role == "duplicate")
        assert duplicate.associated_with == original.relative_to(session).as_posix()
        assert any(w.code == "duplicate_source" for w in discovery.warnings)

    def test_which_duplicate_survives_does_not_depend_on_directory_order(
        self, tmp_path: Path
    ) -> None:
        """INV-02. The kept copy is the lexicographically first path, always."""
        session, config = small_session(tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1),))
        original = session / "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav"
        shutil.copy(original, session / "raw/tx-a/AAA_first_orig.wav")

        discovery = discover(session, config)
        kept = [s.file.relative_path for s in discovery.all_sources() if s.role == "selected"]
        assert kept == ["raw/tx-a/AAA_first_orig.wav"]

    def test_a_processed_copy_never_displaces_its_original(self, tmp_path: Path) -> None:
        """A regression, and a nasty one to diagnose from its symptom.

        `_edit` sorts before `_orig`. When duplicate resolution ranked copies by path
        alone, a byte-identical edit won its group, the original became the duplicate,
        and the track then had "only processed audio" — a fatal error raised two rules
        downstream with nothing pointing back at the ordering that caused it.
        """
        session, config = small_session(tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1),))
        original = session / "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav"
        edit = session / "raw/tx-a/TX01_MIC001_20260815_190000_edit.wav"
        shutil.copy(original, edit)
        assert edit.name < original.name, "the fixture only bites if the edit sorts first"

        discovery = discover(session, config)
        assert roles(discovery)[original.relative_to(session).as_posix()] == "selected"
        assert roles(discovery)[edit.relative_to(session).as_posix()] == "duplicate"

    def test_a_duplicate_across_two_tracks_warns_distinctly(self, tmp_path: Path) -> None:
        """Identical bytes in two track directories means one of them is attributed to
        the wrong person — a different problem from a duplicated file in one place."""
        session, config = small_session(
            tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1),), tracks=2
        )
        shutil.copy(
            session / "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav",
            session / "raw/tx-b/TX02_MIC001_20260815_190000_orig.wav",
        )

        discovery = discover(session, config)
        across = [w for w in discovery.warnings if w.code == "duplicate_across_tracks"]
        assert len(across) == 1
        assert "wrong person" in across[0].message

    def test_an_unrecognized_filename_is_still_inspected(self, tmp_path: Path) -> None:
        """OQ-003 is open. A grammar used as an inclusion filter would make a firmware
        rename look like an empty session."""
        session, config = small_session(
            tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1, filename="mystery-take.wav"),)
        )
        discovery = discover(session, config)

        assert roles(discovery) == {"raw/tx-a/mystery-take.wav": "selected"}
        assert any(w.code == "unrecognized_filename" for w in discovery.warnings)
        assert any(w.code == "variant_not_determined" for w in discovery.warnings)

    def test_a_file_labelled_for_another_transmitter_warns(self, tmp_path: Path) -> None:
        """The spec's "files belonging to more than one apparent transmitter" warning.

        A warning and not an error: OQ-002 says the label is not unique across kits, so
        it is a signal rather than evidence.
        """
        session, config = small_session(
            tmp_path,
            chunks=(
                FixtureChunk(0, 4800, sequence=1),
                FixtureChunk(
                    4800, 4800, sequence=2, filename="TX05_MIC002_20260815_190001_orig.wav"
                ),
            ),
        )
        discovery = discover(session, config)

        mixed = [w for w in discovery.warnings if w.code == "mixed_transmitter_labels"]
        assert len(mixed) == 1
        assert "TX01, TX05" in mixed[0].message
        assert set(roles(discovery).values()) == {"selected"}, "both files are still Alice's"

    def test_a_sequence_discontinuity_warns(self, tmp_path: Path) -> None:
        session, config = small_session(
            tmp_path,
            chunks=(
                FixtureChunk(0, 4800, sequence=1),
                FixtureChunk(4800, 4800, sequence=7),
            ),
        )
        discovery = discover(session, config)
        gaps = [w for w in discovery.warnings if w.code == "sequence_discontinuity"]
        assert len(gaps) == 1
        assert "1 to 7" in gaps[0].message

    def test_a_non_audio_file_is_reported_and_skipped(self, tmp_path: Path) -> None:
        session, config = small_session(tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1),))
        (session / "raw/tx-a/notes.txt").write_text("field log", encoding="utf-8")

        discovery = discover(session, config)
        assert "raw/tx-a/notes.txt" not in roles(discovery)
        assert any(w.code == "unexpected_file_type" for w in discovery.warnings)


class TestRoster:
    def test_auto_activates_only_directories_with_a_usable_original(self, tmp_path: Path) -> None:
        session, config = small_session(
            tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1),), tracks=2
        )
        for path in (session / "raw/tx-b").iterdir():
            path.unlink()

        discovery = discover(session, config)
        assert discovery.active_track_ids == ("tx-a",)
        assert any(w.code == "track_inactive" for w in discovery.warnings)

    def test_an_inactive_track_is_reported_rather_than_dropped(self, tmp_path: Path) -> None:
        """The spec: reported as inactive with a warning, not silently omitted."""
        session, config = small_session(
            tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1),), tracks=2
        )
        for path in (session / "raw/tx-b").iterdir():
            path.unlink()

        discovery = discover(session, config)
        absent = next(t for t in discovery.tracks if t.track_id == "tx-b")
        assert not absent.active
        assert absent.inactive_reason == "no usable original recording was found"
        assert absent.speaker_name == "Bob"

    def test_an_explicit_list_makes_a_missing_track_fatal(self, tmp_path: Path) -> None:
        """The only way to tell an intentional absence from a capture failure."""
        session, config = small_session(
            tmp_path,
            chunks=(FixtureChunk(0, 4800, sequence=1),),
            tracks=2,
            active_tracks=("tx-a", "tx-b"),
        )
        for path in (session / "raw/tx-b").iterdir():
            path.unlink()

        with pytest.raises(DiscoveryError, match="tx-b") as raised:
            discover(session, config)
        assert raised.value.code == "required_track_missing"
        assert "capture failure" in str(raised.value)

    def test_a_track_left_off_an_explicit_list_is_inactive_not_fatal(self, tmp_path: Path) -> None:
        session, config = small_session(
            tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1),), tracks=2, active_tracks=("tx-a",)
        )
        discovery = discover(session, config)

        assert discovery.active_track_ids == ("tx-a",)
        excluded = next(t for t in discovery.tracks if t.track_id == "tx-b")
        assert excluded.inactive_reason == "not named in the explicit active_tracks list"

    def test_a_missing_directory_is_listed(self, tmp_path: Path) -> None:
        session, config = small_session(
            tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1),), tracks=2
        )
        shutil.rmtree(session / "raw/tx-b")

        discovery = discover(session, config)
        assert discovery.missing_directories == ("raw/tx-b",)
        assert discovery.empty_directories == ()

    def test_an_empty_directory_is_distinguished_from_a_missing_one(self, tmp_path: Path) -> None:
        """They mean different things: one recorder was never plugged in, the other
        recorded nothing."""
        session, config = small_session(
            tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1),), tracks=2
        )
        for path in (session / "raw/tx-b").iterdir():
            path.unlink()

        discovery = discover(session, config)
        assert discovery.empty_directories == ("raw/tx-b",)
        assert discovery.missing_directories == ()

    def test_the_derivation_is_recorded_as_a_decision(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        config = load_session_config(canonical_fixture.session_dir / "session.yaml")
        discovery = discover(canonical_fixture.session_dir, config)
        derived = next(d for d in discovery.decisions if d.code == "active_tracks_derived")
        assert "tx-a, tx-b, tx-c, tx-d, tx-e, tx-f" in derived.detail


class TestUnconfiguredDirectories:
    def test_an_unconfigured_directory_is_never_attributed_to_a_speaker(
        self, tmp_path: Path
    ) -> None:
        """INV-11's sharpest edge. The files are captured — the gate requires per-file
        capture of everything — but they belong to nobody, and there is no field on an
        unassigned source that could hold a speaker."""
        session, config = small_session(tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1),))
        stranger = session / "raw/tx-z"
        stranger.mkdir()
        write_wav(
            stranger / "TX01_MIC001_20260815_190000_orig.wav",
            np.zeros(4800, dtype=np.float32),
            sample_rate=48000,
        )

        discovery = discover(session, config)
        assert discovery.extra_directories == ("raw/tx-z",)
        assert [s.file.relative_path for s in discovery.unassigned] == [
            "raw/tx-z/TX01_MIC001_20260815_190000_orig.wav"
        ]
        assert discovery.unassigned[0].role == "unassigned"
        assert discovery.unassigned[0].reason_code == "directory_not_configured"

        attributed = {s.file.relative_path for t in discovery.tracks for s in t.sources}
        assert "raw/tx-z/TX01_MIC001_20260815_190000_orig.wav" not in attributed

    def test_a_matching_tx_label_does_not_pull_a_file_into_a_track(self, tmp_path: Path) -> None:
        """The exact failure INV-11 exists to prevent: the stray file is labelled TX01,
        the same as Alice's, and it still is not hers."""
        session, config = small_session(tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1),))
        stranger = session / "raw/tx-q"
        stranger.mkdir()
        write_wav(
            stranger / "TX01_MIC002_20260815_190500_orig.wav",
            np.zeros(4800, dtype=np.float32),
            sample_rate=48000,
        )

        discovery = discover(session, config)
        alice = next(t for t in discovery.tracks if t.track_id == "tx-a")
        assert len(alice.sources) == 1
        assert len(discovery.unassigned) == 1

    def test_audio_loose_in_the_raw_root_is_captured_but_unassigned(self, tmp_path: Path) -> None:
        session, config = small_session(tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1),))
        write_wav(session / "raw/stray.wav", np.zeros(4800, dtype=np.float32), sample_rate=48000)

        discovery = discover(session, config)
        assert [s.file.relative_path for s in discovery.unassigned] == ["raw/stray.wav"]
        assert any(w.code == "stray_audio_file" for w in discovery.warnings)


class TestOverridesMustMatchSomething:
    def test_an_override_matching_no_source_is_fatal(self, tmp_path: Path) -> None:
        """ADR-0007. A silently ignored override is the failure the mechanism exists to
        prevent, and a mistyped path is that failure with a typo in front of it."""
        session, config = small_session(
            tmp_path,
            chunks=(FixtureChunk(0, 4800, sequence=1),),
            overrides={
                "raw/tx-a/TX01_MIC999_20260815_190000_orig.wav": {
                    "start_timecode": "19:00:00:00",
                    "reason": "from the field log",
                }
            },
        )
        with pytest.raises(RecoveryError, match="matches") as raised:
            discover(session, config)
        assert raised.value.code == "recovery_override_unmatched"

    def test_the_message_lists_what_was_actually_there(self, tmp_path: Path) -> None:
        """So the fix is reading the error rather than going to look."""
        session, config = small_session(
            tmp_path,
            chunks=(FixtureChunk(0, 4800, sequence=1),),
            overrides={"raw/tx-a/typo.wav": {"start_offset_samples": 0, "reason": "measured"}},
        )
        with pytest.raises(RecoveryError) as raised:
            discover(session, config)
        assert "TX01_MIC001_20260815_190000_orig.wav" in str(raised.value)

    def test_an_override_naming_a_real_source_is_accepted(self, tmp_path: Path) -> None:
        session, config = small_session(
            tmp_path,
            chunks=(FixtureChunk(0, 4800, sequence=1),),
            overrides={
                "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav": {
                    "start_offset_samples": -1200,
                    "reason": "clap-measured",
                }
            },
        )
        assert discover(session, config).active_track_ids == ("tx-a",)

    def test_an_override_may_name_an_unassigned_file(self, tmp_path: Path) -> None:
        """Unassigned sources are still discovered sources, so an override aimed at one
        is not a typo — even though nothing will consume it."""
        session, _ = small_session(tmp_path, chunks=(FixtureChunk(0, 4800, sequence=1),))
        stranger = session / "raw/tx-z"
        stranger.mkdir()
        write_wav(stranger / "loose.wav", np.zeros(4800, dtype=np.float32), sample_rate=48000)

        document = (session / "session.yaml").read_text(encoding="utf-8")
        document = document.replace(
            "  source_time_overrides: {}",
            "  source_time_overrides:\n"
            '    "raw/tx-z/loose.wav":\n'
            "      start_offset_samples: 0\n"
            '      reason: "measured"',
        )
        (session / "session.yaml").write_text(document, encoding="utf-8")

        discover(session, load_session_config(session / "session.yaml"))


class TestDeterminism:
    def test_two_discoveries_of_the_same_session_agree_exactly(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        config = load_session_config(canonical_fixture.session_dir / "session.yaml")
        assert discover(canonical_fixture.session_dir, config) == discover(
            canonical_fixture.session_dir, config
        )

    def test_warnings_and_decisions_are_sorted(self, tmp_path: Path) -> None:
        """Directory iteration order must not reach the report (INV-02)."""
        session, config = small_session(
            tmp_path,
            chunks=(
                FixtureChunk(0, 4800, sequence=1),
                FixtureChunk(
                    4800, 4800, sequence=9, filename="TX05_MIC009_20260815_190001_orig.wav"
                ),
            ),
        )
        discovery = discover(session, config)
        codes_seen = [(w.code, w.path or "", w.message) for w in discovery.warnings]
        assert codes_seen == sorted(codes_seen)
        subjects = [(d.code, d.subject) for d in discovery.decisions]
        assert subjects == sorted(subjects)

"""A whole `inspect` run: what it writes, what it refuses, and what it never touches.

The determinism tests here are behavioural rather than structural, which is deliberate.
Asserting that no manifest field is time-typed does not establish INV-03 — a hostname
serializes as a plain string and a cache count as an integer, and both would sail
through. So the clock, the cache state, and the filesystem timestamps are each varied
directly, and the assertion is on the bytes.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from dnd_audio.artifacts.manifest import (
    BwfSampleReferenceRecord,
    SessionOffsetRecord,
    TimecodeRecord,
)
from dnd_audio.determinism import sha256_file
from dnd_audio.errors import ExitCode
from dnd_audio.fixtures import (
    FixtureChunk,
    FixtureSession,
    FixtureTrack,
    FixtureTruth,
    build_session,
)
from dnd_audio.fixtures.wav import BroadcastMetadata, write_wav
from dnd_audio.inspection.probe import ProbeResult
from dnd_audio.inspection.riff import read_inventory
from dnd_audio.inspection.runner import PROBE_DIRNAME, run_inspect

EARLY = dt.datetime(2026, 8, 15, 19, 0, 0, tzinfo=dt.UTC)
LATE = dt.datetime(2031, 1, 2, 3, 4, 5, tzinfo=dt.UTC)


def manifest_bytes(session_dir: Path) -> bytes:
    return (session_dir / "work" / "manifest.json").read_bytes()


class TestWhatItProduces:
    def test_the_run_succeeds_and_writes_both_artifacts(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        result = run_inspect(canonical_fixture.session_dir, now=EARLY)

        assert result.exit_code is ExitCode.OK
        assert result.manifest_path.is_file()
        assert result.report_path.is_file()
        assert result.report.overall_status.value == "complete"

    def test_every_selected_source_is_captured(self, canonical_fixture: FixtureTruth) -> None:
        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        sources = {
            source.relative_path: source
            for track in result.manifest.tracks
            for source in track.sources
        }
        assert set(sources) == {chunk.relative_path for chunk in canonical_fixture.chunks}

        for chunk in canonical_fixture.chunks:
            source = sources[chunk.relative_path]
            assert source.sha256 == chunk.sha256
            assert source.size_bytes == chunk.size_bytes
            assert source.container is not None
            assert source.container.sample_rate == 48000
            assert source.container.sample_count == chunk.n_samples
            assert source.container.sample_count_agrees is True

    def test_the_start_time_matches_the_fixtures_truth(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The truth record states the time reference before any file exists; this is
        the round trip through a real container and a real FFprobe."""
        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        for track in result.manifest.tracks:
            for source in track.sources:
                chunk = next(
                    c for c in canonical_fixture.chunks if c.relative_path == source.relative_path
                )
                assert source.start_time is not None
                evidence = source.start_time.evidence
                if isinstance(evidence, BwfSampleReferenceRecord):
                    assert evidence.samples == chunk.time_reference
                else:
                    # tx-f carries a timecode tag; 30 fps at 48 kHz is 1600 samples a
                    # frame, exactly.
                    assert isinstance(evidence, TimecodeRecord)
                    assert evidence.frames * 1600 == chunk.time_reference

    def test_the_riff_inventory_holds_the_chunk_ffprobe_never_reported(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        source = result.manifest.tracks[0].sources[0]
        assert source.riff is not None
        found = {chunk.chunk_id for chunk in source.riff.chunks}
        assert {"fmt ", "bext", "XPRV", "iXML", "data"} <= found

        sidecar = json.loads(
            (canonical_fixture.session_dir / source.probe.sidecar_path).read_bytes()  # type: ignore[union-attr]
        )
        assert "XPRV" not in json.dumps(sidecar)

    def test_the_ffprobe_sidecar_is_verbatim_and_content_addressed(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        for track in result.manifest.tracks:
            for source in track.sources:
                assert source.probe is not None
                path = canonical_fixture.session_dir / source.probe.sidecar_path
                assert path.is_file()
                assert sha256_file(path) == source.probe.sha256
                assert path.name == f"{source.probe.sha256}.json"
                assert source.probe.command[0] == "ffprobe"

    def test_the_manifest_validates_against_the_checked_in_schema(
        self, canonical_fixture: FixtureTruth, repo_root: Path
    ) -> None:
        """Against the committed artifact, not against the model that produced it."""
        from jsonschema import Draft202012Validator

        run_inspect(canonical_fixture.session_dir, now=EARLY)
        schema = json.loads(
            (repo_root / "schemas" / "manifest.schema.json").read_text(encoding="utf-8")
        )
        Draft202012Validator(schema).validate(
            json.loads(manifest_bytes(canonical_fixture.session_dir))
        )

    def test_the_sidecar_directory_holds_one_file_per_source(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        run_inspect(canonical_fixture.session_dir, now=EARLY)
        sidecars = list((canonical_fixture.session_dir / PROBE_DIRNAME).iterdir())
        assert len(sidecars) == len(canonical_fixture.chunks)


class TestUnexpectedFormats:
    """The spec's "warn about unexpected formats" selection rule.

    A warning here and a fatal error in M2, deliberately. The spec lists a non-48 kHz
    selected source among the fatal errors, but rejecting it is timeline construction's
    job; refusing to *record* a file this milestone can perfectly well describe would
    throw away the diagnostic that explains the later failure.
    """

    def _one_track(self, tmp_path: Path, chunk: FixtureChunk) -> Path:
        build_session(
            FixtureSession(
                session_id="2026-08-15",
                title="Session 01",
                tracks=(
                    FixtureTrack(
                        track_id="tx-a",
                        speaker_id="alice",
                        speaker_name="Alice",
                        receiver_id="rx-a",
                        receiver_channel=1,
                        tx_label="TX01",
                        chunks=(chunk,),
                    ),
                ),
            ),
            tmp_path,
        )
        return tmp_path

    def test_a_44_1_khz_source_warns_but_is_still_recorded(self, tmp_path: Path) -> None:
        session = self._one_track(tmp_path, FixtureChunk(0, 4410, sequence=1, sample_rate=44100))
        result = run_inspect(session, now=EARLY)

        assert result.exit_code is ExitCode.OK, "M1 records it; M2 is where it is fatal"
        source = result.manifest.tracks[0].sources[0]
        assert source.container is not None
        assert source.container.sample_rate == 44100
        codes = {note.code for note in source.warnings}
        assert "unexpected_sample_rate" in codes
        assert "M2" in next(
            n.message for n in source.warnings if n.code == "unexpected_sample_rate"
        )

    def test_the_reference_of_a_44_1_khz_source_is_read_at_its_own_rate(
        self, tmp_path: Path
    ) -> None:
        """Reading it as 48000ths of a second would misplace the file by 8.75%."""
        session = self._one_track(tmp_path, FixtureChunk(0, 4410, sequence=1, sample_rate=44100))
        result = run_inspect(session, now=EARLY)
        evidence = result.manifest.tracks[0].sources[0].start_time.evidence  # type: ignore[union-attr]
        assert isinstance(evidence, BwfSampleReferenceRecord)
        assert evidence.sample_rate == 44100

    def test_a_conforming_source_warns_about_nothing(self, canonical_fixture: FixtureTruth) -> None:
        """A warning that fires on healthy input is noise, and noise is ignored."""
        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        for track in result.manifest.tracks:
            for source in track.sources:
                assert source.warnings == [], f"{source.relative_path}: {source.warnings}"


class TestByteStability:
    def test_two_runs_are_byte_identical(self, canonical_fixture: FixtureTruth) -> None:
        run_inspect(canonical_fixture.session_dir, now=EARLY)
        first = manifest_bytes(canonical_fixture.session_dir)
        run_inspect(canonical_fixture.session_dir, now=LATE)
        assert manifest_bytes(canonical_fixture.session_dir) == first

    def test_injected_clock_and_cache_state_change_no_byte(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The INV-03 proof.

        The second run has a different wall clock *and* a full cache, so if either
        reached the manifest these bytes would differ. A structural "no time-typed
        field" check would not catch a hostname or a counter; this does.
        """
        run_inspect(canonical_fixture.session_dir, now=EARLY, use_cache=False)
        cold = manifest_bytes(canonical_fixture.session_dir)

        warm_result = run_inspect(canonical_fixture.session_dir, now=LATE)
        assert manifest_bytes(canonical_fixture.session_dir) == cold
        assert warm_result.report.telemetry.cache_hits > 0, "the second run must be warm"

    def test_a_relocated_session_produces_the_same_manifest(
        self, canonical_fixture: FixtureTruth, tmp_path: Path
    ) -> None:
        """FFprobe echoes the filename it was given into its own output, so probing by
        absolute path would make a copied session produce a different manifest with
        nothing to explain why."""
        run_inspect(canonical_fixture.session_dir, now=EARLY)
        here = manifest_bytes(canonical_fixture.session_dir)

        elsewhere = tmp_path / "somewhere" / "much" / "deeper"
        shutil.copytree(canonical_fixture.session_dir, elsewhere)
        shutil.rmtree(elsewhere / "work")
        run_inspect(elsewhere, now=LATE)

        assert manifest_bytes(elsewhere) == here

    def test_touching_every_source_changes_no_byte(self, canonical_fixture: FixtureTruth) -> None:
        """INV-12, behaviourally. Modification time is not an input, so changing every
        one of them must move nothing — including the timing decisions."""
        run_inspect(canonical_fixture.session_dir, now=EARLY, use_cache=False)
        before = manifest_bytes(canonical_fixture.session_dir)

        for chunk in canonical_fixture.chunks:
            os.utime(canonical_fixture.session_dir / chunk.relative_path, (0, 0))

        run_inspect(canonical_fixture.session_dir, now=EARLY, use_cache=False)
        assert manifest_bytes(canonical_fixture.session_dir) == before

    def test_the_manifest_has_no_time_typed_field(self, canonical_fixture: FixtureTruth) -> None:
        """A cheap tripwire, not the proof. It catches someone adding `generated_at`;
        it would not catch a hostname, which is what the behavioural tests are for."""
        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        fields = set(type(result.manifest).model_fields)
        assert not {"generated_at", "started_at", "finished_at", "hostname"} & fields
        document = json.dumps(result.manifest.model_dump(mode="json"))
        for telemetry in ("cache_hits", "cache_misses", "stage_seconds", "elapsed"):
            assert telemetry not in document


class TestRawIsUntouched:
    def test_raw_is_byte_identical_after_a_full_run(self, canonical_fixture: FixtureTruth) -> None:
        """INV-01, over *every* file under raw/, not only the ones inspection selected.

        A non-audio file is included on purpose: "we did not touch what we read" is a
        weaker claim than the invariant makes, and an accidental normalization pass
        would most likely hit the files nobody was thinking about.
        """
        (canonical_fixture.session_dir / "raw" / "tx-a" / "field-log.txt").write_text(
            "three claps at the top", encoding="utf-8"
        )
        before = _snapshot(canonical_fixture.session_dir / "raw")

        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        assert result.exit_code is ExitCode.OK
        assert _snapshot(canonical_fixture.session_dir / "raw") == before

    def test_a_source_modified_during_the_run_is_fatal(
        self, canonical_fixture: FixtureTruth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The verification has to be able to fail, or it is decoration.

        A source is corrupted mid-inspection by a patched probe. The run must notice,
        fail, and — because the cache commits only after the check — leave no cache
        entry describing bytes that are gone.
        """
        real = run_inspect.__globals__["run_ffprobe"]
        victim = canonical_fixture.for_track("tx-b")[0].relative_path

        def meddle(session_dir: Path, relative_path: str) -> ProbeResult:
            result: ProbeResult = real(session_dir, relative_path)
            if relative_path == victim:
                target = session_dir / victim
                target.write_bytes(target.read_bytes() + b"tampered")
            return result

        monkeypatch.setattr("dnd_audio.inspection.runner.run_ffprobe", meddle)
        result = run_inspect(canonical_fixture.session_dir, now=EARLY)

        assert result.exit_code is ExitCode.FATAL
        errors = [error for stage in result.report.stages for error in stage.errors]
        assert [error.code for error in errors] == ["raw_sources_modified"]
        assert victim in errors[0].message
        assert not list((canonical_fixture.session_dir / "work/cache/inspect").glob("*.json"))

    def test_output_paths_inside_raw_are_fatal(self, tmp_path: Path) -> None:
        """The spec lists it among the fatal errors, and it must be checked before the
        first write rather than diagnosed after it."""
        spec = FixtureSession(
            session_id="2026-08-15",
            title="Session 01",
            tracks=(
                FixtureTrack(
                    track_id="tx-a",
                    speaker_id="alice",
                    speaker_name="Alice",
                    receiver_id="rx-a",
                    receiver_channel=1,
                    tx_label="TX01",
                    chunks=(FixtureChunk(0, 4800, sequence=1),),
                ),
            ),
        )
        build_session(spec, tmp_path)

        # Move the track under `work/`, so the manifest would be written inside the
        # directory the sources live in.
        (tmp_path / "work").mkdir(exist_ok=True)
        shutil.move(str(tmp_path / "raw" / "tx-a"), str(tmp_path / "work" / "tx-a"))
        document = (tmp_path / "session.yaml").read_text(encoding="utf-8")
        (tmp_path / "session.yaml").write_text(
            document.replace("input: raw/tx-a", "input: work/tx-a"), encoding="utf-8"
        )

        result = run_inspect(tmp_path, now=EARLY)
        assert result.exit_code is ExitCode.FATAL
        errors = [error for stage in result.report.stages for error in stage.errors]
        assert errors[0].code == "output_inside_raw"


class TestUnassignedSources:
    def test_a_file_in_an_unconfigured_directory_reaches_the_manifest_unattributed(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        stranger = canonical_fixture.session_dir / "raw" / "tx-z"
        stranger.mkdir()
        # Distinct bytes, so this exercises unassigned-ness rather than duplicate
        # detection — the stray-copy case has its own test below.
        write_wav(
            stranger / "TX01_MIC001_20260815_190000_orig.wav",
            np.full(4800, 0.25, dtype=np.float32),
            sample_rate=48000,
            broadcast=BroadcastMetadata(time_reference=3283200000),
        )

        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        assert result.exit_code is ExitCode.OK
        assert [source.relative_path for source in result.manifest.unassigned] == [
            "raw/tx-z/TX01_MIC001_20260815_190000_orig.wav"
        ]
        assert result.manifest.roster.extra_directories == ["raw/tx-z"]

        attributed = {s.relative_path for t in result.manifest.tracks for s in t.sources}
        assert "raw/tx-z/TX01_MIC001_20260815_190000_orig.wav" not in attributed

    def test_an_unassigned_file_is_captured_as_fully_as_a_selected_one(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The spec: "for every candidate audio file, run ffprobe and retain…".

        An earlier version skipped probing anything that no stage would consume, which
        left ignored edits, duplicates, and strays with no container facts, no sidecar,
        and no RIFF inventory — and the operator asking "why was this ignored" is asking
        about exactly those files.
        """
        stranger = canonical_fixture.session_dir / "raw" / "tx-z"
        stranger.mkdir()
        write_wav(
            stranger / "loose.wav",
            np.full(4800, 0.25, dtype=np.float32),
            sample_rate=48000,
            broadcast=BroadcastMetadata(time_reference=3283200000),
        )

        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        loose = next(s for s in result.manifest.unassigned if s.relative_path.endswith("loose.wav"))

        assert loose.container is not None
        assert loose.container.sample_rate == 48000
        assert loose.container.sample_count == 4800
        assert loose.probe is not None
        assert (canonical_fixture.session_dir / loose.probe.sidecar_path).is_file()
        assert loose.riff is not None
        assert {c.chunk_id for c in loose.riff.chunks} >= {"fmt ", "data"}
        assert loose.start_time is not None

    def test_an_ignored_edit_is_captured_too(self, tmp_path: Path) -> None:
        build_session(
            FixtureSession(
                session_id="2026-08-15",
                title="Session 01",
                tracks=(
                    FixtureTrack(
                        track_id="tx-a",
                        speaker_id="alice",
                        speaker_name="Alice",
                        receiver_id="rx-a",
                        receiver_channel=1,
                        tx_label="TX01",
                        chunks=(
                            FixtureChunk(0, 4800, sequence=1),
                            FixtureChunk(0, 4800, sequence=1, variant="edit"),
                        ),
                    ),
                ),
            ),
            tmp_path,
        )
        result = run_inspect(tmp_path, now=EARLY)
        edit = next(
            s for t in result.manifest.tracks for s in t.sources if s.role == "associated_edit"
        )
        assert edit.container is not None, "an ignored edit is still a candidate"
        assert edit.probe is not None
        assert edit.riff is not None

    def test_a_stray_without_timing_is_recorded_rather_than_fatal(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """INV-12 is about the sources the pipeline uses.

        A WAV in a directory nobody configured has no obligation to carry a timecode,
        and letting one fail the whole session would be an own goal.
        """
        stranger = canonical_fixture.session_dir / "raw" / "tx-z"
        stranger.mkdir()
        write_wav(stranger / "no-timing.wav", np.zeros(4800, dtype=np.float32), sample_rate=48000)

        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        assert result.exit_code is ExitCode.OK
        loose = next(s for s in result.manifest.unassigned)
        assert loose.start_time is None
        assert "no_timing_evidence" in {note.code for note in loose.warnings}

    def test_a_stray_copy_never_displaces_a_configured_tracks_recording(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """A regression, and the reason duplicate ranking knows about attribution.

        Duplicate resolution ranked copies by path alone. A stray at `raw/aaa-stray.wav`
        sorts before `raw/tx-a/TX01_…`, so it won — marking Alice's real original a
        duplicate of a file attributed to nobody, and silently costing the track its
        source. INV-11 violated from the direction nothing else guards.
        """
        target = canonical_fixture.for_track("tx-a")[0]
        stray = canonical_fixture.session_dir / "raw" / "aaa-stray.wav"
        shutil.copy(canonical_fixture.session_dir / target.relative_path, stray)
        assert target.relative_path > "raw/aaa-stray.wav", "the stray must sort first to bite"

        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        assert result.exit_code is ExitCode.OK

        alice = next(t for t in result.manifest.tracks if t.track_id == "tx-a")
        kept = next(s for s in alice.sources if s.relative_path == target.relative_path)
        assert kept.role == "selected", "the track's own recording must win"
        assert alice.active

        loser = next(s for s in result.manifest.unassigned if "aaa-stray" in s.relative_path)
        assert loser.role == "duplicate"
        assert loser.associated_with == target.relative_path


class TestOverridesReachBothArtifacts:
    def test_an_override_appears_in_both_manifest_and_report(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The spec requires overrides to be recorded prominently."""
        target = canonical_fixture.for_track("tx-a")[0]
        document = (canonical_fixture.session_dir / "session.yaml").read_text(encoding="utf-8")
        document = document.replace(
            "  source_time_overrides: {}",
            f"  source_time_overrides:\n"
            f'    "{target.relative_path}":\n'
            f"      sha256: {target.sha256}\n"
            f"      start_offset_samples: -1200\n"
            f'      reason: "clap-measured; the bext reference was damaged"',
        )
        (canonical_fixture.session_dir / "session.yaml").write_text(document, encoding="utf-8")

        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        assert result.exit_code is ExitCode.OK

        source = next(
            s
            for track in result.manifest.tracks
            for s in track.sources
            if s.relative_path == target.relative_path
        )
        assert source.start_time is not None
        assert source.start_time.strategy == "recovery_override_offset"
        assert isinstance(source.start_time.evidence, SessionOffsetRecord)
        assert source.start_time.evidence.samples == -1200
        assert source.start_time.override_reason is not None
        assert "clap-measured" in source.start_time.override_reason

        # And the report — the half this test's name always claimed and never checked.
        applied = [d for d in result.report.decisions if d.code == "recovery_override_applied"]
        assert [d.subject for d in applied] == [target.relative_path]
        assert "clap-measured" in applied[0].detail
        assert applied[0].details["strategy"] == "recovery_override_offset"
        assert applied[0].details["evidence"] == "session_offset_samples"

    def test_the_override_wins_over_a_perfectly_good_bext_reference(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """An override exists for when the file's own metadata is wrong. If the file
        won, the escape hatch would be unreachable on exactly the files that have one."""
        target = canonical_fixture.for_track("tx-a")[0]
        inventory = read_inventory(canonical_fixture.session_dir / target.relative_path)
        assert inventory.find("bext") is not None, "the fixture must have good metadata"

        document = (canonical_fixture.session_dir / "session.yaml").read_text(encoding="utf-8")
        (canonical_fixture.session_dir / "session.yaml").write_text(
            document.replace(
                "  source_time_overrides: {}",
                f"  source_time_overrides:\n"
                f'    "{target.relative_path}":\n'
                f'      start_timecode: "20:00:00:00"\n'
                f'      reason: "the receiver was jammed an hour late"',
            ),
            encoding="utf-8",
        )

        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        source = next(
            s
            for track in result.manifest.tracks
            for s in track.sources
            if s.relative_path == target.relative_path
        )
        assert source.start_time is not None
        assert source.start_time.strategy == "recovery_override_timecode"


def _snapshot(directory: Path) -> dict[str, tuple[str, int]]:
    return {
        str(path.relative_to(directory)): (sha256_file(path), path.stat().st_size)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class TestInvOneCannotBeBypassed:
    """Regressions for two ways the INV-01 machinery was defeatable.

    Both were found by independent review, and both share a shape: the *check* was
    present and looked right, while the thing it checked was not what INV-01 protects.
    """

    def test_a_symlinked_output_directory_is_refused(self, tmp_path: Path) -> None:
        """`output -> raw/tx-a` used to write the report into a track's source directory.

        The check compared lexical paths, so `output/ingest-report.json` did not *look*
        like it was inside `raw/`. The snapshot could not catch it either: the report is
        written after the snapshot is verified. One symlink, invariant gone.
        """
        session = _one_track_session(tmp_path)
        (session / "output").symlink_to(session / "raw" / "tx-a", target_is_directory=True)

        result = run_inspect(session, now=EARLY)
        assert result.exit_code is ExitCode.FATAL
        codes = [e.code for s in result.report.stages for e in s.errors]
        assert codes == ["output_inside_raw"]
        assert not (session / "raw" / "tx-a" / "ingest-report.json").exists()

    def test_a_symlinked_work_directory_is_refused(self, tmp_path: Path) -> None:
        session = _one_track_session(tmp_path)
        (session / "work").symlink_to(session / "raw" / "tx-a", target_is_directory=True)

        result = run_inspect(session, now=EARLY)
        assert result.exit_code is ExitCode.FATAL
        assert [e.code for s in result.report.stages for e in s.errors] == ["output_inside_raw"]

    def test_a_track_at_the_session_root_is_still_protected(self, tmp_path: Path) -> None:
        """`input: "tx-a"` is valid configuration, and used to disable the check entirely.

        The scan root is then `"."`, which was filtered out — so the snapshot was empty,
        `_verify_unchanged` compared two empty dicts, and INV-01's proof passed no matter
        what happened to the sources. The bug was in the verification, so only a test
        that *modifies a source* can detect it.
        """
        session = _one_track_session(tmp_path, at_root=True)
        real = run_inspect.__globals__["run_ffprobe"]
        source = "tx-a/TX01_MIC001_20260815_190000_orig.wav"
        assert (session / source).is_file()

        def meddle(session_dir: Path, relative_path: str) -> ProbeResult:
            result: ProbeResult = real(session_dir, relative_path)
            target = session_dir / source
            target.write_bytes(target.read_bytes() + b"tampered")
            return result

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("dnd_audio.inspection.runner.run_ffprobe", meddle)
            result = run_inspect(session, now=EARLY)

        assert result.exit_code is ExitCode.FATAL
        assert [e.code for s in result.report.stages for e in s.errors] == ["raw_sources_modified"]

    def test_a_track_at_the_session_root_does_not_ingest_its_own_output(
        self, tmp_path: Path
    ) -> None:
        """With the root at `"."`, `work/` and `output/` are siblings of the track."""
        session = _one_track_session(tmp_path, at_root=True)
        first = run_inspect(session, now=EARLY)
        assert first.exit_code is ExitCode.OK

        second = run_inspect(session, now=LATE)
        assert second.exit_code is ExitCode.OK
        assert second.manifest.roster.extra_directories == []
        assert second.manifest.unassigned == []


class TestEveryFailureStillWritesAReport:
    """INV-13 does not have an exemption for failures nobody anticipated."""

    def test_a_track_holding_only_duplicates_fails_cleanly(self, tmp_path: Path) -> None:
        """Was an uncaught pydantic ValidationError and no report at all.

        Root cause: the inactive reason keyed off whether *any* file was found rather
        than whether any was *selected*, and those differ exactly here.
        """
        session = _two_track_session(tmp_path)
        for path in (session / "raw/tx-b").iterdir():
            path.unlink()
        shutil.copy(
            session / "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav",
            session / "raw/tx-b/TX02_MIC001_20260815_190000_orig.wav",
        )

        result = run_inspect(session, now=EARLY)
        assert result.exit_code is ExitCode.OK, "a duplicate is a warning, not a failure"
        assert (session / "output" / "ingest-report.json").is_file()

        bob = next(t for t in result.manifest.tracks if t.track_id == "tx-b")
        assert not bob.active
        assert bob.inactive_reason is not None
        assert "byte-identical" in bob.inactive_reason

    def test_an_unreadable_source_fails_with_a_report_not_a_traceback(self, tmp_path: Path) -> None:
        session = _one_track_session(tmp_path)
        target = session / "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav"
        target.chmod(0o000)
        try:
            result = run_inspect(session, now=EARLY)
        finally:
            target.chmod(0o644)

        assert result.exit_code is ExitCode.FATAL
        assert (session / "output" / "ingest-report.json").is_file()
        errors = [e for s in result.report.stages for e in s.errors]
        assert errors[0].code == "internal_error"
        for stage in result.report.stages:
            if stage.stage.value != "inspect":
                assert stage.skip_reason

    def test_a_failed_rerun_removes_the_manifest_the_last_success_left(
        self, tmp_path: Path
    ) -> None:
        """The case the previous test could not see, because it started from nothing.

        A stale manifest is worse than none: it looks current, describes a session that
        no longer inspects, and nothing inside it says so.
        """
        session = _one_track_session(tmp_path)
        assert run_inspect(session, now=EARLY).exit_code is ExitCode.OK
        assert (session / "work" / "manifest.json").is_file()

        (session / "session.yaml").write_text("not: a valid session\n", encoding="utf-8")
        second = run_inspect(session, now=LATE)

        assert second.exit_code is ExitCode.FATAL
        assert not (session / "work" / "manifest.json").exists()
        assert (session / "output" / "ingest-report.json").is_file()

    def test_a_run_that_never_read_a_config_reports_no_config_hash(self, tmp_path: Path) -> None:
        """Rather than a string of sixty-four zeroes, which is a valid-looking lie."""
        (tmp_path / "raw").mkdir()
        result = run_inspect(tmp_path, now=EARLY)
        assert result.report.provenance.config_hash is None


class TestCacheReuseKeepsTheSidecar:
    def test_a_cache_hit_whose_sidecar_is_gone_re_probes(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Deleting `work/ffprobe/` while keeping `work/cache/` is an ordinary thing.

        It used to produce a manifest referencing files that do not exist, exit 0, and
        say nothing — so the "raw FFprobe JSON is retained" gate held on a first run and
        quietly stopped holding on every one after.
        """
        run_inspect(canonical_fixture.session_dir, now=EARLY)
        sidecars = canonical_fixture.session_dir / PROBE_DIRNAME
        for path in sidecars.iterdir():
            path.unlink()

        result = run_inspect(canonical_fixture.session_dir, now=LATE)
        assert result.exit_code is ExitCode.OK
        for track in result.manifest.tracks:
            for source in track.sources:
                assert source.probe is not None
                assert (canonical_fixture.session_dir / source.probe.sidecar_path).is_file()


def _one_track_session(tmp_path: Path, *, at_root: bool = False) -> Path:
    build_session(
        FixtureSession(
            session_id="2026-08-15",
            title="Session 01",
            tracks=(
                FixtureTrack(
                    track_id="tx-a",
                    speaker_id="alice",
                    speaker_name="Alice",
                    receiver_id="rx-a",
                    receiver_channel=1,
                    tx_label="TX01",
                    chunks=(FixtureChunk(0, 4800, sequence=1),),
                ),
            ),
        ),
        tmp_path,
    )
    if at_root:
        shutil.move(str(tmp_path / "raw" / "tx-a"), str(tmp_path / "tx-a"))
        (tmp_path / "raw").rmdir()
        document = (tmp_path / "session.yaml").read_text(encoding="utf-8")
        (tmp_path / "session.yaml").write_text(
            document.replace("input: raw/tx-a", "input: tx-a"), encoding="utf-8"
        )
    return tmp_path


def _two_track_session(tmp_path: Path) -> Path:
    build_session(
        FixtureSession(
            session_id="2026-08-15",
            title="Session 01",
            tracks=tuple(
                FixtureTrack(
                    track_id=f"tx-{letter}",
                    speaker_id=name.lower(),
                    speaker_name=name,
                    receiver_id=f"rx-{letter}",
                    receiver_channel=1,
                    tx_label=f"TX0{index + 1}",
                    chunks=(FixtureChunk(0, 4800, sequence=1),),
                )
                for index, (letter, name) in enumerate([("a", "Alice"), ("b", "Bob")])
            ),
        ),
        tmp_path,
    )
    return tmp_path


class TestAnUnusableStrayDoesNotFailTheSession:
    """A regression introduced *by* the fix that made every candidate get probed.

    Capturing everything is right; letting everything fail the run is not. These pin the
    asymmetry: a source the pipeline will use must be inspectable, and one it will not
    must merely be described as best it can be.
    """

    def test_a_corrupt_stray_is_recorded_not_fatal(self, canonical_fixture: FixtureTruth) -> None:
        stranger = canonical_fixture.session_dir / "raw" / "tx-z"
        stranger.mkdir()
        (stranger / "broken.wav").write_bytes(b"this is not a wav at all")

        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        assert result.exit_code is ExitCode.OK, "five good transmitters must still inspect"

        broken = next(s for s in result.manifest.unassigned if "broken" in s.relative_path)
        assert broken.container is None
        assert "capture_failed" in {note.code for note in broken.warnings}

        stage = next(s for s in result.report.stages if s.stage.value == "inspect")
        assert "capture_failed" in {w.code for w in stage.warnings}

    def test_a_corrupt_selected_source_is_still_fatal(self, tmp_path: Path) -> None:
        """The asymmetry has to bite in the other direction too, or it is just leniency."""
        session = _one_track_session(tmp_path)
        target = session / "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav"
        target.write_bytes(b"this is not a wav at all")

        result = run_inspect(session, now=EARLY)
        assert result.exit_code is ExitCode.FATAL
        codes = [e.code for s in result.report.stages for e in s.errors]
        assert len(codes) == 1
        assert codes[0] in {"probe_failed", "unreadable_container"}

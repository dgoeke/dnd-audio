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
        shutil.copy(
            canonical_fixture.session_dir / canonical_fixture.chunks[0].relative_path,
            stranger / "TX01_MIC001_20260815_190000_orig.wav",
        )

        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        assert result.exit_code is ExitCode.OK
        assert [source.relative_path for source in result.manifest.unassigned] == [
            "raw/tx-z/TX01_MIC001_20260815_190000_orig.wav"
        ]
        assert result.manifest.roster.extra_directories == ["raw/tx-z"]

        attributed = {s.relative_path for t in result.manifest.tracks for s in t.sources}
        assert "raw/tx-z/TX01_MIC001_20260815_190000_orig.wav" not in attributed


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

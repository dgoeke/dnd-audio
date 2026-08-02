"""What the report says, including when inspection fails.

INV-13's whole point is the failure case, and arguing that the builder was called does
not establish it. Every case in :class:`TestFailurePaths` drives the real CLI through a
real broken session and asserts on the report that survived it.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from dnd_audio.cli import app
from dnd_audio.errors import ExitCode
from dnd_audio.fixtures import (
    FixtureChunk,
    FixtureSession,
    FixtureTrack,
    FixtureTruth,
    build_session,
)
from dnd_audio.inspection.probe import ProbeError, ProbeResult
from dnd_audio.inspection.runner import run_inspect

runner = CliRunner()
EARLY = dt.datetime(2026, 8, 15, 19, 0, 0, tzinfo=dt.UTC)


def read_report(session_dir: Path) -> dict[str, object]:
    payload = (session_dir / "output" / "ingest-report.json").read_bytes()
    document = json.loads(payload)
    assert isinstance(document, dict)
    return document


def errors_of(document: dict[str, object]) -> list[dict[str, str]]:
    stages = document["stages"]
    assert isinstance(stages, list)
    return [error for stage in stages for error in stage["errors"]]


def override_session(tmp_path: Path, replacements: dict[str, str] | None = None) -> Path:
    """A one-track session whose `session.yaml` can be patched by substring."""
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
    if replacements:
        document = (tmp_path / "session.yaml").read_text(encoding="utf-8")
        for old, new in replacements.items():
            assert old in document, f"{old!r} is not in the generated session.yaml"
            document = document.replace(old, new)
        (tmp_path / "session.yaml").write_text(document, encoding="utf-8")
    return tmp_path


#: The empty-overrides line the generator writes, and what to put in its place.
NO_OVERRIDES = "  source_time_overrides: {}"


def with_override(body: str) -> dict[str, str]:
    return {NO_OVERRIDES: "  source_time_overrides:\n" + body}


class TestASuccessfulReport:
    def test_it_validates_against_the_checked_in_schema(
        self, canonical_fixture: FixtureTruth, repo_root: Path
    ) -> None:
        run_inspect(canonical_fixture.session_dir, now=EARLY)
        schema = json.loads(
            (repo_root / "schemas" / "ingest-report.schema.json").read_text(encoding="utf-8")
        )
        Draft202012Validator(schema).validate(read_report(canonical_fixture.session_dir))

    def test_the_roster_shows_known_observed_and_per_track_counts(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The gate's wording, asserted on the report rather than on discovery."""
        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        roster = result.report.roster
        assert roster is not None
        assert roster.known_tracks == ["tx-a", "tx-b", "tx-c", "tx-d", "tx-e", "tx-f"]
        assert roster.active_tracks == roster.known_tracks
        assert roster.inactive_tracks == []
        assert roster.file_counts == dict.fromkeys(roster.known_tracks, 2)

    def test_it_lists_missing_empty_and_extra_directories(self, tmp_path: Path) -> None:
        """All three at once, because they are easy to conflate and mean different
        things: never plugged in, recorded nothing, and nobody configured it."""
        spec = FixtureSession(
            session_id="2026-08-15",
            title="Session 01",
            tracks=tuple(
                FixtureTrack(
                    track_id=f"tx-{letter}",
                    speaker_id=name.lower(),
                    speaker_name=name,
                    receiver_id=f"rx-{letter}",
                    receiver_channel=1,
                    tx_label="TX01",
                    chunks=(FixtureChunk(0, 4800, sequence=1),),
                )
                for index, (letter, name) in enumerate(
                    [("a", "Alice"), ("b", "Bob"), ("c", "Carol")]
                )
            ),
        )
        build_session(spec, tmp_path)

        shutil.rmtree(tmp_path / "raw/tx-b")
        for path in (tmp_path / "raw/tx-c").iterdir():
            path.unlink()
        (tmp_path / "raw/tx-z").mkdir()

        result = run_inspect(tmp_path, now=EARLY)
        roster = result.report.roster
        assert roster is not None
        assert roster.missing_directories == ["raw/tx-b"]
        assert roster.empty_directories == ["raw/tx-c"]
        assert roster.extra_directories == ["raw/tx-z"]
        assert roster.active_tracks == ["tx-a"]
        assert roster.inactive_tracks == ["tx-b", "tx-c"]

    def test_provenance_carries_the_config_hash_tools_and_command(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The spec's observability list: source hashes, configuration hash, dependency
        versions, and the exact commands used."""
        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        provenance = result.report.provenance

        assert provenance.config_hash == result.manifest.config_hash
        assert set(provenance.tool_versions) == {"ffmpeg", "ffprobe"}
        assert provenance.tool_versions["ffprobe"].startswith("ffprobe version")
        assert provenance.commands == [
            "ffprobe -v error -print_format json -show_format -show_streams -i <source>"
        ]
        assert "dnd_audio.inspection" in provenance.package_versions

    def test_source_hashes_are_in_the_manifest_the_report_hashes(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The report points at the manifest rather than duplicating every hash; the
        link has to be a hash of the actual bytes or it proves nothing."""
        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        deliverables = {d.relative_path: d for d in result.report.provenance.deliverables}
        assert set(deliverables) == {"work/manifest.json"}

        from dnd_audio.determinism import sha256_file

        assert deliverables["work/manifest.json"].sha256 == sha256_file(result.manifest_path)

    def test_the_report_never_lists_itself(self, canonical_fixture: FixtureTruth) -> None:
        """ADR-0003: a file cannot contain the hash of its own final bytes."""
        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        paths = [d.relative_path for d in result.report.provenance.deliverables]
        assert not any(path.endswith("ingest-report.json") for path in paths)

    def test_every_stage_is_accounted_for(self, canonical_fixture: FixtureTruth) -> None:
        result = run_inspect(canonical_fixture.session_dir, now=EARLY)
        statuses = {stage.stage.value: stage.status.value for stage in result.report.stages}
        assert statuses == {
            "inspect": "complete",
            "reconstruct": "skipped",
            "activity": "skipped",
            "transcribe": "skipped",
            "render": "skipped",
            "mix": "skipped",
        }
        for stage in result.report.stages:
            if stage.status.value == "skipped":
                assert stage.skip_reason, f"{stage.stage} was skipped without saying why"

    def test_discovery_warnings_reach_the_report(self, tmp_path: Path) -> None:
        session = override_session(tmp_path)
        (session / "raw/tx-a/notes.txt").write_text("field log", encoding="utf-8")

        result = run_inspect(session, now=EARLY)
        inspect_stage = next(s for s in result.report.stages if s.stage.value == "inspect")
        assert "unexpected_file_type" in {w.code for w in inspect_stage.warnings}


class TestFailurePaths:
    """INV-13, through the real CLI, on five real broken sessions.

    Each asserts the same four things, because all four are what the invariant claims:
    the report exists, `inspect` is failed with a structured error, the five stages that
    did not run say why, and the exit code is nonzero.
    """

    @staticmethod
    def _assert_failed_cleanly(session: Path, code: str) -> dict[str, object]:
        result = runner.invoke(app, ["inspect", str(session)])
        assert result.exit_code == ExitCode.FATAL

        document = read_report(session)
        assert document["overall_status"] == "failed"

        stages = document["stages"]
        assert isinstance(stages, list)
        inspect_stage = next(s for s in stages if s["stage"] == "inspect")
        assert inspect_stage["status"] == "failed"
        assert [error["code"] for error in inspect_stage["errors"]] == [code]

        for stage in stages:
            if stage["stage"] != "inspect":
                assert stage["status"] == "skipped"
                assert stage["skip_reason"]
        return document

    def test_a_source_with_no_timing_evidence(self, tmp_path: Path) -> None:
        session = override_session(tmp_path)
        for path in (session / "raw/tx-a").iterdir():
            path.unlink()
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
                        chunks=(FixtureChunk(0, 4800, sequence=1, timecode_source="none"),),
                    ),
                ),
            ),
            session,
        )
        document = self._assert_failed_cleanly(session, "no_reliable_timecode")
        assert "source_time_overrides" in errors_of(document)[0]["message"]

    def test_a_processed_only_source_without_permission(self, tmp_path: Path) -> None:
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
                        chunks=(FixtureChunk(0, 4800, sequence=1, variant="edit"),),
                    ),
                ),
            ),
            tmp_path,
        )
        self._assert_failed_cleanly(tmp_path, "processed_audio_only")

    def test_a_required_track_with_nothing_usable(self, tmp_path: Path) -> None:
        session = override_session(tmp_path, {"active_tracks: auto": "active_tracks: [tx-a]"})
        for path in (session / "raw/tx-a").iterdir():
            path.unlink()
        self._assert_failed_cleanly(session, "required_track_missing")

    def test_an_override_whose_hash_does_not_match(self, tmp_path: Path) -> None:
        session = override_session(
            tmp_path,
            with_override(
                '    "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav":\n'
                # A hash that is not this file's. Quoted, because an unquoted all-digit
                # value would be YAML's idea of an integer.
                f'      sha256: "{"b" * 64}"\n'
                '      start_timecode: "19:00:00:00"\n'
                '      reason: "from the field log"'
            ),
        )
        document = self._assert_failed_cleanly(session, "recovery_override_unusable")
        assert "expects sha256" in errors_of(document)[0]["message"]

    def test_an_override_that_matches_no_source(self, tmp_path: Path) -> None:
        session = override_session(
            tmp_path,
            with_override(
                '    "raw/tx-a/typo.wav":\n      start_offset_samples: 0\n      reason: "measured"'
            ),
        )
        self._assert_failed_cleanly(session, "recovery_override_unmatched")

    def test_ffprobe_failing_outright(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tool is not always going to work, and when it does not the run has to
        say so rather than producing a manifest with a hole in it."""
        session = override_session(tmp_path)

        def refuse(session_dir: Path, relative_path: str) -> ProbeResult:
            message = f"ffprobe failed on {relative_path} (exit 1): simulated"
            raise ProbeError(message)

        monkeypatch.setattr("dnd_audio.inspection.runner.run_ffprobe", refuse)
        result = run_inspect(session, now=EARLY)

        assert result.exit_code is ExitCode.FATAL
        document = read_report(session)
        assert [error["code"] for error in errors_of(document)] == ["probe_failed"]

    def test_a_session_with_no_configuration(self, tmp_path: Path) -> None:
        (tmp_path / "raw").mkdir()
        self._assert_failed_cleanly(tmp_path, "invalid_configuration")

    def test_no_manifest_is_left_behind_by_a_failed_run(self, tmp_path: Path) -> None:
        """A first run that fails must not leave a manifest.

        The *stale* case — an earlier run succeeded and a later one failed — is the one
        that actually bit, and it lives in
        `test_inspect_run.py::test_a_failed_rerun_removes_the_manifest_the_last_success_left`
        because it needs a successful run first. This test cannot see it, and no longer
        claims to.
        """
        session = override_session(tmp_path, {"active_tracks: auto": "active_tracks: [tx-a]"})
        for path in (session / "raw/tx-a").iterdir():
            path.unlink()

        runner.invoke(app, ["inspect", str(session)])
        assert not (session / "work" / "manifest.json").exists()

    def test_a_failed_run_stages_no_cache_entries(self, tmp_path: Path) -> None:
        session = override_session(
            tmp_path,
            with_override(
                '    "raw/tx-a/typo.wav":\n      start_offset_samples: 0\n      reason: "measured"'
            ),
        )
        runner.invoke(app, ["inspect", str(session)])
        assert not list((session / "work/cache/inspect").glob("*.json"))


class TestPerSourceWarningsReachTheReport:
    """The spec asks the report to carry warnings; nesting them in the manifest hides
    them from every consumer that reads the report, which is what a report is for."""

    def test_an_unexpected_sample_rate_appears_in_the_reports_warnings(
        self, tmp_path: Path
    ) -> None:
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
                        chunks=(FixtureChunk(0, 4410, sequence=1, sample_rate=44100),),
                    ),
                ),
            ),
            tmp_path,
        )
        result = run_inspect(tmp_path, now=EARLY)
        stage = next(s for s in result.report.stages if s.stage.value == "inspect")
        warnings = {w.code: w for w in stage.warnings}

        assert "unexpected_sample_rate" in warnings
        assert warnings["unexpected_sample_rate"].path == (
            "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav"
        )

    def test_the_reports_warnings_are_sorted(self, tmp_path: Path) -> None:
        """Two runs must not differ by warning order (INV-02 for the decision sections)."""
        session = override_session(tmp_path)
        (session / "raw/tx-a/notes.txt").write_text("field log", encoding="utf-8")
        result = run_inspect(session, now=EARLY)
        stage = next(s for s in result.report.stages if s.stage.value == "inspect")
        keys = [(w.code, w.path or "", w.message) for w in stage.warnings]
        assert keys == sorted(keys)

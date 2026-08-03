"""`dnd-audio mix`, end to end on real session directories.

The unit files test the envelope, the cache and the encode loop against constructed inputs.
This one runs the whole composed stage — inspect, reconstruct, activity, mix, encode — against
real audio on disk, which is the only thing that can show the pieces agreeing.

Two habits from earlier closeouts are followed deliberately. Every failure test **starts from
a stale MP3 and report already on disk**, because a test that starts from an empty directory
cannot distinguish "removed the stale one" from "never wrote one". And every claim about
caching is checked by watching whether the work was *entered*, not by observing that its
output exists.

The detector is scripted from the fixture's own declared truth (INV-10); nothing here loads
Silero.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dnd_audio.activity import ACTIVITY_RELATIVE_PATH
from dnd_audio.activity.runner import DetectorBundle
from dnd_audio.artifacts.activity import ActivityGraph
from dnd_audio.artifacts.report import StageName, StageStatus
from dnd_audio.determinism import sha256_file
from dnd_audio.errors import ExitCode
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.inspection import OUTPUT_DIRNAME
from dnd_audio.mix import MIX_CACHE_DIRNAME, MP3_RELATIVE_PATH
from dnd_audio.mix.encode import encode_mp3
from dnd_audio.mix.loudness import Measurement
from dnd_audio.mix.render import render_mix
from dnd_audio.mix.runner import MixResult, mix_outputs, run_mix
from dnd_audio.transcript.runner import run_transcribe

REPORT = f"{OUTPUT_DIRNAME}/ingest-report.json"

#: Substituted for every free-text field on the graph, so a mixer that read one would produce
#: different samples and the assertion would fail rather than pass on a rebuilt document.
_NO_SAMPLE_MAY_DEPEND_ON_THIS = "REWRITTEN PROSE THAT NO SAMPLE MAY DEPEND ON"


def scripted(session_dir: Path) -> DetectorBundle:
    """The session's own declared fake detector — speech *and* bleed, as a real one finds."""
    from dnd_audio.transcript.fakemodels import load_fake_models

    return load_fake_models(session_dir).detector


def mixed(canonical_fixture: FixtureTruth, **kwargs: Any) -> MixResult:
    result = run_mix(
        canonical_fixture.session_dir, detector=scripted(canonical_fixture.session_dir), **kwargs
    )
    assert result.encode is not None, [
        f"{error.code}: {error.message}" for stage in result.report.stages for error in stage.errors
    ]
    return result


def stale_artifacts(session_dir: Path) -> tuple[Path, Path]:
    """Plant an MP3 and a report from an imaginary earlier run."""
    mp3 = session_dir / MP3_RELATIVE_PATH
    report = session_dir / REPORT
    mp3.parent.mkdir(parents=True, exist_ok=True)
    mp3.write_bytes(b"not an mp3, but it looks like a deliverable")
    report.write_text(json.dumps({"stale": True}), encoding="utf-8")
    return mp3, report


class TestTheCanonicalSession:
    def test_it_produces_an_mp3_that_decodes_and_meets_every_target(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Acceptance criterion 8, end to end: exists, decodes, mono, near the expected
        duration, and inside the configured true-peak ceiling."""
        result = mixed(canonical_fixture)
        assert result.exit_code is ExitCode.OK
        assert result.mp3_path.exists()

        encoded = result.encode
        assert encoded is not None
        assert encoded.facts.codec == "mp3"
        assert encoded.facts.channels == 1
        assert encoded.facts.bit_rate_kbps == 128
        assert encoded.facts.tags["title"] == "Session 01"
        assert encoded.facts.tags["album"] == "2026-08-15"

        decoded = encoded.accepted.measurement
        assert abs(decoded.n_samples - 504_000) <= 1152
        assert decoded.true_peak_dbtp_mb is not None
        assert decoded.true_peak_dbtp_mb <= -150 + 30

    def test_the_report_covers_all_six_stages(self, canonical_fixture: FixtureTruth) -> None:
        result = mixed(canonical_fixture)
        status = {stage.stage: stage.status for stage in result.report.stages}
        assert status[StageName.INSPECT] is StageStatus.COMPLETE
        assert status[StageName.RECONSTRUCT] is StageStatus.COMPLETE
        assert status[StageName.ACTIVITY] is StageStatus.COMPLETE
        assert status[StageName.MIX] is StageStatus.COMPLETE
        assert status[StageName.TRANSCRIBE] is StageStatus.SKIPPED
        assert status[StageName.RENDER] is StageStatus.SKIPPED
        for stage in result.report.stages:
            if stage.status is StageStatus.SKIPPED:
                assert stage.skip_reason

    def test_the_mp3_is_a_deliverable_and_the_intermediate_is_not(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The spec puts the intermediate in `work/` "not as a required user-facing
        deliverable", so it is cache and the report does not advertise it."""
        result = mixed(canonical_fixture)
        paths = {item.relative_path for item in result.report.provenance.deliverables}
        assert MP3_RELATIVE_PATH in paths
        assert not any(path.startswith("work/cache/mix") for path in paths)

    def test_every_measurement_and_correction_reaches_the_report(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """ "Retain all measurements in the report." M5 publishes no document of its own
        (ADR-0022), so this subsection is the audit trail."""
        result = mixed(canonical_fixture)
        codes = {decision.code for decision in result.report.decisions}
        assert {
            "mix_level_correction",
            "mix_intermediate",
            "mix_encode_attempt",
            "mix_encoded",
        } <= (codes)

        attempts = [d for d in result.report.decisions if d.code == "mix_encode_attempt"]
        assert attempts
        for attempt in attempts:
            assert {"gain_mb", "integrated_lufs_mb", "true_peak_dbtp_mb", "decoded_samples"} <= set(
                attempt.details
            )

    def test_the_exact_ffmpeg_commands_are_recorded(self, canonical_fixture: FixtureTruth) -> None:
        """The spec's observability section asks for them by name."""
        result = mixed(canonical_fixture)
        commands = result.report.provenance.commands
        assert any("ebur128" in command for command in commands)
        assert any("libmp3lame" in command for command in commands)
        assert "ffmpeg" in result.report.provenance.tool_versions

    def test_a_second_run_reuses_the_intermediate(self, canonical_fixture: FixtureTruth) -> None:
        """Checked by watching whether the renderer was *entered*, not by observing that the
        file exists — the habit M1's closeout records."""
        mixed(canonical_fixture)

        entered = 0
        original = render_mix

        def watched(*args: Any, **kwargs: Any) -> Any:
            nonlocal entered
            entered += 1
            return original(*args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("dnd_audio.mix.runner.render_mix", watched)
            second = mixed(canonical_fixture)

        assert entered == 0
        assert second.exit_code is ExitCode.OK
        assert second.report.telemetry.cache_hits > 0

    def test_the_decision_subsection_does_not_move_between_a_cold_and_a_warm_run(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """INV-02: the report as a whole is exempt, its decision subsection is not.

        The first version of `mix_intermediate` carried `from_cache` and said "rendered" or
        "reused from cache", so two runs over an unchanged session disagreed about a session
        neither had changed. Cache state is telemetry, and `record_cache` already carries it.
        Found by M5's code review.
        """
        cold = mixed(canonical_fixture, use_cache=False)
        warm = mixed(canonical_fixture)
        assert warm.report.telemetry.cache_hits > 0, "the warm run did not hit the cache"

        def decisions(result: MixResult) -> list[dict[str, Any]]:
            return [item.model_dump(mode="json") for item in result.report.decisions]

        assert decisions(warm) == decisions(cold)

    def test_no_cache_re_renders(self, canonical_fixture: FixtureTruth) -> None:
        """The contrast, so the test above is about the cache rather than about the run."""
        mixed(canonical_fixture)

        entered = 0
        original = render_mix

        def watched(*args: Any, **kwargs: Any) -> Any:
            nonlocal entered
            entered += 1
            return original(*args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("dnd_audio.mix.runner.render_mix", watched)
            mixed(canonical_fixture, use_cache=False)

        assert entered == 1

    def test_the_intermediate_is_byte_identical_across_runs(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """INV-02 on the artifact M5 caches. Re-rendered from scratch both times."""
        session_dir = canonical_fixture.session_dir
        mixed(canonical_fixture, use_cache=False)
        first = {
            path: sha256_file(path) for path in (session_dir / MIX_CACHE_DIRNAME).glob("*.wav")
        }
        assert first

        mixed(canonical_fixture, use_cache=False)
        second = {
            path: sha256_file(path) for path in (session_dir / MIX_CACHE_DIRNAME).glob("*.wav")
        }
        assert second == first


class TestInv09:
    """The mix consumes the graph and nothing downstream of it. Three separate proofs."""

    def test_nothing_in_the_mix_package_reaches_the_transcript_layer(self) -> None:
        """The **transitive** import closure, in a subprocess, rather than a grep of one
        directory for one string.

        A grep sees a direct import and nothing else; this sees an import three modules deep,
        which is how a dependency actually arrives. The technique is `test_silero.py`'s, which
        proves Torch is never imported at all.
        """
        program = (
            "import sys, importlib;"
            "importlib.import_module('dnd_audio.mix.runner');"
            "leaked=[m for m in sys.modules if m.startswith('dnd_audio.transcript')];"
            "print(leaked)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=True
        )
        assert completed.stdout.strip() == "[]", completed.stdout

    def test_the_intermediate_does_not_move_when_the_transcript_branch_runs(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The mix must produce identical samples whether or not ASR ran at all."""
        session_dir = canonical_fixture.session_dir
        mixed(canonical_fixture, use_cache=False)
        before = {
            path.name: sha256_file(path) for path in (session_dir / MIX_CACHE_DIRNAME).glob("*.wav")
        }

        transcribed = run_transcribe(session_dir, fake_models=True)
        assert transcribed.records is not None

        mixed(canonical_fixture, use_cache=False)
        after = {
            path.name: sha256_file(path) for path in (session_dir / MIX_CACHE_DIRNAME).glob("*.wav")
        }
        assert after == before

    def test_rewriting_the_graphs_prose_does_not_change_a_single_sample(
        self,
        canonical_fixture: FixtureTruth,
        canonical_activity_graph: ActivityGraph,
        tmp_path: Path,
    ) -> None:
        """The prohibition the INV-09 field allowlist cannot express.

        `ActivityDecision.detail` and `ActivityNote.message` are unrestricted strings on the
        frozen contract, so nothing structural stops the mixer reading them — M3's review
        raised exactly this and M5's charter carries it as a risk.

        The prose is rewritten **on the graph object the renderer is handed**, not on
        `work/activity.json`. Rewriting the file was this test's first shape and it proved
        nothing: `run_mix` rebuilds that document from the attribution cache before mixing, so
        the edit was overwritten and the mixer never saw it. A mixer that read
        `ActivityDecision.detail` passed. Found by M5's code review; the two renders below go
        to different destinations so no cache can stand in for either.
        """
        session_dir, timeline = self._reconstructed(canonical_fixture)
        rewritten = canonical_activity_graph.model_copy(
            update={
                "warnings": [
                    note.model_copy(update={"message": _NO_SAMPLE_MAY_DEPEND_ON_THIS})
                    for note in canonical_activity_graph.warnings
                ],
                "decisions": [
                    decision.model_copy(update={"detail": _NO_SAMPLE_MAY_DEPEND_ON_THIS})
                    for decision in canonical_activity_graph.decisions
                ],
            }
        )
        assert canonical_activity_graph.warnings or canonical_activity_graph.decisions, (
            "the fixture's graph has no free text in it, so this test proves nothing"
        )
        assert rewritten != canonical_activity_graph, "the rewrite changed nothing"

        plain = self._render(session_dir, timeline, canonical_activity_graph, tmp_path / "a.wav")
        edited = self._render(session_dir, timeline, rewritten, tmp_path / "b.wav")
        assert edited == plain

    @staticmethod
    def _reconstructed(fixture: FixtureTruth) -> tuple[Path, Any]:
        from dnd_audio.timeline.runner import run_ingest

        result = run_ingest(fixture.session_dir)
        assert result.timeline is not None
        return fixture.session_dir, result.timeline

    @staticmethod
    def _render(session_dir: Path, timeline: Any, graph: ActivityGraph, destination: Path) -> bytes:
        from dnd_audio.config import EnvelopeConfig
        from dnd_audio.mix.envelope import EnvelopeStream
        from dnd_audio.mix.levels import level_corrections

        settings = EnvelopeConfig()
        track_ids = tuple(track.track_id for track in timeline.tracks)
        render_mix(
            destination,
            session_dir=session_dir,
            timeline=timeline,
            track_ids=track_ids,
            envelope=EnvelopeStream(
                graph,
                settings=settings,
                corrections=level_corrections(graph, settings=settings),
                track_ids=track_ids,
            ),
        )
        return destination.read_bytes()


class TestFailures:
    def test_a_missing_session_file_still_writes_a_report(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        session_dir = canonical_fixture.session_dir
        _, report = stale_artifacts(session_dir)
        (session_dir / "session.yaml").unlink()

        result = run_mix(session_dir)

        assert result.exit_code is not ExitCode.OK
        assert result.report_written
        assert json.loads(report.read_text(encoding="utf-8")) != {"stale": True}

    def test_a_failed_run_removes_the_stale_mp3(self, canonical_fixture: FixtureTruth) -> None:
        """A deliverable that looks current beside a report calling its stage failed is worse
        than none."""
        session_dir = canonical_fixture.session_dir
        mp3, _ = stale_artifacts(session_dir)
        (session_dir / "session.yaml").unlink()

        result = run_mix(session_dir)

        assert result.exit_code is not ExitCode.OK
        assert not mp3.exists()

    def test_an_encode_failure_keeps_the_activity_artifacts_it_already_hashed(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Two commit points, and the reason for them (ADR-0021).

        The encode reads no source audio, so a failure there cannot invalidate the graph — and
        deleting a stage's artifacts after the report has hashed them would leave it
        advertising a file that is gone.
        """
        session_dir = canonical_fixture.session_dir

        def exploding(*args: Any, **kwargs: Any) -> Any:
            message = "the encoder fell over"
            raise OSError(message)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("dnd_audio.mix.runner.encode_mp3", exploding)
            result = run_mix(session_dir, detector=scripted(session_dir))

        assert result.exit_code is not ExitCode.OK
        status = {stage.stage: stage.status for stage in result.report.stages}
        assert status[StageName.ACTIVITY] is StageStatus.COMPLETE
        assert status[StageName.MIX] is StageStatus.FAILED
        assert (session_dir / ACTIVITY_RELATIVE_PATH).exists()
        paths = {item.relative_path for item in result.report.provenance.deliverables}
        assert ACTIVITY_RELATIVE_PATH in paths

    def test_an_uncompliant_encode_still_records_every_measurement_it_took(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The gate asks for all measurements retained **in the report**, and the run that
        matters is the one that failed. Recording only on the success path left an exhausted
        retry budget with one error string. Found by M5's code review."""
        session_dir = canonical_fixture.session_dir
        overshooting = Measurement(
            integrated_lufs_mb=-1600, true_peak_dbtp_mb=500, n_samples=504_000, command="scripted"
        )
        real = encode_mp3

        def never_compliant(*args: Any, **kwargs: Any) -> Any:
            """The real retry loop, driven by a decode that always overshoots the ceiling."""
            return real(*args, **{**kwargs, "measurer": lambda _path: overshooting})

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("dnd_audio.mix.runner.encode_mp3", never_compliant)
            result = run_mix(session_dir, detector=scripted(session_dir))

        assert result.exit_code is not ExitCode.OK
        attempts = [d for d in result.report.decisions if d.code == "mix_encode_attempt"]
        assert len(attempts) == 4, "one first attempt plus the default three retries"
        assert all("true_peak" in d.details["failures"] for d in attempts)
        assert any(d.code == "mix_intermediate" for d in result.report.decisions)
        assert sum("libmp3lame" in c for c in result.report.provenance.commands) == 4

    def test_a_partial_run_never_exits_zero(self, canonical_fixture: FixtureTruth) -> None:
        """INV-13, on the branch that can genuinely be half successful."""
        session_dir = canonical_fixture.session_dir

        def exploding(*args: Any, **kwargs: Any) -> Any:
            message = "the encoder fell over"
            raise OSError(message)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("dnd_audio.mix.runner.encode_mp3", exploding)
            result = run_mix(session_dir, detector=scripted(session_dir))

        assert result.exit_code is ExitCode.PARTIAL


class TestTheOutputSet:
    def test_it_declares_the_mp3_and_the_cache_as_well_as_everything_activity_writes(
        self, tmp_path: Path
    ) -> None:
        """Declared as data so that adding an output and forgetting to protect it is a visible
        omission from one list (INV-01)."""
        outputs = mix_outputs(tmp_path)
        assert outputs["the mixed MP3"] == tmp_path / MP3_RELATIVE_PATH
        assert outputs["the mix cache"] == tmp_path / MIX_CACHE_DIRNAME
        assert "the activity graph" in outputs
        assert "the manifest" in outputs or any("manifest" in label for label in outputs)

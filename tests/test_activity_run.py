"""`dnd-audio activity`, end to end, on real session directories.

`tests/test_activity_bleed.py` tests the rule against synthetic acoustics. This file runs
the whole composed stage — inspect, reconstruct, detect, attribute — against real audio on
disk, which is the only thing that can show the pieces agreeing: that the detector really is
driven over the 16 kHz derivative the timeline named, that a lag survives resampling, that a
failure still leaves a report behind, and that a rerun is byte-identical.

Two habits from M1's and M2's closeouts are followed deliberately. Every failure test
**starts from a stale graph and report already on disk**, because a test that starts from an
empty directory cannot distinguish "removed the stale one" from "never wrote one". And every
claim about caching is checked by watching whether the work was *entered*, not by observing
that its output exists.

The detector is scripted from the fixture's own declared truth (INV-10). The real Silero
model is exercised in `tests/test_silero.py` under `host_smoke`; nothing here loads it.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import pytest
import yaml

from dnd_audio.activity import ACTIVITY_RELATIVE_PATH, DETECTION_DIRNAME
from dnd_audio.activity.runner import ActivityResult, DetectorBundle, run_activity
from dnd_audio.artifacts.activity import ActivityGraph
from dnd_audio.artifacts.report import StageName, StageStatus
from dnd_audio.errors import ExitCode
from dnd_audio.fakes import ScriptedActivityDetector
from dnd_audio.fixtures import FixtureSession, FixtureTruth, build_session
from dnd_audio.fixtures.variants import (
    DELAYED_BLEED_SAMPLES,
    delayed_bleed_session,
    mutual_bleed_session,
)
from dnd_audio.inspection import OUTPUT_DIRNAME
from dnd_audio.interfaces import AudioWindow, SpeechSpan
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE, TIMELINE_RELATIVE_PATH

#: The fixture's declared bleed delay, on the grid the detector works at.
CANONICAL_BLEED_LAG: Final = 144 // 3
DELAYED_BLEED_LAG: Final = DELAYED_BLEED_SAMPLES // 3


def leaky(truth: FixtureTruth) -> DetectorBundle:
    """A detector that fires on bleed as well as speech — which is what a real one does.

    Driven from `FixtureTruth.leaky_activity_spans`, at the **derivative** rate: the
    detection pass hands out windows of the 16 kHz audio, so 48 kHz spans would land past the
    end of every window and the graph would come out empty with every assertion still
    passing. That failure is silent, which is why the helper takes the rate explicitly.
    """
    detector = ScriptedActivityDetector(
        truth.leaky_activity_spans(sample_rate=DERIVATIVE_SAMPLE_RATE)
    )
    return DetectorBundle(identity=detector.identity(), make=lambda _track: detector)


def honest(truth: FixtureTruth) -> DetectorBundle:
    """A detector that fires only on real speech. The fixture's ground truth, unmodified."""
    detector = ScriptedActivityDetector(truth.activity_spans(sample_rate=DERIVATIVE_SAMPLE_RATE))
    return DetectorBundle(identity=detector.identity(), make=lambda _track: detector)


@pytest.fixture
def a_session(tmp_path: Path) -> Callable[[FixtureSession], FixtureTruth]:
    def build(spec: FixtureSession) -> FixtureTruth:
        return build_session(spec, tmp_path / spec.session_id)

    return build


def stale_artifacts(session_dir: Path) -> tuple[Path, Path]:
    """Plant a graph and a report from an imaginary earlier run.

    Every failure test starts from these. Beginning with an empty directory would let a test
    pass whether the code removed a stale artifact or simply never wrote one — the exact
    defect M1's closeout records finding in its own suite.
    """
    graph = session_dir / ACTIVITY_RELATIVE_PATH
    report = session_dir / OUTPUT_DIRNAME / "ingest-report.json"
    for path, payload in ((graph, {"stale": True}), (report, {"stale": True})):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    return graph, report


def graph_of(result: ActivityResult) -> ActivityGraph:
    assert result.graph is not None, [
        f"{error.code}: {error.message}" for stage in result.report.stages for error in stage.errors
    ]
    return result.graph


class TestTheCanonicalSession:
    """The four cases M3's gate names, on the six-transmitter fixture."""

    def test_solo_speech_is_attributed_to_the_speaker_and_nobody_else(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """tx-a speaks at 5.2 s and four other lavs hear it. One candidate should survive."""
        graph = graph_of(
            run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))
        )
        speaking = [c for c in graph.candidates if c.start_sample < 300000]

        owner = [c for c in speaking if c.decision == "retained"]
        assert [c.track_id for c in owner] == ["tx-a"]
        assert {c.track_id for c in speaking if c.decision == "suppressed"} == {
            "tx-b",
            "tx-d",
            "tx-e",
            "tx-f",
        }
        assert all(
            c.suppressed_by_candidate_id == owner[0].candidate_id
            for c in speaking
            if c.decision == "suppressed"
        )

    def test_the_reported_lag_is_the_delay_the_fixture_wrote(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """3 ms of air, through a 3:1 decimation, still 3 ms.

        The number is the fixture's own declaration divided by the decimation factor, not
        anything read back from the pipeline.
        """
        graph = graph_of(
            run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))
        )
        suppressed = [c for c in graph.candidates if c.decision == "suppressed"]
        assert suppressed
        for candidate in suppressed:
            record = next(
                item
                for item in candidate.evidence
                if item.other_candidate_id == candidate.suppressed_by_candidate_id
            )
            assert record.lag_derivative_samples == CANONICAL_BLEED_LAG
            assert record.correlation_permille >= 900

    def test_genuine_two_person_overlap_survives_on_both_tracks(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """tx-d and tx-e speak at once at 6.8 s. Losing either is the failure that matters."""
        graph = graph_of(
            run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))
        )
        overlap = [c for c in graph.candidates if 300000 < c.start_sample < 400000]
        assert {c.track_id for c in overlap} == {"tx-d", "tx-e"}
        assert all(c.decision == "retained" for c in overlap)

    def test_speech_after_a_gap_is_kept(self, canonical_fixture: FixtureTruth) -> None:
        """tx-c was switched off and came back. Its 8.5 s line is real speech."""
        graph = graph_of(
            run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))
        )
        late = [c for c in graph.candidates if c.start_sample > 400000]
        assert [(c.track_id, c.decision) for c in late] == [("tx-c", "retained")]

    def test_every_candidate_covers_the_speech_the_fixture_declared(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Against the generator's declared truth, not against anything read back.

        Padding and the detector's 32 ms frame both widen a region, so the candidate must
        *contain* the declared interval — a candidate that merely overlaps it would have
        clipped a word.
        """
        graph = graph_of(
            run_activity(canonical_fixture.session_dir, detector=honest(canonical_fixture))
        )
        for interval in canonical_fixture.speech:
            owner = [
                c
                for c in graph.candidates
                if c.track_id == interval.track_id
                and c.start_sample <= interval.start_sample
                and c.end_sample >= interval.end_sample
            ]
            assert owner, f"no candidate covers {interval.track_id} at {interval.start_sample}"

    def test_an_honest_detector_suppresses_nothing(self, canonical_fixture: FixtureTruth) -> None:
        """Ground truth has no bleed in it, so nothing should be taken away.

        The contrast with the leaky runs above: the gate reacts to what the detector found,
        not to the fixture's geometry.
        """
        graph = graph_of(
            run_activity(canonical_fixture.session_dir, detector=honest(canonical_fixture))
        )
        assert all(c.decision == "retained" for c in graph.candidates)
        assert [c.track_id for c in graph.candidates] == ["tx-a", "tx-d", "tx-e", "tx-c"]


class TestTheBleedFixtures:
    def test_a_bleed_delayed_near_the_edge_of_the_window_is_still_found(
        self, a_session: Callable[[FixtureSession], FixtureTruth]
    ) -> None:
        """25 ms, against a ±30 ms window. A zero-lag correlator finds nothing here."""
        truth = a_session(delayed_bleed_session())
        graph = graph_of(run_activity(truth.session_dir, detector=leaky(truth)))

        suppressed = [c for c in graph.candidates if c.decision == "suppressed"]
        assert [c.track_id for c in suppressed] == ["tx-b"]
        record = suppressed[0].evidence[0]
        assert record.lag_derivative_samples == DELAYED_BLEED_LAG
        assert record.correlation_permille >= 900

    def test_two_real_speakers_at_unequal_levels_both_survive(
        self, a_session: Callable[[FixtureSession], FixtureTruth]
    ) -> None:
        """The case review produced against the first plan (ADR-0014).

        Both numeric conditions point at bleed; the veto keeps the quiet speaker. The
        threshold-by-threshold proof — and the contrast that shows the veto is what did it —
        is in `tests/test_activity_bleed.py`; this asserts the whole pipeline reaches the
        same answer over real audio.
        """
        truth = a_session(mutual_bleed_session())
        graph = graph_of(run_activity(truth.session_dir, detector=leaky(truth)))

        # Padding and the detector's 32 ms frame both move a candidate's start earlier than
        # the sample the fixture declared, so the window is generous on that side.
        overlap = [c for c in graph.candidates if c.start_sample >= 13 * 48000]
        assert {c.track_id for c in overlap} == {"tx-a", "tx-b"}
        assert all(c.decision == "retained" for c in overlap)

        quiet = next(c for c in overlap if c.track_id == "tx-b")
        record = quiet.evidence[0]
        assert record.outcome == "vetoed_by_track_level"
        assert record.correlation_permille >= 500
        assert record.score_margin_permille >= 150
        assert quiet.ambiguous is True

    def test_a_track_with_enough_speech_gets_a_reference(
        self, a_session: Callable[[FixtureSession], FixtureTruth]
    ) -> None:
        truth = a_session(mutual_bleed_session())
        graph = graph_of(run_activity(truth.session_dir, detector=leaky(truth)))
        assert all(track.speech_reference_mbfs is not None for track in graph.tracks)


class TestDeterminism:
    def test_a_rerun_is_byte_identical(self, canonical_fixture: FixtureTruth) -> None:
        """INV-02. Unchanged input, unchanged configuration, unchanged bytes."""
        first = run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))
        before = (canonical_fixture.session_dir / ACTIVITY_RELATIVE_PATH).read_bytes()
        second = run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))
        after = (canonical_fixture.session_dir / ACTIVITY_RELATIVE_PATH).read_bytes()

        assert first.exit_code is ExitCode.OK
        assert second.exit_code is ExitCode.OK
        assert before == after

    def test_a_cold_rerun_produces_the_same_bytes_as_a_warm_one(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The cache must not be the reason two runs agree.

        Without this, a byte-stability test proves only that the second run read the first
        one's output back — which is true of any cache, including a wrong one.
        """
        run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))
        warm = (canonical_fixture.session_dir / ACTIVITY_RELATIVE_PATH).read_bytes()
        run_activity(
            canonical_fixture.session_dir, detector=leaky(canonical_fixture), use_cache=False
        )
        cold = (canonical_fixture.session_dir / ACTIVITY_RELATIVE_PATH).read_bytes()
        assert warm == cold

    def test_no_float_reaches_the_document(self, canonical_fixture: FixtureTruth) -> None:
        """Walked over the real artifact, not over a hand-built one."""
        run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))
        document = json.loads((canonical_fixture.session_dir / ACTIVITY_RELATIVE_PATH).read_text())
        assert _floats(document, "") == []


def _floats(value: Any, path: str) -> list[str]:
    if isinstance(value, bool):
        return []
    if isinstance(value, float):
        return [f"{path} = {value!r}"]
    if isinstance(value, dict):
        return [f for key, item in value.items() for f in _floats(item, f"{path}.{key}")]
    if isinstance(value, list):
        return [f for i, item in enumerate(value) for f in _floats(item, f"{path}[{i}]")]
    return []


class TestCaching:
    def test_a_second_run_reuses_both_caches(self, canonical_fixture: FixtureTruth) -> None:
        first = run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))
        second = run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))
        assert first.report.telemetry.cache_misses > 0
        assert second.report.telemetry.cache_misses == 0
        assert second.report.telemetry.cache_hits > first.report.telemetry.cache_hits

    def test_the_detector_is_not_entered_on_a_warm_run(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Proved by watching the detector, not by counting cache entries.

        A cache key nothing consults would pass a hit-count assertion perfectly well.
        """
        run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))

        calls: list[str] = []

        class Watchful(ScriptedActivityDetector):
            def detect(self, window: AudioWindow) -> tuple[SpeechSpan, ...]:
                calls.append(window.track_id)
                return super().detect(window)

        watched = Watchful(
            canonical_fixture.leaky_activity_spans(sample_rate=DERIVATIVE_SAMPLE_RATE)
        )
        bundle = DetectorBundle(identity=watched.identity(), make=lambda _track: watched)
        run_activity(canonical_fixture.session_dir, detector=bundle)
        assert calls == []

    def test_tuning_a_bleed_threshold_reuses_every_detection(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """ADR-0016's whole claim, end to end.

        A bleed threshold cannot change a per-frame probability, so re-running with a new one
        must not re-enter the detector — and on a real session that is the difference between
        seconds and hours.
        """
        run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))

        document = yaml.safe_load((canonical_fixture.session_dir / "session.yaml").read_text())
        # The margin rather than the correlation: a delayed copy of the same synthetic signal
        # correlates at essentially 1.000, so no correlation threshold below 1 changes the
        # outcome and the assertion below would prove nothing.
        document.setdefault("activity", {})["bleed"] = {"min_score_margin": 0.99}
        (canonical_fixture.session_dir / "session.yaml").write_text(yaml.safe_dump(document))

        calls: list[str] = []

        class Watchful(ScriptedActivityDetector):
            def detect(self, window: AudioWindow) -> tuple[SpeechSpan, ...]:
                calls.append(window.track_id)
                return super().detect(window)

        watched = Watchful(
            canonical_fixture.leaky_activity_spans(sample_rate=DERIVATIVE_SAMPLE_RATE)
        )
        bundle = DetectorBundle(identity=watched.identity(), make=lambda _track: watched)
        result = run_activity(canonical_fixture.session_dir, detector=bundle)

        assert calls == [], "a bleed threshold must not re-run inference"
        graph = graph_of(result)
        assert all(c.decision == "retained" for c in graph.candidates), (
            "at a required margin of 0.99 nothing should be suppressed, which is what shows "
            "attribution really did re-run"
        )

    def test_the_probability_file_is_written_and_sized(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        graph = graph_of(
            run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))
        )
        for track in graph.tracks:
            path = canonical_fixture.session_dir / track.probability_relative_path
            assert path.parent == canonical_fixture.session_dir / DETECTION_DIRNAME
            assert path.stat().st_size == track.probability_frames * 2


class TestTheReport:
    def test_three_stages_are_recorded_and_the_rest_are_skipped(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        result = run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))
        status = {stage.stage: stage.status for stage in result.report.stages}
        assert status[StageName.INSPECT] is StageStatus.COMPLETE
        assert status[StageName.RECONSTRUCT] is StageStatus.COMPLETE
        assert status[StageName.ACTIVITY] is StageStatus.COMPLETE
        assert status[StageName.TRANSCRIBE] is StageStatus.SKIPPED
        assert status[StageName.MIX] is StageStatus.SKIPPED
        assert all(
            stage.skip_reason
            for stage in result.report.stages
            if stage.status is StageStatus.SKIPPED
        )

    def test_both_deliverables_are_hashed(self, canonical_fixture: FixtureTruth) -> None:
        result = run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))
        produced = {item.relative_path for item in result.report.provenance.deliverables}
        assert {TIMELINE_RELATIVE_PATH, ACTIVITY_RELATIVE_PATH} <= produced

    def test_the_detector_identity_reaches_the_report(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """INV-08 and the gate both require it: what decided this is part of the record."""
        bundle = leaky(canonical_fixture)
        result = run_activity(canonical_fixture.session_dir, detector=bundle)
        assert bundle.identity.variant_digest is not None
        assert result.report.provenance.model_identity["vad"] == bundle.identity.variant_digest

    def test_the_scoring_diagnostics_reach_the_report(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The gate asks for the scoring function's diagnostics in `ingest-report.json`.

        Prose is not a diagnostic. An operator asking why a speaker disappeared needs the
        four terms and the three measurements that decided it, in the artifact they open
        first — and they must equal the graph's own values, not a second rounding of them.
        """
        result = run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))
        graph = graph_of(result)
        suppressed = next(c for c in graph.candidates if c.decision == "suppressed")
        decision = next(
            item for item in result.report.decisions if item.subject == suppressed.candidate_id
        )

        assert decision.details["score_permille"] == str(suppressed.score_permille)
        assert decision.details["score_level_permille"] == str(suppressed.score_level_permille)
        assert decision.details["score_confidence_permille"] == str(
            suppressed.score_confidence_permille
        )
        assert decision.details["score_dominance_permille"] == str(
            suppressed.score_dominance_permille
        )
        assert decision.details["score_correlation_permille"] == str(
            suppressed.score_correlation_permille
        )
        assert decision.details["against_candidate_id"] == suppressed.suppressed_by_candidate_id
        assert decision.details["outcome"] == "suppresses"
        assert decision.details["lag_derivative_samples"] == str(CANONICAL_BLEED_LAG)

    def test_a_veto_records_the_evidence_it_overrode(
        self, a_session: Callable[[FixtureSession], FixtureTruth]
    ) -> None:
        """A retained candidate has no suppressor, so the near-miss is what needs showing."""
        truth = a_session(mutual_bleed_session())
        result = run_activity(truth.session_dir, detector=leaky(truth))
        vetoed = next(item for item in result.report.decisions if item.code == "bleed_vetoed")
        assert vetoed.details["outcome"] == "vetoed_by_track_level"
        assert int(vetoed.details["score_margin_permille"]) >= 150
        assert int(vetoed.details["correlation_permille"]) >= 500
        assert vetoed.details["relative_level_mb"] != "unknown"

    def test_every_suppression_is_an_auditable_decision(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The spec requires rejected alternatives to be recorded, in the report as well."""
        result = run_activity(canonical_fixture.session_dir, detector=leaky(canonical_fixture))
        graph = graph_of(result)
        suppressed = {c.candidate_id for c in graph.candidates if c.decision == "suppressed"}
        recorded = {
            decision.subject
            for decision in result.report.decisions
            if decision.code == "bleed_suppressed"
        }
        assert suppressed == recorded
        assert suppressed


class TestFailuresStillProduceAReport:
    """INV-13, across a composed run. Every one starts from stale artifacts on disk."""

    def test_an_absent_model_fails_the_activity_stage(
        self, canonical_fixture: FixtureTruth, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The default path, with no model fetched. It must fail nameably, not traceback."""
        monkeypatch.setenv("DND_AUDIO_MODELS_DIR", str(tmp_path / "empty-models"))
        graph_path, _ = stale_artifacts(canonical_fixture.session_dir)

        result = run_activity(canonical_fixture.session_dir)

        assert result.exit_code is not ExitCode.OK
        assert not graph_path.exists(), "a failed run must not leave a stale graph behind"
        failed = next(stage for stage in result.report.stages if stage.stage is StageName.ACTIVITY)
        assert failed.status is StageStatus.FAILED
        assert failed.errors[0].code == "model_unavailable"
        assert "models fetch" in failed.errors[0].message

    def test_a_detector_that_raises_fails_the_stage_and_writes_a_report(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        class Exploding(ScriptedActivityDetector):
            def detect(self, window: AudioWindow) -> tuple[SpeechSpan, ...]:
                message = "the detector fell over"
                raise RuntimeError(message)

        detector = Exploding({})
        graph_path, report_path = stale_artifacts(canonical_fixture.session_dir)
        result = run_activity(
            canonical_fixture.session_dir,
            detector=DetectorBundle(identity=detector.identity(), make=lambda _t: detector),
        )

        assert result.exit_code is not ExitCode.OK
        assert not graph_path.exists()
        assert json.loads(report_path.read_text())["overall_status"] != "complete"
        failed = next(s for s in result.report.stages if s.stage is StageName.ACTIVITY)
        assert failed.status is StageStatus.FAILED
        assert "fell over" in failed.errors[0].message

    def test_a_source_changed_mid_run_fails_and_commits_no_cache(
        self, canonical_fixture: FixtureTruth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """INV-01 and INV-08 together, in the order M2's closeout says they must run.

        A run that reads a source, builds work from those bytes, and then discovers the file
        changed must leave *nothing* behind — otherwise restoring the original file makes the
        poisoned entry a valid hit forever.

        Asserted over **every** cache directory rather than over detection alone. Checking
        only the cache this milestone added is how the same defect survived one layer up:
        `_inspect` published its sidecars itself, three lines below a docstring promising it
        did not, and a test naming `DETECTION_DIRNAME` could never have seen it (M3's verify
        phase). A glob has no such blind spot, including for whatever M5 caches next.
        """
        session_dir = canonical_fixture.session_dir
        stale_artifacts(session_dir)
        target = session_dir / canonical_fixture.chunks[0].relative_path
        original = target.read_bytes()

        from dnd_audio.activity.bleed import attribute

        def corrupting(*args: Any, **kwargs: Any) -> Any:
            target.write_bytes(original[:-4] + b"\x00\x00\x00\x00")
            return attribute(*args, **kwargs)

        # Patched by name on the *runner*, which imported it directly: patching the defining
        # module would leave the runner holding the original reference.
        monkeypatch.setattr("dnd_audio.activity.runner.attribute", corrupting)
        result = run_activity(session_dir, detector=leaky(canonical_fixture))

        assert result.exit_code is not ExitCode.OK
        assert not (session_dir / ACTIVITY_RELATIVE_PATH).exists()
        sidecars = sorted(
            path.relative_to(session_dir).as_posix()
            for path in (session_dir / "work" / "cache").rglob("*.json")
        )
        assert sidecars == [], (
            f"a run that failed INV-01 committed {len(sidecars)} cache sidecar(s): {sidecars}. "
            f"Restoring the source makes every one of them a valid hit forever (INV-08)."
        )

    def test_a_failed_run_leaves_no_timeline_the_report_disowns(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """INV-13: nothing survives a run that the report calls failed.

        `timeline.json` is written *before* attribution here, because the attribution cache
        key is keyed on its hash. So a run that fails during detection has already overwritten
        it, and leaving it there published a file the same report calls `reconstruct: failed`
        and does not list among its deliverable hashes. M4 and M5 read that file; a timeline
        no run vouches for is exactly the artifact INV-13 exists to prevent.

        Started from a *valid* timeline rather than an empty directory, so the test
        distinguishes "removed it" from "never got that far".
        """
        session_dir = canonical_fixture.session_dir
        timeline_path = session_dir / TIMELINE_RELATIVE_PATH
        assert run_activity(session_dir, detector=leaky(canonical_fixture)).exit_code is ExitCode.OK
        assert timeline_path.exists(), "precondition: a good run wrote one"

        class Exploding(ScriptedActivityDetector):
            def detect(self, window: AudioWindow) -> tuple[SpeechSpan, ...]:
                message = "the detector fell over"
                raise RuntimeError(message)

        detector = Exploding({})
        result = run_activity(
            session_dir,
            detector=DetectorBundle(identity=detector.identity(), make=lambda _t: detector),
        )

        assert result.exit_code is not ExitCode.OK
        assert not timeline_path.exists(), (
            "a failed run left a timeline behind that its own report calls failed and does "
            "not hash as a deliverable"
        )
        produced = {item.relative_path for item in result.report.provenance.deliverables}
        assert TIMELINE_RELATIVE_PATH not in produced

    def test_a_report_that_would_land_inside_raw_is_not_written(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """INV-01 outranks INV-13 here: a report is regenerable, a source directory is not."""
        session_dir = canonical_fixture.session_dir
        (session_dir / OUTPUT_DIRNAME).mkdir(parents=True, exist_ok=True)
        (session_dir / OUTPUT_DIRNAME).rmdir()
        (session_dir / OUTPUT_DIRNAME).symlink_to(session_dir / "raw" / "tx-a")

        result = run_activity(session_dir, detector=leaky(canonical_fixture))

        assert result.exit_code is ExitCode.FATAL
        assert result.report_written is False
        assert not (session_dir / "raw" / "tx-a" / "ingest-report.json").exists()

    def test_a_failure_before_inspection_still_reports_every_stage(
        self, tmp_path: Path, instant: dt.datetime
    ) -> None:
        """A run that dies at configuration load has no stages of its own to report.

        `build()` refuses a report with a gap in it, so without the fallback this produced no
        report at all — which is precisely what INV-13 exists to prevent.
        """
        empty = tmp_path / "not-a-session"
        empty.mkdir()
        result = run_activity(empty, now=instant)

        assert result.exit_code is not ExitCode.OK
        assert {stage.stage for stage in result.report.stages} == set(StageName)
        assert result.report_path.exists()

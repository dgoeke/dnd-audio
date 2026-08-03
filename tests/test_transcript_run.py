"""`transcribe` and `render`, end to end on the canonical fixture.

The gate's own summary: an end-to-end transcript from synthetic input with no Qwen, output
that validates against the checked-in schema, byte-stable Markdown and JSON on a rerun, and a
`render` that regenerates both from records alone. Everything here runs the real composition —
the same function the CLI calls — rather than a test-only arrangement of the parts.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest

from dnd_audio.artifacts.records import TranscriptRecords
from dnd_audio.artifacts.report import OverallStatus, StageName, StageStatus
from dnd_audio.determinism import sha256_file
from dnd_audio.errors import ExitCode
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.interfaces import TranscriptionResult
from dnd_audio.transcript import (
    ASR_DIRNAME,
    RECORDS_RELATIVE_PATH,
    TRANSCRIPT_JSON_RELATIVE_PATH,
    TRANSCRIPT_MARKDOWN_RELATIVE_PATH,
)
from dnd_audio.transcript.runner import TranscriberBundle, run_render, run_transcribe


@pytest.fixture
def transcribed(canonical_fixture: FixtureTruth) -> Any:
    """The canonical fixture, transcribed from its own declared script."""
    return run_transcribe(canonical_fixture.session_dir, fake_models=True), canonical_fixture


class TestTheCanonicalSession:
    def test_it_transcribes_end_to_end_with_no_model(self, transcribed: Any) -> None:
        result, _ = transcribed
        assert result.exit_code is ExitCode.OK
        assert result.records is not None
        assert result.report.overall_status is OverallStatus.COMPLETE

    def test_every_speaker_who_spoke_is_in_the_transcript(self, transcribed: Any) -> None:
        """The fixture declares four utterances on four tracks, before any audio exists."""
        result, truth = transcribed
        spoken = {interval.track_id for interval in truth.speech}
        assert {segment.track_id for segment in result.records.retained()} == spoken

    def test_the_text_is_the_text_the_fixture_declared(self, transcribed: Any) -> None:
        result, truth = transcribed
        declared = {interval.text for interval in truth.speech}
        assert {segment.text for segment in result.records.retained()} == declared

    def test_the_two_simultaneous_speakers_are_marked_as_overlap(self, transcribed: Any) -> None:
        """`tx-d` and `tx-e` at 6.8–7.8 s: the fixture's declared two-person overlap."""
        result, _ = transcribed
        overlapping = {s.track_id for s in result.records.retained() if s.overlap}
        assert overlapping == {"tx-d", "tx-e"}

    def test_bleed_never_reaches_the_transcript(self, transcribed: Any) -> None:
        """`tx-a`'s solo bleeds into four tracks and the fake ASR is scripted to transcribe
        it there. Every copy is gone before a word is submitted, because M3's gate suppressed
        the candidate — which is what "transcribe retained segments" buys."""
        result, _ = transcribed
        alice = "We should go back to Zephyrine."
        carriers = [s for s in result.records.retained() if s.text == alice]
        assert [segment.track_id for segment in carriers] == ["tx-a"]

    def test_post_gap_speech_lands_after_the_gap(self, transcribed: Any) -> None:
        """`tx-c` stops at 5.0 s and speaks at 8.5 s; a bug that slides audio earlier shows
        up here as a segment in the wrong place."""
        result, _ = transcribed
        carol = next(s for s in result.records.retained() if s.track_id == "tx-c")
        assert carol.start_sample >= 408_000

    def test_the_transcript_validates_against_the_checked_in_schema(
        self, transcribed: Any, repo_root: Path
    ) -> None:
        _, truth = transcribed
        schema = json.loads(
            (repo_root / "schemas" / "transcript.schema.json").read_text(encoding="utf-8")
        )
        document = json.loads(
            (truth.session_dir / TRANSCRIPT_JSON_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        jsonschema.validate(document, schema)

    def test_the_records_validate_against_the_checked_in_schema(
        self, transcribed: Any, repo_root: Path
    ) -> None:
        _, truth = transcribed
        schema = json.loads(
            (repo_root / "schemas" / "transcript-records.schema.json").read_text(encoding="utf-8")
        )
        document = json.loads(
            (truth.session_dir / RECORDS_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        jsonschema.validate(document, schema)

    def test_the_markdown_is_the_specs_format(self, transcribed: Any) -> None:
        _, truth = transcribed
        rendered = (truth.session_dir / TRANSCRIPT_MARKDOWN_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
        assert rendered.startswith("# Session 01\n\n")
        assert "**[00:00:05.200] Alice:** We should go back to Zephyrine.\n" in rendered
        assert "**[00:00:06.800] Dan [overlap]:** Absolutely not.\n" in rendered


class TestTheReport:
    def test_five_stages_complete_and_the_mix_is_skipped(self, transcribed: Any) -> None:
        result, _ = transcribed
        statuses = {stage.stage: stage.status for stage in result.report.stages}
        assert statuses[StageName.INSPECT] is StageStatus.COMPLETE
        assert statuses[StageName.RECONSTRUCT] is StageStatus.COMPLETE
        assert statuses[StageName.ACTIVITY] is StageStatus.COMPLETE
        assert statuses[StageName.TRANSCRIBE] is StageStatus.COMPLETE
        assert statuses[StageName.RENDER] is StageStatus.COMPLETE
        assert statuses[StageName.MIX] is StageStatus.SKIPPED

    def test_provenance_carries_the_transcriber_identity(self, transcribed: Any) -> None:
        """The spec: "Model versions and package versions must appear in the report"."""
        result, _ = transcribed
        identity = result.report.provenance.model_identity
        assert identity["asr"] == "session-script"
        assert identity["asr_max_new_tokens"] == "1024"
        assert identity["asr_language"] == "English"
        assert len(identity["asr_variant"]) == 64

    def test_every_deliverable_is_hashed(self, transcribed: Any) -> None:
        result, _ = transcribed
        produced = {item.relative_path for item in result.report.provenance.deliverables}
        assert produced == {
            "work/manifest.json",
            "work/timeline.json",
            "work/activity.json",
            RECORDS_RELATIVE_PATH,
            TRANSCRIPT_JSON_RELATIVE_PATH,
            TRANSCRIPT_MARKDOWN_RELATIVE_PATH,
        }

    def test_the_fake_models_warning_reaches_the_report(self, transcribed: Any) -> None:
        result, _ = transcribed
        codes = {note.code for stage in result.report.stages for note in stage.warnings}
        assert "fake_models_in_use" in codes


class TestRerun:
    def test_both_deliverables_and_the_records_are_byte_stable(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        session_dir = canonical_fixture.session_dir
        first = run_transcribe(session_dir, fake_models=True)
        before = {
            path: sha256_file(session_dir / path)
            for path in (
                RECORDS_RELATIVE_PATH,
                TRANSCRIPT_JSON_RELATIVE_PATH,
                TRANSCRIPT_MARKDOWN_RELATIVE_PATH,
            )
        }
        second = run_transcribe(session_dir, fake_models=True)
        after = {path: sha256_file(session_dir / path) for path in before}
        assert first.exit_code is second.exit_code is ExitCode.OK
        assert after == before

    def test_the_second_run_is_served_from_the_asr_cache(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        session_dir = canonical_fixture.session_dir
        run_transcribe(session_dir, fake_models=True)
        second = run_transcribe(session_dir, fake_models=True)
        assert second.report.telemetry.cache_misses == 0
        assert second.report.telemetry.cache_hits > 0

    def test_no_cache_re_runs_without_changing_the_answer(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        session_dir = canonical_fixture.session_dir
        warm = run_transcribe(session_dir, fake_models=True)
        cold = run_transcribe(session_dir, fake_models=True, use_cache=False)
        assert cold.records == warm.records


class TestRender:
    def test_it_regenerates_both_outputs_from_records_alone(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Run with the graph, the timeline and every cache deleted: if `render` needed any
        of them this would fail rather than quietly re-deriving them."""
        session_dir = canonical_fixture.session_dir
        run_transcribe(session_dir, fake_models=True)
        expected = {
            path: (session_dir / path).read_bytes()
            for path in (TRANSCRIPT_JSON_RELATIVE_PATH, TRANSCRIPT_MARKDOWN_RELATIVE_PATH)
        }
        for path in (TRANSCRIPT_JSON_RELATIVE_PATH, TRANSCRIPT_MARKDOWN_RELATIVE_PATH):
            (session_dir / path).unlink()
        (session_dir / "work" / "activity.json").unlink()
        (session_dir / "work" / "timeline.json").unlink()
        shutil.rmtree(session_dir / "work" / "cache")

        result = run_render(session_dir)
        assert result.exit_code is ExitCode.OK
        assert {path: (session_dir / path).read_bytes() for path in expected} == expected

    def test_it_loads_no_model(self, canonical_fixture: FixtureTruth, monkeypatch: Any) -> None:
        """A spy on the two seams, so "no model" is checked rather than assumed."""
        session_dir = canonical_fixture.session_dir
        run_transcribe(session_dir, fake_models=True)

        import dnd_audio.transcript.fakemodels as fakemodels

        def refuse(*_args: Any, **_kwargs: Any) -> Any:
            message = "render loaded a model"
            raise AssertionError(message)

        monkeypatch.setattr(fakemodels, "load_fake_models", refuse)
        assert run_render(session_dir).exit_code is ExitCode.OK

    def test_absent_records_fail_clearly_and_still_write_a_report(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The spec asks for exactly this: "fail clearly if the required transcript records
        do not exist"."""
        result = run_render(canonical_fixture.session_dir)
        assert result.exit_code is ExitCode.FATAL
        errors = [error for stage in result.report.stages for error in stage.errors]
        assert [error.code for error in errors] == ["transcript_records_missing"]
        assert result.report_path.exists()

    def test_unreadable_records_fail_clearly(self, canonical_fixture: FixtureTruth) -> None:
        session_dir = canonical_fixture.session_dir
        run_transcribe(session_dir, fake_models=True)
        (session_dir / RECORDS_RELATIVE_PATH).write_text("{not json", encoding="utf-8")
        result = run_render(session_dir)
        assert result.exit_code is ExitCode.FATAL
        codes = {error.code for stage in result.report.stages for error in stage.errors}
        assert codes == {"transcript_records_unreadable"}

    def test_records_from_another_configuration_are_rendered_with_a_warning(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """They say which configuration they were made under, so this is detectable."""
        session_dir = canonical_fixture.session_dir
        run_transcribe(session_dir, fake_models=True)
        document = json.loads((session_dir / RECORDS_RELATIVE_PATH).read_text(encoding="utf-8"))
        document["config_hash"] = "0" * 64
        (session_dir / RECORDS_RELATIVE_PATH).write_text(json.dumps(document), encoding="utf-8")

        result = run_render(session_dir)
        assert result.exit_code is ExitCode.OK
        codes = {note.code for stage in result.report.stages for note in stage.warnings}
        assert "transcript_records_stale" in codes


class TestTheCapHoldsOnARealSession:
    def test_no_submitted_waveform_exceeds_max_segment_s(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Asserted over what the transcriber was actually asked, not over the plan."""
        session_dir = canonical_fixture.session_dir
        recorded: list[int] = []

        class Watching:
            def transcribe(self, request: Any) -> TranscriptionResult:
                recorded.append(len(request.audio))
                return TranscriptionResult(request_id=request.request_id, text="something")

        from dnd_audio.transcript.fakemodels import load_fake_models

        fake = load_fake_models(session_dir)
        result = run_transcribe(
            session_dir,
            detector=fake.detector,
            transcriber=TranscriberBundle(
                transcriber=Watching(), name="watching", variant_digest="a" * 64
            ),
        )
        assert result.exit_code is ExitCode.OK
        assert recorded
        assert max(recorded) <= 120 * 16_000


class TestFailureBehaviour:
    def test_without_the_asr_runtime_the_run_fails_with_a_report(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """`transcribe` has only one branch, so here a model failure *is* the run failing.

        Unlike `process`, which must still produce the MP3, this command exists to produce
        a transcript. What INV-13 requires is that it fail visibly: a failed stage, a
        structured error, a written report, and a nonzero exit — never a traceback.
        """
        result = run_transcribe(canonical_fixture.session_dir)

        assert result.exit_code is not ExitCode.OK
        assert result.records is None
        assert result.report_written
        assert result.report_path.is_file()

    def test_that_failure_names_the_group_and_the_way_around_it(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Two actionable routes: install the runtime, or transcribe a synthetic session
        from its own declared script. An operator hitting this needs to be told both."""
        result = run_transcribe(canonical_fixture.session_dir)
        errors = " ".join(error.message for stage in result.report.stages for error in stage.errors)

        assert "asr-qwen" in errors
        assert "--fake-models" in errors

    def test_the_transcript_stage_is_the_one_marked_failed(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Model resolution happens before the snapshot is acted on, so the stages upstream
        of it never ran — but the report must still account for every one of them (INV-13)."""
        result = run_transcribe(canonical_fixture.session_dir)
        stages = {stage.stage: stage.status for stage in result.report.stages}

        assert stages[StageName.TRANSCRIBE] == "failed"
        assert set(stages) == set(StageName)

    def test_a_missing_fake_models_file_is_fatal_rather_than_a_fallback(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        session_dir = canonical_fixture.session_dir
        (session_dir / "fake-models.json").unlink()
        result = run_transcribe(session_dir, fake_models=True)
        assert result.exit_code is ExitCode.FATAL
        assert result.records is None

    def test_a_failed_run_leaves_no_stale_transcript(self, canonical_fixture: FixtureTruth) -> None:
        """A stale deliverable beside a report calling the stage failed is worse than none:
        the file looks current and nothing in it says otherwise."""
        session_dir = canonical_fixture.session_dir
        run_transcribe(session_dir, fake_models=True)
        assert (session_dir / TRANSCRIPT_JSON_RELATIVE_PATH).exists()

        (session_dir / "fake-models.json").unlink()
        result = run_transcribe(session_dir, fake_models=True)
        assert result.exit_code is ExitCode.FATAL
        for path in (
            RECORDS_RELATIVE_PATH,
            TRANSCRIPT_JSON_RELATIVE_PATH,
            TRANSCRIPT_MARKDOWN_RELATIVE_PATH,
        ):
            assert not (session_dir / path).exists()

    def test_a_failure_after_a_commit_point_leaves_the_later_cache_uncommitted(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """INV-08, scoped to what the composed run actually promises (ADR-0021).

        The invariant's original wording — "a failed run leaves no sidecar anywhere under
        `work/cache`" — describes a single-commit run and is not what this one does. Two
        commit points mean a failure during ASR keeps the activity caches, which were built
        from bytes *this run verified* before publishing them; what must not survive is a
        sidecar for a cache whose verification never happened. The name says that now,
        because the previous one said the opposite of what the body asserts (M4's verify
        phase, found by independent review)."""
        session_dir = canonical_fixture.session_dir
        source = next((session_dir / "raw" / "tx-a").glob("*.wav"))
        original = source.read_bytes()

        class Corrupting:
            """Mutates a source once ASR has already read the derivative it was built from."""

            def transcribe(self, request: Any) -> TranscriptionResult:
                source.write_bytes(original[:-4000])
                return TranscriptionResult(request_id=request.request_id, text="x")

        from dnd_audio.transcript.fakemodels import load_fake_models

        fake = load_fake_models(session_dir)
        result = run_transcribe(
            session_dir,
            detector=fake.detector,
            transcriber=TranscriberBundle(
                transcriber=Corrupting(), name="corrupting", variant_digest="b" * 64
            ),
        )
        # `partial`, not `failed`: inspection, the timeline and the graph all genuinely
        # completed, and their deliverables are still on disk to be hashed — asserted rather
        # than claimed, because the claim was false until M4's verify phase. INV-13 only
        # requires that a partial run never exits zero, and ADR-0005 spends exit 4 on saying
        # which it was.
        assert result.exit_code is ExitCode.PARTIAL
        assert (session_dir / "work" / "timeline.json").exists()
        assert (session_dir / "work" / "activity.json").exists()
        assert int(result.exit_code) != 0
        # Every raw response the run wrote is present and **inert**: the sidecar that would
        # make it findable was never committed, so nothing can ever be served from it. The
        # data file is the ASR cache's equivalent of the detection cache's `.probs`.
        raw = list((session_dir / ASR_DIRNAME).glob("*.raw.json"))
        assert raw
        for document in raw:
            sidecar = document.with_name(document.name.removesuffix(".raw.json") + ".json")
            assert not sidecar.exists(), sidecar
        # The activity caches *are* committed, and that is the point of two commit points:
        # they were built from bytes this run verified, and the corruption happened after.
        assert list((session_dir / "work" / "cache" / "activity").rglob("*.json"))


class TestAlignmentFailureNeverFailsTheSession:
    def test_the_segment_survives_the_run_warns_and_the_exit_is_clean(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The spec: "retain the segment-level transcript and emit a warning rather than
        failing the entire session"."""
        session_dir = canonical_fixture.session_dir

        class NeverAligns:
            def transcribe(self, request: Any) -> TranscriptionResult:
                return TranscriptionResult(
                    request_id=request.request_id,
                    text="the words are here and their times are not",
                    alignment_status="segment_only",
                )

        from dnd_audio.transcript.fakemodels import load_fake_models

        fake = load_fake_models(session_dir)
        result = run_transcribe(
            session_dir,
            detector=fake.detector,
            transcriber=TranscriberBundle(
                transcriber=NeverAligns(), name="never-aligns", variant_digest="e" * 64
            ),
        )

        assert result.exit_code is ExitCode.OK
        assert result.records is not None
        retained = result.records.retained()
        assert retained
        assert {segment.alignment_status for segment in retained} == {"segment_only"}
        assert all(segment.words == [] for segment in retained)
        assert all(segment.text for segment in retained)
        codes = {note.code for stage in result.report.stages for note in stage.warnings}
        assert "alignment_failed" in codes

        document = json.loads(
            (session_dir / TRANSCRIPT_JSON_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        assert {s["provenance"]["alignment_status"] for s in document["segments"]} == {
            "segment_only"
        }


class TestInv09:
    def test_nothing_under_activity_imports_the_transcript_package(self, repo_root: Path) -> None:
        """Structural, because the invariant is about direction: the graph is model-
        independent and the transcript branch is downstream of it, forever."""
        for path in (repo_root / "src" / "dnd_audio" / "activity").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "dnd_audio.transcript" not in text, path

    def test_the_graph_is_unchanged_by_the_transcript_branch(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        session_dir = canonical_fixture.session_dir
        run_transcribe(session_dir, fake_models=True)
        graph = session_dir / "work" / "activity.json"
        before = sha256_file(graph)

        run_transcribe(session_dir, fake_models=True)
        assert sha256_file(graph) == before

    def test_a_write_into_the_graph_fails_the_run(self, canonical_fixture: FixtureTruth) -> None:
        """The check exists to catch a write from anywhere, so prove it can fire.

        A verification that is present, looks right, and cannot fail is the shape this
        project keeps finding (M1's closeout, M2's INV-01 hole). This one is driven by a
        transcriber that writes into the graph while ASR is running.
        """
        session_dir = canonical_fixture.session_dir
        graph_path = session_dir / "work" / "activity.json"

        class Meddling:
            def transcribe(self, request: Any) -> TranscriptionResult:
                graph_path.write_text(
                    graph_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
                )
                return TranscriptionResult(request_id=request.request_id, text="x")

        from dnd_audio.transcript.fakemodels import load_fake_models

        fake = load_fake_models(session_dir)
        result = run_transcribe(
            session_dir,
            detector=fake.detector,
            transcriber=TranscriberBundle(
                transcriber=Meddling(), name="meddling", variant_digest="f" * 64
            ),
        )
        codes = {error.code for stage in result.report.stages for error in stage.errors}
        assert codes == {"activity_graph_modified"}
        assert result.exit_code is not ExitCode.OK

    def test_no_asr_text_reaches_the_graph(self, canonical_fixture: FixtureTruth) -> None:
        """The hazard M3's review deferred: `ActivityDecision.detail` and `ActivityNote
        .message` are unrestricted strings on the field allowlist, so nothing structural
        stops text-derived content being written into them."""
        session_dir = canonical_fixture.session_dir
        result = run_transcribe(session_dir, fake_models=True)
        assert result.records is not None
        graph = json.loads((session_dir / "work" / "activity.json").read_text(encoding="utf-8"))
        prose = " ".join(
            [
                *(note["message"] for note in graph["warnings"]),
                *(decision["detail"] for decision in graph["decisions"]),
            ]
        )
        for segment in result.records.retained():
            assert segment.text not in prose


class TestBoundedMemory:
    def test_a_transcription_happens_before_the_last_read(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """INV-07 over the composed path, with M2's technique: one ordered event log, and a
        submission before the last read. Nothing that builds every request's audio up front
        can satisfy that — which is exactly what a planner returning `TranscriptionRequest`
        objects would have done."""
        session_dir = canonical_fixture.session_dir
        events: list[str] = []

        from dnd_audio.timeline.reader import DerivativeReader
        from dnd_audio.transcript.fakemodels import load_fake_models

        original = DerivativeReader.read

        def watched(self: Any, track_id: str, start: int, n: int) -> Any:
            events.append("read")
            return original(self, track_id, start, n)

        class Watching:
            def transcribe(self, request: Any) -> TranscriptionResult:
                events.append("transcribe")
                return TranscriptionResult(request_id=request.request_id, text="x")

        fake = load_fake_models(session_dir)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(DerivativeReader, "read", watched)
            result = run_transcribe(
                session_dir,
                detector=fake.detector,
                transcriber=TranscriberBundle(
                    transcriber=Watching(), name="watching", variant_digest="c" * 64
                ),
            )

        assert result.exit_code is ExitCode.OK
        assert "transcribe" in events
        assert events.index("transcribe") < len(events) - 1 - events[::-1].index("read")

    def test_no_request_holds_more_than_the_cap(self, canonical_fixture: FixtureTruth) -> None:
        """The other half of the bound: one request in memory, and it is bounded."""
        session_dir = canonical_fixture.session_dir
        largest = 0

        class Watching:
            def transcribe(self, request: Any) -> TranscriptionResult:
                nonlocal largest
                largest = max(largest, int(np.asarray(request.audio.samples).nbytes))
                return TranscriptionResult(request_id=request.request_id, text="x")

        from dnd_audio.transcript.fakemodels import load_fake_models

        fake = load_fake_models(session_dir)
        run_transcribe(
            session_dir,
            detector=fake.detector,
            transcriber=TranscriberBundle(
                transcriber=Watching(), name="watching", variant_digest="d" * 64
            ),
        )
        assert 0 < largest <= 120 * 16_000 * 4


def test_the_records_declare_the_graph_and_configuration_they_describe(
    canonical_fixture: FixtureTruth,
) -> None:
    session_dir = canonical_fixture.session_dir
    result = run_transcribe(session_dir, fake_models=True)
    graph = json.loads((session_dir / "work" / "activity.json").read_text(encoding="utf-8"))
    records = TranscriptRecords.model_validate_json(
        (session_dir / RECORDS_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    assert records.activity_cache_key == graph["attribution_cache_key"]
    assert records.timeline_sha256 == sha256_file(session_dir / "work" / "timeline.json")
    assert records.config_hash == result.report.provenance.config_hash


class TestAPartialRunReportsOnlyWhatSurvived:
    """INV-13: "hashes of every deliverable actually produced".

    The composed run commits at two points, so an ASR failure leaves reconstruction and
    attribution genuinely complete — and already hashed as deliverables. Cleanup used to
    delete both anyway, so the report advertised the hash of a file that was gone. Either
    behaviour is defensible on its own; the two together are not, and a report naming a
    deliverable that is not there is the exact failure INV-13 exists to prevent.
    """

    @staticmethod
    def _failed_during_asr(session_dir: Path) -> Any:
        class Exploding:
            def transcribe(self, request: Any) -> TranscriptionResult:
                message = "the disk went away mid-ASR"
                raise OSError(message)

        from dnd_audio.transcript.fakemodels import load_fake_models

        fake = load_fake_models(session_dir)
        return run_transcribe(
            session_dir,
            detector=fake.detector,
            transcriber=TranscriberBundle(
                transcriber=Exploding(), name="boom", variant_digest="9" * 64
            ),
        )

    def test_every_hashed_deliverable_is_still_on_disk(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        session_dir = canonical_fixture.session_dir
        result = self._failed_during_asr(session_dir)

        assert result.exit_code is ExitCode.PARTIAL
        produced = [item.relative_path for item in result.report.provenance.deliverables]
        assert produced
        for relative in produced:
            assert (session_dir / relative).exists(), (
                f"the report hashes {relative}, which is not on disk. A deliverable a run "
                f"advertises must be one it actually produced (INV-13)."
            )

    def test_the_artifacts_of_a_completed_stage_survive(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The other half: `activity: complete` means the graph is really there."""
        session_dir = canonical_fixture.session_dir
        result = self._failed_during_asr(session_dir)

        statuses = {stage.stage: stage.status for stage in result.report.stages}
        assert statuses[StageName.RECONSTRUCT] is StageStatus.COMPLETE
        assert statuses[StageName.ACTIVITY] is StageStatus.COMPLETE
        assert (session_dir / "work" / "timeline.json").exists()
        assert (session_dir / "work" / "activity.json").exists()

    def test_the_stages_that_failed_leave_nothing_behind(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Keeping a completed stage's artifacts must not keep a failed stage's."""
        session_dir = canonical_fixture.session_dir
        run_transcribe(session_dir, fake_models=True)
        assert (session_dir / TRANSCRIPT_JSON_RELATIVE_PATH).exists()

        result = self._failed_during_asr(session_dir)

        assert result.exit_code is ExitCode.PARTIAL
        for relative in (
            RECORDS_RELATIVE_PATH,
            TRANSCRIPT_JSON_RELATIVE_PATH,
            TRANSCRIPT_MARKDOWN_RELATIVE_PATH,
        ):
            assert not (session_dir / relative).exists(), relative

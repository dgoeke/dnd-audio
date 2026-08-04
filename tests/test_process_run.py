"""`dnd-audio process`: activity once, two branches, one report.

Four properties, and only the first is the one the spec's sentence about transcription is
usually read as asking for:

1. a transcription failure leaves the MP3 and the report, with the transcript stage failed and
   a nonzero exit;
2. a **mix** failure does not cancel the transcript branch either — independence is a property
   of the control flow, and a sequential mix-first implementation that let a mix exception
   propagate would satisfy (1) and violate this;
3. activity executes exactly once, not once per branch;
4. either branch failing still accounts for every stage.

M5's plan review is why there are four rather than one: the first draft of the plan proposed
only (1), and (2) is the one a plausible implementation gets wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from dnd_audio.activity import ACTIVITY_RELATIVE_PATH
from dnd_audio.activity.runner import perform_activity
from dnd_audio.artifacts.report import OverallStatus, StageName, StageStatus
from dnd_audio.errors import DiscoveryError, ExitCode
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.inspection import OUTPUT_DIRNAME
from dnd_audio.interfaces import TranscriptionResult
from dnd_audio.mix import MP3_RELATIVE_PATH
from dnd_audio.models import SNAPSHOT_FETCH_COMMAND
from dnd_audio.orchestrate import ProcessResult, process_outputs, run_process
from dnd_audio.transcript import (
    RECORDS_RELATIVE_PATH,
    TRANSCRIPT_JSON_RELATIVE_PATH,
    TRANSCRIPT_MARKDOWN_RELATIVE_PATH,
)
from dnd_audio.transcript.runner import TranscriberBundle, perform_transcript

REPORT = f"{OUTPUT_DIRNAME}/ingest-report.json"


def processed(fixture: FixtureTruth, **kwargs: Any) -> ProcessResult:
    return run_process(fixture.session_dir, fake_models=True, **kwargs)


def _status(result: ProcessResult) -> dict[StageName, StageStatus]:
    return {stage.stage: stage.status for stage in result.report.stages}


class Exploding:
    """A transcriber that fails the way a real model failing looks: partway through."""

    def transcribe(self, request: Any) -> TranscriptionResult:
        message = "the ASR model fell over"
        raise RuntimeError(message)


class TestBothBranchesSucceed:
    def test_it_produces_the_mp3_and_both_transcript_files(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        result = processed(canonical_fixture)
        session_dir = canonical_fixture.session_dir

        assert result.exit_code is ExitCode.OK
        assert result.report.overall_status is OverallStatus.COMPLETE
        for relative in (
            MP3_RELATIVE_PATH,
            TRANSCRIPT_JSON_RELATIVE_PATH,
            TRANSCRIPT_MARKDOWN_RELATIVE_PATH,
            RECORDS_RELATIVE_PATH,
            ACTIVITY_RELATIVE_PATH,
        ):
            assert (session_dir / relative).exists(), relative

    def test_every_stage_is_complete_and_every_deliverable_is_hashed(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        result = processed(canonical_fixture)
        assert set(_status(result).values()) == {StageStatus.COMPLETE}
        paths = {item.relative_path for item in result.report.provenance.deliverables}
        assert {
            MP3_RELATIVE_PATH,
            TRANSCRIPT_JSON_RELATIVE_PATH,
            TRANSCRIPT_MARKDOWN_RELATIVE_PATH,
            RECORDS_RELATIVE_PATH,
        } <= paths

    def test_activity_executes_exactly_once(self, canonical_fixture: FixtureTruth) -> None:
        """ "Run activity once." Two branches reading the same graph must not each build it —
        that would be twice the inference and two documents that could disagree."""
        calls = 0
        original = perform_activity

        def watched(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("dnd_audio.orchestrate.perform_activity", watched)
            result = processed(canonical_fixture)

        assert calls == 1
        assert result.exit_code is ExitCode.OK

    def test_the_mix_matches_what_mix_alone_would_have_produced(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """INV-09 through the composed command: running ASR beside the mix changes no sample."""
        from dnd_audio.determinism import sha256_file
        from dnd_audio.mix import MIX_CACHE_DIRNAME
        from dnd_audio.mix.runner import run_mix
        from dnd_audio.transcript.fakemodels import load_fake_models

        session_dir = canonical_fixture.session_dir
        detector = load_fake_models(session_dir).detector
        run_mix(session_dir, detector=detector, use_cache=False)
        alone = {
            path.name: sha256_file(path) for path in (session_dir / MIX_CACHE_DIRNAME).glob("*.wav")
        }
        assert alone

        processed(canonical_fixture, use_cache=False)
        together = {
            path.name: sha256_file(path) for path in (session_dir / MIX_CACHE_DIRNAME).glob("*.wav")
        }
        assert together == alone

    def test_an_assembly_only_setting_leaves_the_mix_intermediate_content_unchanged(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """INV-09's artifact proof, not merely a projection or selected cache identity."""
        from dnd_audio.determinism import sha256_file
        from dnd_audio.mix import MIX_CACHE_DIRNAME

        session_dir = canonical_fixture.session_dir
        first = processed(canonical_fixture, use_cache=False)
        before = {
            path.name: sha256_file(path) for path in (session_dir / MIX_CACHE_DIRNAME).glob("*.wav")
        }
        assert first.exit_code is ExitCode.OK
        assert before

        config_path = session_dir / "session.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config.setdefault("transcript", {})["leading_ownership_grace_ms"] = 40
        config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

        second = processed(canonical_fixture)
        after = {
            path.name: sha256_file(path) for path in (session_dir / MIX_CACHE_DIRNAME).glob("*.wav")
        }
        assert second.exit_code is ExitCode.OK
        assert set(after.values()) == set(before.values())


class TestATranscriptionFailureDoesNotCancelTheMix:
    """The spec's own sentence, and the gate criterion this milestone must demonstrate."""

    @pytest.fixture
    def failed(self, canonical_fixture: FixtureTruth) -> ProcessResult:
        from dnd_audio.transcript.fakemodels import load_fake_models

        fake = load_fake_models(canonical_fixture.session_dir)
        return run_process(
            canonical_fixture.session_dir,
            detector=fake.detector,
            transcriber=TranscriberBundle(
                transcriber=Exploding(), name="exploding", variant_digest="a" * 64
            ),
        )

    def test_the_mp3_is_still_produced_and_hashed(
        self, failed: ProcessResult, canonical_fixture: FixtureTruth
    ) -> None:
        assert (canonical_fixture.session_dir / MP3_RELATIVE_PATH).exists()
        paths = {item.relative_path for item in failed.report.provenance.deliverables}
        assert MP3_RELATIVE_PATH in paths
        assert failed.encode is not None

    def test_the_transcript_stage_is_failed_and_the_mix_stage_is_complete(
        self, failed: ProcessResult
    ) -> None:
        status = _status(failed)
        assert status[StageName.MIX] is StageStatus.COMPLETE
        assert status[StageName.TRANSCRIBE] is StageStatus.FAILED
        assert status[StageName.RENDER] is StageStatus.FAILED
        assert status[StageName.ACTIVITY] is StageStatus.COMPLETE

    def test_process_exits_nonzero(self, failed: ProcessResult) -> None:
        """ "So automation cannot mistake partial output for full success" (INV-13)."""
        assert failed.report.overall_status is OverallStatus.PARTIAL
        assert failed.exit_code is ExitCode.PARTIAL
        assert int(failed.exit_code) != 0

    def test_no_stale_transcript_survives(
        self, failed: ProcessResult, canonical_fixture: FixtureTruth
    ) -> None:
        """A transcript beside a report calling its stage failed is worse than none."""
        for relative in (
            RECORDS_RELATIVE_PATH,
            TRANSCRIPT_JSON_RELATIVE_PATH,
            TRANSCRIPT_MARKDOWN_RELATIVE_PATH,
        ):
            assert not (canonical_fixture.session_dir / relative).exists(), relative

    def test_the_error_names_what_went_wrong(self, failed: ProcessResult) -> None:
        messages = [error.message for stage in failed.report.stages for error in stage.errors]
        assert any("fell over" in message for message in messages)


class TestAMixFailureDoesNotCancelTheTranscript:
    """The other direction, which ordering alone does not give.

    A sequential mix-first implementation that let a mix exception propagate would satisfy
    every test above and fail every test here. Independence has to be a property of the
    control flow.
    """

    @pytest.fixture
    def failed(self, canonical_fixture: FixtureTruth) -> ProcessResult:
        def exploding(*args: Any, **kwargs: Any) -> Any:
            message = "the encoder fell over"
            raise OSError(message)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("dnd_audio.orchestrate.encode_deliverable", exploding)
            return processed(canonical_fixture)

    def test_the_transcript_is_still_produced(
        self, failed: ProcessResult, canonical_fixture: FixtureTruth
    ) -> None:
        assert failed.records is not None
        for relative in (TRANSCRIPT_JSON_RELATIVE_PATH, TRANSCRIPT_MARKDOWN_RELATIVE_PATH):
            assert (canonical_fixture.session_dir / relative).exists(), relative

    def test_the_render_ran_because_transcription_succeeded(self, failed: ProcessResult) -> None:
        """ "Render the transcript branch when transcription succeeds" — even when the other
        branch did not."""
        status = _status(failed)
        assert status[StageName.TRANSCRIBE] is StageStatus.COMPLETE
        assert status[StageName.RENDER] is StageStatus.COMPLETE

    def test_the_mix_stage_is_failed_and_leaves_no_stale_mp3(
        self, failed: ProcessResult, canonical_fixture: FixtureTruth
    ) -> None:
        assert _status(failed)[StageName.MIX] is StageStatus.FAILED
        assert not (canonical_fixture.session_dir / MP3_RELATIVE_PATH).exists()

    def test_it_still_exits_nonzero(self, failed: ProcessResult) -> None:
        assert failed.report.overall_status is OverallStatus.PARTIAL
        assert failed.exit_code is ExitCode.PARTIAL


class TestTheReportIsAlwaysFinalized:
    """ "Always finalize the structured report" — including when nothing worked."""

    def test_a_broken_session_still_writes_a_report_accounting_for_every_stage(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        session_dir = canonical_fixture.session_dir
        report_path = session_dir / REPORT
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"stale": True}), encoding="utf-8")
        (session_dir / "session.yaml").unlink()

        result = run_process(session_dir, fake_models=True)

        assert result.exit_code is not ExitCode.OK
        assert result.report_written
        assert len(result.report.stages) == len(StageName)
        assert json.loads(report_path.read_text(encoding="utf-8")) != {"stale": True}

    def test_a_source_corrupted_after_the_mix_commit_fails_the_run(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The gap the final unconditional verification exists for (ADR-0024).

        Nothing after the mix reads source audio, so without a verification at the end a
        change during ASR would go unnoticed — and INV-01's guarantee is about a complete run,
        not about each cache write. Driven from inside the transcript branch, which is the
        only place that window is reachable from.
        """
        session_dir = canonical_fixture.session_dir
        victim = session_dir / canonical_fixture.chunks[0].relative_path
        original = perform_transcript

        def corrupting(*args: Any, **kwargs: Any) -> Any:
            found = original(*args, **kwargs)
            with victim.open("r+b") as handle:
                handle.seek(0, 2)
                handle.write(b"\x00" * 16)
            return found

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("dnd_audio.orchestrate.perform_transcript", corrupting)
            result = processed(canonical_fixture)

        codes = {error.code for stage in result.report.stages for error in stage.errors}
        assert "raw_sources_modified" in codes
        assert result.exit_code is not ExitCode.OK

    def test_only_the_final_check_can_see_tampering_after_both_branches_verified(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The differentiating case, and the reason the test above is not one.

        Corrupting a source from inside `perform_transcript` is caught by the transcript
        branch's *own* `verify_unchanged`, so that test passes with the final check deleted.
        Here the corruption happens after both branches have verified and committed — writing
        the deliverables is the last thing either branch does — so the outer check at the end
        of `run_process` is the only thing left that can see it. Found by M5's independent
        review.
        """
        from dnd_audio.transcript.runner import write_transcript_deliverables

        session_dir = canonical_fixture.session_dir
        victim = session_dir / canonical_fixture.chunks[0].relative_path
        original = write_transcript_deliverables

        def corrupting(*args: Any, **kwargs: Any) -> Any:
            found = original(*args, **kwargs)
            with victim.open("r+b") as handle:
                handle.seek(0, 2)
                handle.write(b"\x00" * 16)
            return found

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("dnd_audio.orchestrate.write_transcript_deliverables", corrupting)
            result = processed(canonical_fixture)

        assert result.exit_code is not ExitCode.OK
        codes = {error.code for stage in result.report.stages for error in stage.errors}
        assert codes == {"raw_sources_modified"}

    def test_a_branch_that_diagnosed_itself_keeps_its_own_error(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """ADR-0024: the final check's error "does not replace whichever error the branch
        already reported".

        It did. `_failed` stamped the outer exception over every stage neither branch had
        recorded, so an ASR crash concurrent with unrelated tampering was reported as
        tampering and the real diagnostic was gone. Found by M5's independent review.
        """
        from dnd_audio.transcript.runner import write_transcript_deliverables

        session_dir = canonical_fixture.session_dir
        victim = session_dir / canonical_fixture.chunks[0].relative_path
        original = write_transcript_deliverables

        def corrupting(*args: Any, **kwargs: Any) -> Any:
            found = original(*args, **kwargs)
            with victim.open("r+b") as handle:
                handle.seek(0, 2)
                handle.write(b"\x00" * 16)
            return found

        def exploding(*args: Any, **kwargs: Any) -> Any:
            message = "the encoder fell over"
            raise OSError(message)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("dnd_audio.orchestrate.write_transcript_deliverables", corrupting)
            patch.setattr("dnd_audio.orchestrate.encode_deliverable", exploding)
            result = processed(canonical_fixture)

        stages = {stage.stage: stage for stage in result.report.stages}
        # The mix keeps the diagnosis it made; the transcript branch reported nothing of its
        # own, so the final check's error is the right one for it.
        assert [e.message for e in stages[StageName.MIX].errors] == ["the encoder fell over"]
        assert {e.code for e in stages[StageName.TRANSCRIBE].errors} == {"raw_sources_modified"}
        assert result.exit_code is not ExitCode.OK

    def test_unavailable_asr_models_cost_the_transcript_and_not_the_mix(
        self, session_without_asr_models: FixtureTruth
    ) -> None:
        """INV-09, in the case that stopped being hypothetical when M6b landed.

        Until M6b this raised `NotImplementedError` and stopped the whole run, which was
        right while it meant "the adapter does not exist yet" (ADR-0005). Now the adapter
        exists and what is missing is the opt-in `asr-qwen` group — an ordinary
        transcription failure, and exactly the one the invariant exists for. Model
        resolution still happens *before* any cache is written, so nothing is half-finished;
        what changed is that its failure belongs to the transcript branch rather than to the
        run.

        The absence is configured rather than ambient — `session_without_asr_models` pins a
        revision nothing has installed — so this asserts a property of the code on every
        environment instead of a property of whichever one happens to be running it.
        """
        result = run_process(session_without_asr_models.session_dir)

        assert result.exit_code is not ExitCode.OK
        assert result.records is None
        assert result.encode is not None, "the mix must survive a transcription failure"
        assert result.mp3_path.is_file()

        stages = {stage.stage: stage.status for stage in result.report.stages}
        assert stages[StageName.MIX] == "complete"
        assert stages[StageName.TRANSCRIBE] == "failed"

    def test_a_missing_fake_models_file_never_falls_back_to_the_real_detector(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """`--fake-models` is an assertion about what ran, and it is not softenable.

        `transcribe` has refused this since M4; `process` did not, and the hole opened when
        M6b taught `_resolve_or_defer` to turn a model failure into the transcript branch's
        error so the mix could survive it (INV-09). That is right for a host with no ASR
        weights and wrong here: under `--fake-models` this one call builds *both* seams from
        `fake-models.json`, so a file that will not load is a detector failure too. Caught,
        the run continued with `models=None` and the caller's `detector` — `None` from the
        CLI — and activity therefore built the **real** Silero detector. An operator who
        explicitly asked for fake models would have received a real MP3 and a real activity
        graph off real detection, with only a failed transcript stage as a hint, on any host
        where the VAD model happens to be installed. Found by M6b's code review.
        """
        session_dir = canonical_fixture.session_dir
        (session_dir / "fake-models.json").unlink()

        result = run_process(session_dir, fake_models=True)

        assert result.exit_code is not ExitCode.OK
        assert result.records is None
        assert result.encode is None, "no mix may be produced from a detector nobody asked for"
        assert not result.mp3_path.exists()
        stages = {stage.stage: stage.status for stage in result.report.stages}
        assert stages[StageName.ACTIVITY] != "complete"

    def test_a_discovery_failure_while_resolving_models_is_never_softened(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The other half of `_resolve_or_defer`'s rule, and it had no test until a mutation
        run showed that removing it broke nothing.

        Softening an ASR failure into the transcript branch's error is what INV-09 asks for.
        Softening a *discovery* failure is not: `output_inside_raw` is INV-01's fatality, and
        a run that continued past it would go on to write outputs into `raw/`. Resolving
        models never walks the session's paths, so this cannot arise today — which is exactly
        why it is asserted rather than trusted: the clause protects against a future
        `resolve_models` that does, and an untested guard against a future change is a guard
        that will be deleted by the person making it.

        `ModelError` is a sibling of `DiscoveryError` rather than a subclass, so the
        ordinary missing-weights path above is unaffected — which the test beside this one
        continues to prove.
        """
        raised = DiscoveryError("the output directory is inside raw/", code="output_inside_raw")

        def refusing(*args: Any, **kwargs: Any) -> Any:
            raise raised

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("dnd_audio.orchestrate.resolve_models", refusing)
            result = run_process(canonical_fixture.session_dir)

        # Fatal for the whole run, not partial — and still a written report rather than a
        # traceback, because INV-13 is not suspended by INV-01 being the thing that failed.
        assert result.exit_code is ExitCode.FATAL
        assert result.report.overall_status is OverallStatus.FAILED
        assert result.encode is None
        assert not (canonical_fixture.session_dir / MP3_RELATIVE_PATH).exists()
        codes = {error.code for stage in result.report.stages for error in stage.errors}
        assert "output_inside_raw" in codes

    def test_that_failure_names_what_to_do_about_it(
        self, session_without_asr_models: FixtureTruth
    ) -> None:
        """A structured error nobody can act on is a worse artifact than none."""
        result = run_process(session_without_asr_models.session_dir)
        errors = " ".join(error.message for stage in result.report.stages for error in stage.errors)
        assert SNAPSHOT_FETCH_COMMAND in errors


class TestTheOutputSet:
    def test_it_is_the_union_of_both_branches(self, tmp_path: Path) -> None:
        """One snapshot covers both branches, so one output list has to as well (INV-01)."""
        outputs = process_outputs(tmp_path)
        assert outputs["the mixed MP3"] == tmp_path / MP3_RELATIVE_PATH
        assert outputs["the transcript"] == tmp_path / TRANSCRIPT_JSON_RELATIVE_PATH
        assert outputs["the activity graph"] == tmp_path / ACTIVITY_RELATIVE_PATH
        assert outputs["the ASR cache"].name == "asr"

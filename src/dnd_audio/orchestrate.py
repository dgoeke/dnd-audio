"""`dnd-audio process`: one snapshot, two branches that fail independently, one report.

The spec defines this command precisely, and every clause is a property something here has to
have rather than a description of what usually happens:

> Dependency-aware orchestration of all applicable stages. Run activity once, attempt both
> downstream branches independently, render the transcript branch when transcription succeeds,
> and always finalize the structured report. A failed transcription branch must not cancel or
> skip the mix branch.

ADR-0024 records the shape. Three things about it are easy to get subtly wrong:

**Independence is a property of the control flow, not of the ordering.** Each branch runs in
its own handler and a failure in either records an error and continues to the other. Running
the mix first satisfies every sentence the spec writes about *transcription* failing while
still letting a mix exception abort the run — so the ordering is not the mechanism, and four
tests say so.

**The mix branch goes first anyway**, because it makes "the mix cannot have consumed anything
the transcript branch produced" true by construction as well as by test (INV-09), and because
it is the branch the spec says must survive.

**Recording is deferred until after the final verification.** With four verification points, a
transcript failure before the ASR commit otherwise leaves the window after the mix's *source*
reads unchecked — and INV-01's guarantee is about a complete run, not about each cache write.
A stage marked complete before that check would have to be un-marked afterwards, so nothing is
marked until the check has passed.

**An unusable ASR runtime is a branch failure, and M6b is where that changed.** Until the
adapter existed, a run without `--fake-models` raised before any work and stopped everything:
"this pipeline has not built that yet" is not "your session is broken" (ADR-0005), and a
half-finished run would have been a third and worse answer. The adapter exists now, so the
same situation means something else — the weights are absent, the device is unavailable, or
the opt-in `asr-qwen` group is not installed — and that is an ordinary transcription failure,
which is exactly what INV-09 says must still yield `session.mp3`.

Both halves are kept. Models are still resolved **before any work**, so a model that will not
load costs nothing and leaves nothing behind; but its failure is recorded as the transcript
branch's rather than the run's, and the mix branch proceeds.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dnd_audio.activity import ACTIVITY_RELATIVE_PATH
from dnd_audio.activity.runner import (
    DEFAULT_DETECT_WINDOW,
    ActivityWork,
    DetectorBundle,
    perform_activity,
    remove_activity_artifacts,
)
from dnd_audio.artifacts.activity import ActivityGraph
from dnd_audio.artifacts.records import TranscriptRecords
from dnd_audio.artifacts.report import (
    REPORT_FILENAME,
    IngestReport,
    ReportBuilder,
    ReportWarning,
    StageName,
    StructuredError,
)
from dnd_audio.config import SessionConfig, config_hash, load_session_config
from dnd_audio.determinism import sha256_file, write_json_atomic
from dnd_audio.errors import DiscoveryError, DndAudioError, ExitCode
from dnd_audio.inspection import OUTPUT_DIRNAME
from dnd_audio.mix.encode import EncodeResult
from dnd_audio.mix.runner import (
    MixWork,
    encode_deliverable,
    mix_outputs,
    perform_mix,
    record_mix_stage,
    remove_mix_artifacts,
)
from dnd_audio.raw_guard import (
    RawSnapshot,
    raw_roots,
    reject_outputs_inside_raw,
    snapshot,
    verify_unchanged,
)
from dnd_audio.timeline import TIMELINE_RELATIVE_PATH
from dnd_audio.timeline.reader import DEFAULT_WINDOW_SAMPLES
from dnd_audio.transcript import (
    RECORDS_RELATIVE_PATH,
    TRANSCRIPT_JSON_RELATIVE_PATH,
    TRANSCRIPT_MARKDOWN_RELATIVE_PATH,
)
from dnd_audio.transcript.runner import (
    Models,
    TranscriberBundle,
    perform_transcript,
    record_render_stage,
    resolve_models,
    transcribe_outputs,
    write_transcript_deliverables,
)

__all__ = ["ProcessResult", "process_outputs", "run_process"]


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """What one `process` run produced. Either branch may be ``None``."""

    graph: ActivityGraph | None
    encode: EncodeResult | None
    records: TranscriptRecords | None
    mp3_path: Path
    report: IngestReport
    report_path: Path
    exit_code: ExitCode
    #: False only when writing the report would itself have violated INV-01.
    report_written: bool = True


def process_outputs(session_dir: Path) -> dict[str, Path]:
    """Everything `process` writes: the union of both branches' outputs (INV-01)."""
    return {**mix_outputs(session_dir), **transcribe_outputs(session_dir)}


def run_process(
    session_dir: Path,
    *,
    transcriber: TranscriberBundle | None = None,
    detector: DetectorBundle | None = None,
    fake_models: bool = False,
    now: dt.datetime | None = None,
    use_cache: bool = True,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
    detect_window_samples: int | None = None,
) -> ProcessResult:
    """Run every applicable stage, attempt both branches, and always finalize the report.

    Never raises for an expected failure: a fatal condition becomes a failed stage, a
    structured error, a written report, and a nonzero exit code (INV-13). That now includes an
    unusable ASR runtime, which costs the transcript branch and not the mix — see the module
    docstring.
    """
    started_at = now or dt.datetime.now(dt.UTC)
    paths = _Paths(session_dir)
    builder = ReportBuilder(session_id=session_dir.name, config_hash=None, started_at=started_at)
    state = _State()

    try:
        config = load_session_config(session_dir / "session.yaml")
        builder = ReportBuilder(
            session_id=config.session_id,
            config_hash=config_hash(config),
            started_at=started_at,
        )
        roots = raw_roots(config)
        before = snapshot(session_dir, roots)
        reject_outputs_inside_raw(session_dir, config, roots, process_outputs(session_dir))

        # Resolved before anything is written, so a model that will not load fails before
        # the first cache is written rather than partway through six tracks of inference.
        #
        # **Its failure belongs to the transcript branch, not to the run.** Until M6b this
        # raised `NotImplementedError` and stopped everything, which was right when it meant
        # "the adapter does not exist yet". Now it means the weights are absent, the device
        # is unavailable, or the runtime will not load — which is precisely the transcription
        # failure INV-09 says must still yield `session.mp3`. Letting it kill the run would
        # violate the invariant in the one case it exists for, and on every machine without
        # the opt-in group rather than hypothetically (found by M6b's own CLI test).
        models, model_error = _resolve_or_defer(
            session_dir, config, transcriber, detector, fake_models=fake_models
        )

        work = perform_activity(
            session_dir,
            config,
            builder=builder,
            # The caller's detector when the ASR seam failed to resolve. That is not a
            # fallback: on the non-fake path `resolve_models` passes this argument through
            # untouched, so it is the same value either way — and the *activity* stage has
            # to run regardless, because the mix consumes its graph. A fake-models file that
            # cannot be read fails both seams and does stop the run, which is right: there
            # is no graph then, and a mix of silence is worse than a failure.
            detector=detector if models is None else models.detector,
            use_cache=use_cache,
            mix=True,
            window_samples=window_samples,
            detect_window_samples=detect_window_samples or DEFAULT_DETECT_WINDOW,
        )

        verify_unchanged(session_dir, roots, before)
        work.commit()
        write_json_atomic(paths.graph, work.graph.model_dump(mode="json"))
        state.graph = work.graph
        graph_sha256 = sha256_file(paths.graph)
        builder.stage_complete(StageName.RECONSTRUCT, warnings=_warnings(work.timeline.warnings))
        builder.add_deliverable(paths.timeline, relative_to=session_dir)
        builder.stage_complete(StageName.ACTIVITY, warnings=_warnings(work.graph.warnings))
        builder.add_deliverable(paths.graph, relative_to=session_dir)

        _mix_branch(
            session_dir,
            config,
            work,
            state,
            builder=builder,
            roots=roots,
            before=before,
            use_cache=use_cache,
        )
        if models is None:
            # The branch cannot run, and the reason is already known. Recorded here rather
            # than at resolution time so it lands in the same place as any other transcript
            # failure and the report cannot tell the two apart by accident.
            state.transcript_error = model_error
        else:
            _transcript_branch(
                session_dir,
                config,
                work,
                state,
                models=models,
                builder=builder,
                roots=roots,
                before=before,
                use_cache=use_cache,
                graph_sha256=graph_sha256,
            )

        # INV-01, over the complete run rather than over each commit point. With four
        # verification points a branch that failed before its own would otherwise leave the
        # window after the mix's source reads unchecked, and the invariant is about a run.
        verify_unchanged(session_dir, roots, before)
    except NotImplementedError:
        raise
    except Exception as exc:
        return _failed(exc, session_dir, paths, builder, state, now)

    _record(session_dir, builder, state)
    report = builder.write(paths.report, dt.datetime.now(dt.UTC) if now is None else now)
    return ProcessResult(
        graph=state.graph,
        encode=state.encode,
        records=state.records,
        mp3_path=paths.mp3,
        report=report,
        report_path=paths.report,
        exit_code=report.exit_code(),
    )


@dataclass
class _State:
    """What each branch produced, held as data until the final verification has passed."""

    graph: ActivityGraph | None = None
    mixed: MixWork | None = None
    encode: EncodeResult | None = None
    records: TranscriptRecords | None = None
    mix_error: StructuredError | None = None
    transcript_error: StructuredError | None = None


@dataclass(frozen=True, slots=True)
class _Paths:
    """Everywhere this command reads or writes, derived once."""

    session_dir: Path

    @property
    def graph(self) -> Path:
        return self.session_dir / ACTIVITY_RELATIVE_PATH

    @property
    def timeline(self) -> Path:
        return self.session_dir / TIMELINE_RELATIVE_PATH

    @property
    def records(self) -> Path:
        return self.session_dir / RECORDS_RELATIVE_PATH

    @property
    def transcript(self) -> Path:
        return self.session_dir / TRANSCRIPT_JSON_RELATIVE_PATH

    @property
    def markdown(self) -> Path:
        return self.session_dir / TRANSCRIPT_MARKDOWN_RELATIVE_PATH

    @property
    def mp3(self) -> Path:
        from dnd_audio.mix import MP3_RELATIVE_PATH

        return self.session_dir / MP3_RELATIVE_PATH

    @property
    def report(self) -> Path:
        return self.session_dir / OUTPUT_DIRNAME / REPORT_FILENAME


def _mix_branch(
    session_dir: Path,
    config: SessionConfig,
    work: ActivityWork,
    state: _State,
    *,
    builder: ReportBuilder,
    roots: tuple[str, ...],
    before: RawSnapshot,
    use_cache: bool,
) -> None:
    """Render, verify, commit, encode. A failure here is collected, never propagated."""
    try:
        mixed = perform_mix(session_dir, config, work, use_cache=use_cache)
        verify_unchanged(session_dir, roots, before)
        mixed.cache.commit()
        state.mixed = mixed
        state.encode = encode_deliverable(session_dir, config, mixed, builder=builder)
    except DiscoveryError as exc:
        if exc.code == "output_inside_raw":
            raise
        state.mix_error = _error(exc)
    except Exception as exc:
        state.mix_error = _error(exc)


def _resolve_or_defer(
    session_dir: Path,
    config: SessionConfig,
    transcriber: TranscriberBundle | None,
    detector: DetectorBundle | None,
    *,
    fake_models: bool,
) -> tuple[Models | None, StructuredError | None]:
    """Resolve both model seams, turning an ASR-side failure into the branch's error.

    Only the *transcript* seam may fail softly. A detector failure is not caught here and
    stops the run, and that is deliberate: the mix consumes the activity graph, so a
    detector that will not load has already made both branches impossible and pretending
    otherwise would produce a mix of silence. `resolve_models` builds the detector lazily
    from the fake-models file or takes the caller's, so in practice what fails here is the
    Qwen adapter — which is exactly the thing INV-09 says the mix must survive.
    """
    try:
        return resolve_models(
            session_dir, config, transcriber, detector, fake_models=fake_models
        ), None
    except DiscoveryError as exc:
        if exc.code == "output_inside_raw":
            raise
        return None, _error(exc)
    except Exception as exc:
        return None, _error(exc)


def _transcript_branch(
    session_dir: Path,
    config: SessionConfig,
    work: ActivityWork,
    state: _State,
    *,
    models: Models,
    builder: ReportBuilder,
    roots: tuple[str, ...],
    before: RawSnapshot,
    use_cache: bool,
    graph_sha256: str,
) -> None:
    """Plan, submit, collapse, verify, commit, write. A failure here is collected too."""
    paths = _Paths(session_dir)
    try:
        records, cache = perform_transcript(
            session_dir,
            config,
            work.graph,
            models=models,
            builder=builder,
            timeline_sha256=work.timeline_sha256,
            use_cache=use_cache,
        )
        # INV-09, checked rather than trusted, exactly as `transcribe` checks it: nothing this
        # branch decided may have reached the graph the mix consumed.
        if sha256_file(paths.graph) != graph_sha256:
            message = (
                "the activity graph changed while the transcript branch was running. Nothing "
                "text-derived may reach it: the mix must produce identical samples whether or "
                "not ASR ran at all (INV-09)."
            )
            raise DiscoveryError(message, code="activity_graph_modified")

        verify_unchanged(session_dir, roots, before)
        cache.commit()
        write_json_atomic(paths.records, records.model_dump(mode="json"))
        write_transcript_deliverables(records, session_dir)
        state.records = records
    except DiscoveryError as exc:
        if exc.code == "output_inside_raw":
            raise
        state.transcript_error = _error(exc)
    except Exception as exc:
        state.transcript_error = _error(exc)


def _record(session_dir: Path, builder: ReportBuilder, state: _State) -> None:
    """Turn both branches' outcomes into stage reports, after the final verification.

    Deliberately the last thing that happens: a stage recorded as complete before INV-01 was
    re-checked over the whole run would have to be retracted, and `ReportBuilder` has no way
    to retract one — by design, because a report that changed its mind would be worse than one
    that waited.
    """
    if state.mixed is not None and state.encode is not None:
        record_mix_stage(builder, session_dir, state.mixed, state.encode)
    else:
        error = state.mix_error or _unknown("the mix produced nothing and reported no error")
        builder.stage_failed(StageName.MIX, [error])
        remove_mix_artifacts(session_dir)

    if state.records is not None:
        builder.stage_complete(StageName.TRANSCRIBE, warnings=_warnings(state.records.warnings))
        builder.add_deliverable(session_dir / RECORDS_RELATIVE_PATH, relative_to=session_dir)
        record_render_stage(builder, session_dir)
    else:
        error = state.transcript_error or _unknown(
            "the transcript branch produced nothing and reported no error"
        )
        builder.stage_failed(StageName.TRANSCRIBE, [error])
        # The spec renders "when transcription succeeds"; it did not, so the render is a
        # failure of the same cause rather than a stage nobody thought about (INV-13).
        builder.stage_failed(StageName.RENDER, [error])
        for path in (
            session_dir / RECORDS_RELATIVE_PATH,
            session_dir / TRANSCRIPT_JSON_RELATIVE_PATH,
            session_dir / TRANSCRIPT_MARKDOWN_RELATIVE_PATH,
        ):
            path.unlink(missing_ok=True)


def _failed(
    exc: Exception,
    session_dir: Path,
    paths: _Paths,
    builder: ReportBuilder,
    state: _State,
    now: dt.datetime | None,
) -> ProcessResult:
    """A failure outside both branches: configuration, activity, or the final verification."""
    error = _error(exc)
    completed = [
        stage for stage in (StageName.RECONSTRUCT, StageName.ACTIVITY) if builder.completed(stage)
    ]

    # A branch that already diagnosed its own failure keeps that diagnosis. ADR-0024 says so
    # in as many words — "it does not replace whichever error the branch already reported" —
    # and the first implementation stamped the outer exception over every unrecorded stage, so
    # an ASR crash concurrent with unrelated source tampering was reported as tampering. Found
    # by M5's independent review.
    owned: dict[StageName, StructuredError] = {}
    if state.mix_error is not None:
        owned[StageName.MIX] = state.mix_error
    if state.transcript_error is not None:
        owned[StageName.TRANSCRIBE] = state.transcript_error
        owned[StageName.RENDER] = state.transcript_error

    for stage in StageName:
        if not builder.recorded(stage):
            builder.stage_failed(stage, [owned.get(stage, error)])

    finished = dt.datetime.now(dt.UTC) if now is None else now
    if isinstance(exc, DiscoveryError) and exc.code == "output_inside_raw":
        # INV-01 outranks INV-13, and this returns **before** any cleanup: with
        # `work -> raw/tx-a` every artifact path resolves inside a source directory, so the
        # unlinks below would themselves be writes into `raw/` (ADR-0021).
        return ProcessResult(
            graph=state.graph,
            encode=None,
            records=None,
            mp3_path=paths.mp3,
            report=builder.build(finished),
            report_path=paths.report,
            report_written=False,
            exit_code=ExitCode.FATAL,
        )

    remove_activity_artifacts(session_dir, completed=completed)
    remove_mix_artifacts(session_dir)
    for path in (paths.records, paths.transcript, paths.markdown):
        path.unlink(missing_ok=True)

    report = builder.write(paths.report, finished)
    return ProcessResult(
        graph=state.graph,
        encode=None,
        records=None,
        mp3_path=paths.mp3,
        report=report,
        report_path=paths.report,
        exit_code=report.exit_code(),
    )


class _Note(Protocol):
    """What the report needs from a warning, whichever artifact it came from.

    Three artifacts flatten into one report here — the timeline's, the graph's, and the
    records' — and each keeps its own note type because each belongs to the document that
    carries it. Read-only properties rather than bare annotations, because a bare annotation
    would make the protocol *settable* and a frozen pydantic model does not satisfy that.
    """

    @property
    def code(self) -> str: ...

    @property
    def message(self) -> str: ...

    @property
    def path(self) -> str | None: ...


def _warnings(notes: Sequence[_Note]) -> list[ReportWarning]:
    """Flatten artifact warnings for the report, in a stable order."""
    flattened = [
        ReportWarning(code=note.code, message=note.message, path=note.path) for note in notes
    ]
    return sorted(flattened, key=lambda note: (note.code, note.path or "", note.message))


def _error(exc: BaseException) -> StructuredError:
    code = exc.code if isinstance(exc, DndAudioError) else "internal_error"
    return StructuredError(code=code, message=str(exc) or type(exc).__name__)


def _unknown(message: str) -> StructuredError:
    return StructuredError(code="internal_error", message=message)

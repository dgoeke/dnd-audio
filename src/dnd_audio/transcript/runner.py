"""`dnd-audio transcribe` and `dnd-audio render`.

`transcribe` is the whole left branch of the spec's stage DAG in one command: inspect,
reconstruct, activity, ASR, and the transcript render. `render` is the last of those alone,
regenerating both deliverables from `work/transcript-records.json` with no model, no graph and
no mixer — provable rather than asserted, because the records are the only input it reads
(ADR-0019).

The ordering in `transcribe` is load-bearing and is stated rather than left to be inferred:

1. **Snapshot the raw roots once, and refuse outputs that would land inside them.** Once for
   the whole composed run, over the union of both stages' outputs, so the sources are hashed
   once rather than once per stage (INV-01).
2. **Perform the activity stages** through `perform_activity`, which leaves every cache staged.
3. **Verify INV-01, commit the activity caches, write the graph.** The first of two commit
   points. Two rather than one so that a failure during ASR — which reads no source audio —
   does not throw away six tracks of verified inference.
4. **Plan, submit, normalize, collapse.** One request in memory at a time (INV-07).
5. **Verify INV-01 again, commit the ASR cache, write the records and both deliverables.**
6. **Write one report covering five stages**, whichever way it went (INV-13), with the same
   carve-out the other commands have: when the report's own location resolves inside a source
   directory, nothing is written and INV-01 wins.

**INV-09 runs the other way here.** Nothing this stage decides may reach the activity graph, so
the graph's bytes are hashed when it is written and re-checked after ASR: a stage that wrote
back into it would fail loudly rather than change what the mix does.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from dnd_audio.activity.runner import (
    DEFAULT_DETECT_WINDOW,
    DetectorBundle,
    activity_outputs,
    derivative_paths,
    perform_activity,
    remove_activity_artifacts,
)
from dnd_audio.artifacts.activity import ActivityGraph
from dnd_audio.artifacts.records import (
    TranscriberIdentity,
    TranscriptNote,
    TranscriptRecords,
)
from dnd_audio.artifacts.report import (
    REPORT_FILENAME,
    Decision,
    IngestReport,
    ReportBuilder,
    ReportWarning,
    StageName,
    StructuredError,
)
from dnd_audio.config import SessionConfig, config_hash, load_session_config
from dnd_audio.determinism import sha256_bytes, sha256_file, write_atomic, write_json_atomic
from dnd_audio.errors import ConfigError, DiscoveryError, DndAudioError, ExitCode
from dnd_audio.inspection import OUTPUT_DIRNAME
from dnd_audio.interfaces import Transcriber
from dnd_audio.raw_guard import raw_roots, reject_outputs_inside_raw, snapshot, verify_unchanged
from dnd_audio.timeline import TIMELINE_RELATIVE_PATH
from dnd_audio.timeline.reader import DEFAULT_WINDOW_SAMPLES, DerivativeReader
from dnd_audio.transcript import (
    ASR_DIRNAME,
    RECORDS_RELATIVE_PATH,
    TRANSCRIPT_JSON_RELATIVE_PATH,
    TRANSCRIPT_MARKDOWN_RELATIVE_PATH,
)
from dnd_audio.transcript.asr import transcribe_plans
from dnd_audio.transcript.cache import AsrCache
from dnd_audio.transcript.collapse import collapse
from dnd_audio.transcript.document import build_records
from dnd_audio.transcript.render import build_transcript, render_markdown
from dnd_audio.transcript.requests import plan_context, plan_requests
from dnd_audio.transcript.segments import draft_segments

__all__ = [
    "Models",
    "RenderResult",
    "TranscribeResult",
    "TranscriberBundle",
    "perform_transcript",
    "record_render_stage",
    "render_outputs",
    "resolve_models",
    "run_render",
    "run_transcribe",
    "transcribe_outputs",
    "write_transcript_deliverables",
]

#: The one stage `transcribe` does not run, and why (INV-13).
_SKIPPED_BY_TRANSCRIBE: Final = (
    (StageName.MIX, "`transcribe` is the transcript branch; the mix is `mix` (INV-09)"),
)

#: What `render` does not run. It reads records and writes two files; everything upstream of
#: them already happened, and re-running it would be the opposite of what the command is for.
_SKIPPED_BY_RENDER: Final = (
    (StageName.INSPECT, "`render` regenerates outputs from records; it discovers nothing"),
    (StageName.RECONSTRUCT, "`render` reads no audio"),
    (StageName.ACTIVITY, "`render` reads no audio"),
    (StageName.TRANSCRIBE, "`render` reads cached records rather than running ASR"),
    (StageName.MIX, "`render` never mixes"),
)


@dataclass(frozen=True, slots=True)
class TranscriberBundle:
    """A transcriber and everything about it that reaches a cache key and the report.

    The same shape, and for the same reason, as `DetectorBundle`: the identity is needed
    before any request is submitted, because it is part of the key that decides whether one
    has to be submitted at all. Bundling the two means a caller cannot supply a transcriber
    whose identity describes a different one (INV-08).
    """

    transcriber: Transcriber
    name: str
    model: str | None = None
    model_revision: str | None = None
    aligner: str | None = None
    aligner_revision: str | None = None
    #: Distinguishes two instances of one implementation. A script hashes itself into it.
    variant_digest: str | None = None

    def identity(self, config: SessionConfig, context_sha256: str | None) -> TranscriberIdentity:
        return TranscriberIdentity(
            name=self.name,
            model=self.model,
            model_revision=self.model_revision,
            aligner=self.aligner,
            aligner_revision=self.aligner_revision,
            max_new_tokens=config.asr.max_new_tokens,
            language=config.language,
            context_sha256=context_sha256,
            variant_digest=self.variant_digest,
        )


@dataclass(frozen=True, slots=True)
class TranscribeResult:
    """What one `transcribe` run produced."""

    records: TranscriptRecords | None
    records_path: Path
    transcript_path: Path
    markdown_path: Path
    graph: ActivityGraph | None
    report: IngestReport
    report_path: Path
    exit_code: ExitCode
    #: False only when writing the report would itself have violated INV-01.
    report_written: bool = True


@dataclass(frozen=True, slots=True)
class RenderResult:
    """What one `render` run produced."""

    records: TranscriptRecords | None
    transcript_path: Path
    markdown_path: Path
    report: IngestReport
    report_path: Path
    exit_code: ExitCode
    report_written: bool = True


def transcribe_outputs(session_dir: Path) -> dict[str, Path]:
    """Everything `transcribe` writes, for the INV-01 output check.

    A superset of `activity`'s, because a composed run performs both. Declared as data so
    that adding an output and forgetting to protect it is a visible omission from one list.
    """
    return {
        **activity_outputs(session_dir),
        **render_outputs(session_dir),
        "the transcript records": session_dir / RECORDS_RELATIVE_PATH,
        "the ASR cache": session_dir / ASR_DIRNAME,
    }


def render_outputs(session_dir: Path) -> dict[str, Path]:
    """What `render` writes, which is only the two deliverables."""
    return {
        "the transcript": session_dir / TRANSCRIPT_JSON_RELATIVE_PATH,
        "the transcript markdown": session_dir / TRANSCRIPT_MARKDOWN_RELATIVE_PATH,
    }


def run_transcribe(
    session_dir: Path,
    *,
    transcriber: TranscriberBundle | None = None,
    detector: DetectorBundle | None = None,
    fake_models: bool = False,
    now: dt.datetime | None = None,
    use_cache: bool = True,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
    detect_window_samples: int | None = None,
) -> TranscribeResult:
    """Reconstruct, attribute, transcribe, and render.

    Never raises for an expected failure: a fatal condition becomes a failed stage, a
    structured error, a written report, and a nonzero exit code (INV-13).

    Args:
        transcriber: What to transcribe with. Defaults to the real adapter, which lands in
            M6b — until then a run without ``fake_models`` says so and stops.
        fake_models: Drive both seams from the session's declared `fake-models.json`
            (ADR-0018). Explicit, and fatal if the file is absent.
    """
    started_at = now or dt.datetime.now(dt.UTC)
    paths = _Paths(session_dir)
    builder = _builder(session_dir.name, None, started_at, _SKIPPED_BY_TRANSCRIBE)
    graph: ActivityGraph | None = None

    try:
        config = load_session_config(session_dir / "session.yaml")
        builder = _builder(
            config.session_id, config_hash(config), started_at, _SKIPPED_BY_TRANSCRIBE
        )
        roots = raw_roots(config)
        before = snapshot(session_dir, roots)
        reject_outputs_inside_raw(session_dir, config, roots, transcribe_outputs(session_dir))

        models = resolve_models(session_dir, transcriber, detector, fake_models=fake_models)
        work = perform_activity(
            session_dir,
            config,
            builder=builder,
            detector=models.detector,
            use_cache=use_cache,
            window_samples=window_samples,
            detect_window_samples=detect_window_samples or DEFAULT_DETECT_WINDOW,
        )

        # First commit point. The activity caches are verified and published here rather than
        # at the end, so an ASR failure — which reads no source audio and cannot invalidate
        # them — does not throw away six tracks of inference.
        verify_unchanged(session_dir, roots, before)
        work.commit()
        write_json_atomic(paths.graph, work.graph.model_dump(mode="json"))
        graph = work.graph
        graph_sha256 = sha256_file(paths.graph)

        builder.stage_complete(StageName.RECONSTRUCT, warnings=_notes(work.timeline.warnings))
        builder.add_deliverable(paths.timeline, relative_to=session_dir)
        builder.stage_complete(StageName.ACTIVITY, warnings=_notes(work.graph.warnings))
        builder.add_deliverable(paths.graph, relative_to=session_dir)

        records, cache = perform_transcript(
            session_dir,
            config,
            work.graph,
            models=models,
            builder=builder,
            timeline_sha256=work.timeline_sha256,
            use_cache=use_cache,
        )

        # INV-09, checked rather than trusted: nothing this stage decided may have reached the
        # graph the mix consumes. Re-hashing the file is cheap and catches a write from
        # anywhere, including from code that has no business touching it.
        if sha256_file(paths.graph) != graph_sha256:
            message = (
                "the activity graph changed while the transcript branch was running. Nothing "
                "text-derived may reach it: the mix must produce identical samples whether or "
                "not ASR ran at all (INV-09)."
            )
            raise DiscoveryError(message, code="activity_graph_modified")

        # Second commit point, and the same rule as the first: verify, then publish.
        verify_unchanged(session_dir, roots, before)
        cache.commit()
        write_json_atomic(paths.records, records.model_dump(mode="json"))
        builder.stage_complete(StageName.TRANSCRIBE, warnings=_notes(records.warnings))
        builder.add_deliverable(paths.records, relative_to=session_dir)

        write_transcript_deliverables(records, session_dir)
        record_render_stage(builder, session_dir)
    except NotImplementedError:
        # Deliberately not turned into a failed report. "This pipeline has not built that
        # yet" and "your session is broken" are different answers to different questions, and
        # ADR-0005 spends a distinct exit code on keeping them apart; a report saying
        # `internal_error` would send an operator looking in entirely the wrong place.
        raise
    except Exception as exc:
        return _failed_transcribe(exc, session_dir, paths, builder, graph, now)

    report = builder.write(paths.report, dt.datetime.now(dt.UTC) if now is None else now)
    return TranscribeResult(
        records=records,
        records_path=paths.records,
        transcript_path=paths.transcript,
        markdown_path=paths.markdown,
        graph=graph,
        report=report,
        report_path=paths.report,
        exit_code=report.exit_code(),
    )


def run_render(session_dir: Path, *, now: dt.datetime | None = None) -> RenderResult:
    """Regenerate both deliverables from `work/transcript-records.json`.

    No model, no activity graph, no timeline, no mixer — the records are the only input, which
    is what makes the spec's "must not invoke ASR or audio mixing" a property of this function
    rather than a claim about it. Absent records are a clear, named failure, as the spec asks.
    """
    started_at = now or dt.datetime.now(dt.UTC)
    paths = _Paths(session_dir)
    builder = _builder(session_dir.name, None, started_at, _SKIPPED_BY_RENDER)
    records: TranscriptRecords | None = None

    try:
        config = load_session_config(session_dir / "session.yaml")
        builder = _builder(config.session_id, config_hash(config), started_at, _SKIPPED_BY_RENDER)
        roots = raw_roots(config)
        reject_outputs_inside_raw(session_dir, config, roots, render_outputs(session_dir))

        records = _read_records(paths.records)
        warnings = _stale_records(records, config)
        write_transcript_deliverables(records, session_dir)
        record_render_stage(builder, session_dir, warnings=warnings)
    except Exception as exc:
        error = StructuredError(code=_code_of(exc), message=str(exc) or type(exc).__name__)
        if not builder.recorded(StageName.RENDER):
            builder.stage_failed(StageName.RENDER, [error])
        finished = dt.datetime.now(dt.UTC) if now is None else now
        if isinstance(exc, DiscoveryError) and exc.code == "output_inside_raw":
            return RenderResult(
                records=None,
                transcript_path=paths.transcript,
                markdown_path=paths.markdown,
                report=builder.build(finished),
                report_path=paths.report,
                report_written=False,
                exit_code=ExitCode.FATAL,
            )
        report = builder.write(paths.report, finished)
        return RenderResult(
            records=None,
            transcript_path=paths.transcript,
            markdown_path=paths.markdown,
            report=report,
            report_path=paths.report,
            exit_code=report.exit_code(),
        )

    report = builder.write(paths.report, dt.datetime.now(dt.UTC) if now is None else now)
    return RenderResult(
        records=records,
        transcript_path=paths.transcript,
        markdown_path=paths.markdown,
        report=report,
        report_path=paths.report,
        exit_code=report.exit_code(),
    )


@dataclass(frozen=True, slots=True)
class _Paths:
    """Everywhere this stage reads or writes, derived once."""

    session_dir: Path

    @property
    def graph(self) -> Path:
        from dnd_audio.activity import ACTIVITY_RELATIVE_PATH

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
    def report(self) -> Path:
        return self.session_dir / OUTPUT_DIRNAME / REPORT_FILENAME


@dataclass(frozen=True, slots=True)
class Models:
    """The two seams, resolved together with whatever an operator must be told about them."""

    transcriber: TranscriberBundle
    detector: DetectorBundle | None
    warnings: tuple[TranscriptNote, ...] = ()


def resolve_models(
    session_dir: Path,
    transcriber: TranscriberBundle | None,
    detector: DetectorBundle | None,
    *,
    fake_models: bool,
) -> Models:
    """Resolve both model seams, explicitly and visibly (ADR-0018)."""
    if transcriber is not None:
        return Models(transcriber=transcriber, detector=detector)
    if not fake_models:
        return Models(transcriber=_default_transcriber(session_dir), detector=detector)

    from dnd_audio.transcript.fakemodels import load_fake_models

    fake = load_fake_models(session_dir)
    return Models(
        transcriber=TranscriberBundle(
            transcriber=fake.transcriber, name=fake.name, variant_digest=fake.digest
        ),
        detector=detector or fake.detector,
        warnings=(
            TranscriptNote(
                code="fake_models_in_use",
                message=(
                    "speech detection and transcription came from this session's declared "
                    "fake-models.json, not from a model. Every text in this transcript was "
                    "written by whoever generated the fixture (ADR-0018)."
                ),
            ),
        ),
    )


def _default_transcriber(session_dir: Path) -> TranscriberBundle:
    """The real adapter, which does not exist yet.

    Raises the builtin `NotImplementedError` annotated at the raise site, the same shape every
    other unbuilt stage uses, so `scripts/scan_placeholders.py` can see it. The seam is real
    and everything above it is finished; what is missing is one implementation behind it.
    """
    # DEFERRED: M6b
    raise NotImplementedError(
        f"`transcribe` needs the Qwen adapter, which lands in M6b. Until then, a synthetic "
        f"session can be transcribed from its own declared script with --fake-models "
        f"({session_dir})"
    )


def perform_transcript(
    session_dir: Path,
    config: SessionConfig,
    graph: ActivityGraph,
    *,
    models: Models,
    builder: ReportBuilder,
    timeline_sha256: str,
    use_cache: bool,
) -> tuple[TranscriptRecords, AsrCache]:
    """Plan, submit, normalize, and collapse. Returns the records and the staged cache."""
    glossary = _glossary(session_dir, config)
    identity = models.transcriber.identity(
        config, None if glossary is None else sha256_bytes(glossary.encode("utf-8"))
    )
    cache = AsrCache(session_dir=session_dir, read_enabled=use_cache)
    context = plan_context(graph, asr=config.asr, transcript=config.transcript)
    plans = plan_requests(graph, asr=config.asr, transcript=config.transcript)

    with DerivativeReader(session_dir, _timeline_paths(session_dir, graph)) as audio:
        outcome = transcribe_plans(
            plans,
            read=audio.read,
            transcriber=models.transcriber.transcriber,
            cache=cache,
            identity=identity,
            context=context,
            settings=config.transcript,
            language=config.language,
            glossary=glossary,
        )

    drafts, notes = draft_segments(outcome.outcomes, decimation=context.decimation)
    decided = collapse(
        drafts,
        graph,
        settings=config.transcript.duplicate,
        overlap_min_samples=config.transcript.overlap_min_ms * graph.sample_rate // 1000,
    )
    records = build_records(
        config,
        graph,
        drafts,
        decided.verdicts,
        transcriber=identity,
        timeline_sha256=timeline_sha256,
        warnings=[*models.warnings, *outcome.warnings, *notes],
        decisions=decided.decisions,
    )

    builder.record_cache(hits=cache.hits, misses=cache.misses)
    _record_transcriber(builder, identity)
    for decision in records.decisions:
        builder.record_decision(
            Decision(code=decision.code, subject=decision.subject, detail=decision.detail)
        )
    return records, cache


def _timeline_paths(session_dir: Path, graph: ActivityGraph) -> dict[str, str]:
    """Each track's 16 kHz derivative, read back from the timeline beside the graph.

    The graph names the tracks; the timeline names the files. Reading the timeline rather than
    caching the paths from `perform_activity` keeps this function usable from a run that only
    has the artifacts — which is what `render` proves is possible for its own inputs.
    """
    from dnd_audio.artifacts.timeline import Timeline

    document = (session_dir / TIMELINE_RELATIVE_PATH).read_text(encoding="utf-8")
    timeline = Timeline.model_validate_json(document)
    paths = derivative_paths(timeline)
    return {
        track.track_id: paths[track.track_id] for track in graph.tracks if track.track_id in paths
    }


def _glossary(session_dir: Path, config: SessionConfig) -> str | None:
    """The session's glossary text, or ``None``. Its absence must not block a run."""
    if config.asr.context_file is None:
        return None
    path = session_dir / config.asr.context_file
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def write_transcript_deliverables(records: TranscriptRecords, session_dir: Path) -> None:
    """Write `transcript.json` and `transcript.md`. Records nothing.

    Split from the recording so a caller can perform its **final** INV-01 verification between
    the write and the record — which is what `process` needs, since a stage marked complete
    before that check would have to be un-marked afterwards (ADR-0024).
    """
    paths = _Paths(session_dir)
    write_json_atomic(paths.transcript, build_transcript(records).model_dump(mode="json"))
    write_atomic(paths.markdown, render_markdown(records))


def record_render_stage(
    builder: ReportBuilder,
    session_dir: Path,
    *,
    warnings: Sequence[TranscriptNote] = (),
) -> None:
    """Mark the render complete and hash both deliverables."""
    paths = _Paths(session_dir)
    builder.stage_complete(StageName.RENDER, warnings=_notes(warnings))
    builder.add_deliverable(paths.transcript, relative_to=session_dir)
    builder.add_deliverable(paths.markdown, relative_to=session_dir)


def _read_records(path: Path) -> TranscriptRecords:
    """The records, or a clear failure. The spec asks for the second in as many words."""
    try:
        document = path.read_text(encoding="utf-8")
    except OSError as exc:
        message = (
            f"no transcript records at {path}: `render` regenerates the transcript from the "
            f"records `transcribe` writes, and there are none. Run `dnd-audio transcribe` "
            f"first."
        )
        raise ConfigError(message, code="transcript_records_missing") from exc
    try:
        return TranscriptRecords.model_validate_json(document)
    except ValueError as exc:
        message = f"{path} is not a usable transcript records document: {exc}"
        raise ConfigError(message, code="transcript_records_unreadable") from exc


def _stale_records(records: TranscriptRecords, config: SessionConfig) -> list[TranscriptNote]:
    """Warn when the records describe a different configuration than the one on disk.

    A warning rather than a failure: rendering them is still what was asked for, and the
    records say plainly which configuration they were made under — which is the whole reason
    they carry it (ADR-0019).
    """
    current = config_hash(config)
    if records.config_hash == current:
        return []
    return [
        TranscriptNote(
            code="transcript_records_stale",
            message=(
                f"these records were made under configuration {records.config_hash[:12]} and "
                f"this session now resolves to {current[:12]}. They are rendered as they are; "
                f"re-run `dnd-audio transcribe` to bring them up to date."
            ),
        )
    ]


def _failed_transcribe(
    exc: Exception,
    session_dir: Path,
    paths: _Paths,
    builder: ReportBuilder,
    graph: ActivityGraph | None,
    now: dt.datetime | None,
) -> TranscribeResult:
    """Every failure, not only the ones raised on purpose (INV-13).

    An operator whose run died on an OSError needs a report more than anyone. Every artifact
    of a stage that did **not** complete is removed: a stale transcript beside a report that
    calls the stage failed is worse than none, because the file looks current and nothing in
    it says otherwise. The artifacts of a stage the report calls *complete* are kept, because
    this run commits at two points and their hashes are already in the report — deleting them
    would leave it advertising the hash of a file that is gone (M4's verify phase).
    """
    error = StructuredError(code=_code_of(exc), message=str(exc) or type(exc).__name__)
    completed = [
        stage for stage in (StageName.RECONSTRUCT, StageName.ACTIVITY) if builder.completed(stage)
    ]
    for stage in (
        StageName.INSPECT,
        StageName.RECONSTRUCT,
        StageName.ACTIVITY,
        StageName.TRANSCRIBE,
        StageName.RENDER,
    ):
        if not builder.recorded(stage):
            builder.stage_failed(stage, [error])

    finished = dt.datetime.now(dt.UTC) if now is None else now
    if isinstance(exc, DiscoveryError) and exc.code == "output_inside_raw":
        # INV-01 outranks INV-13 here: writing the failure report would commit the very
        # violation being reported. A report is regenerable; a source directory written into
        # is not.
        #
        # Returned **before** any cleanup, and that ordering is the invariant rather than a
        # detail: `work -> raw/tx-a` makes every artifact path resolve inside a source
        # directory, so unlinking the stale ones is itself a write into `raw/`.
        return TranscribeResult(
            records=None,
            records_path=paths.records,
            transcript_path=paths.transcript,
            markdown_path=paths.markdown,
            graph=graph,
            report=builder.build(finished),
            report_path=paths.report,
            report_written=False,
            exit_code=ExitCode.FATAL,
        )

    remove_activity_artifacts(session_dir, completed=completed)
    for path in (paths.records, paths.transcript, paths.markdown):
        path.unlink(missing_ok=True)

    report = builder.write(paths.report, finished)
    return TranscribeResult(
        records=None,
        records_path=paths.records,
        transcript_path=paths.transcript,
        markdown_path=paths.markdown,
        graph=graph,
        report=report,
        report_path=paths.report,
        exit_code=report.exit_code(),
    )


def _record_transcriber(builder: ReportBuilder, identity: TranscriberIdentity) -> None:
    """Put the transcriber's identity in the report, as INV-08 and the spec require."""
    builder.record_model_identity("asr", identity.model or identity.name)
    if identity.model_revision is not None:
        builder.record_model_identity("asr_revision", identity.model_revision)
    if identity.aligner is not None:
        builder.record_model_identity("aligner", identity.aligner)
    if identity.aligner_revision is not None:
        builder.record_model_identity("aligner_revision", identity.aligner_revision)
    if identity.variant_digest is not None:
        builder.record_model_identity("asr_variant", identity.variant_digest)
    builder.record_model_identity("asr_max_new_tokens", str(identity.max_new_tokens))
    builder.record_model_identity("asr_language", identity.language)
    if identity.context_sha256 is not None:
        builder.record_model_identity("asr_context_sha256", identity.context_sha256)


def _builder(
    session_id: str,
    hash_: str | None,
    started_at: dt.datetime,
    skipped: Sequence[tuple[StageName, str]],
) -> ReportBuilder:
    builder = ReportBuilder(session_id=session_id, config_hash=hash_, started_at=started_at)
    for stage, reason in skipped:
        builder.stage_skipped(stage, reason)
    return builder


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


def _notes(notes: Sequence[_Note]) -> list[ReportWarning]:
    """Flatten artifact warnings for the report, in a stable order."""
    flattened = [
        ReportWarning(code=note.code, message=note.message, path=note.path) for note in notes
    ]
    return sorted(flattened, key=lambda note: (note.code, note.path or "", note.message))


def _code_of(exc: BaseException) -> str:
    if isinstance(exc, DndAudioError):
        return exc.code
    return "internal_error"

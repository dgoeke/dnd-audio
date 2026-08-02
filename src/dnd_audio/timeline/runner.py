"""`dnd-audio ingest`: reconstruct the timeline and derive the working audio.

The ordering is load-bearing and is stated rather than left to be inferred:

1. **Snapshot every file under the raw roots, and refuse outputs that would land inside
   them.** Once, around the whole run — including the inspection it performs — so the
   sources are hashed once rather than twice (INV-01).
2. **Run inspection.** Every time, not only when the manifest is missing. Reusing a
   manifest on a configuration-hash match lets a replaced or deleted WAV pass unnoticed:
   the hashes stay internally consistent and the snapshot only covers mutations *during* a
   run. The spec defines `ingest` as "run `inspect` as needed, then construct…", and M1's
   content cache makes a warm re-inspection cost no FFprobe at all. When every source is a
   cache hit the stage is recorded as `complete` with `origin: reused` — it did complete,
   and its outputs are current, which `skipped` would misreport.
3. **Refuse sources that cannot be placed** — a non-48 kHz rate, chunks disagreeing about
   theirs, a codec the working path cannot read. Before any placement, which is the only
   order in which "fails before timeline construction" is a fact rather than a hope.
4. **Place, lay out, preflight, derive.** The preflight runs after the timeline exists and
   before the first derivative byte, because that is when the real numbers are known and
   still useful.
5. **Write the timeline, then the report.** Whichever way it went (INV-13), with the same
   carve-out M1 has: when the report's own location resolves inside a source directory,
   nothing is written and INV-01 wins.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import scipy

from dnd_audio.artifacts.manifest import Manifest, RationalRate
from dnd_audio.artifacts.report import (
    REPORT_FILENAME,
    Decision,
    IngestReport,
    ReportBuilder,
    ReportWarning,
    StageName,
    StageOrigin,
    StructuredError,
)
from dnd_audio.artifacts.timeline import (
    DerivativeRecord,
    Timeline,
    TimelineProvenance,
    TimelineTrack,
)
from dnd_audio.config import SessionConfig, config_hash, load_session_config
from dnd_audio.determinism import sha256_file, write_json_atomic
from dnd_audio.errors import DiscoveryError, DndAudioError, ExitCode
from dnd_audio.inspection import INSPECTION_SEMANTICS_VERSION, OUTPUT_DIRNAME
from dnd_audio.inspection.cache import CACHE_DIRNAME, InspectionCache
from dnd_audio.inspection.runner import (
    MANIFEST_RELATIVE_PATH,
    PROBE_DIRNAME,
    inspect_session,
)
from dnd_audio.raw_guard import raw_roots, reject_outputs_inside_raw, snapshot, verify_unchanged
from dnd_audio.timecode import parse_frame_rate
from dnd_audio.timeline import (
    CANONICAL_SAMPLE_RATE,
    DERIVATIVE_SAMPLE_RATE,
    TIMELINE_DIRNAME,
    TIMELINE_RELATIVE_PATH,
    TIMELINE_SEMANTICS_VERSION,
)
from dnd_audio.timeline.derivatives import (
    DerivativeCache,
    derivative_identity,
    derivative_relative_path,
)
from dnd_audio.timeline.fir import load_decimation_filter
from dnd_audio.timeline.layout import TrackLayout, build_layout, reject_unusable_sources
from dnd_audio.timeline.origin import determine_origin
from dnd_audio.timeline.preflight import estimate, preflight
from dnd_audio.timeline.reader import DEFAULT_WINDOW_SAMPLES, TrackReader
from dnd_audio.timeline.resample import decimate_stream, output_length
from dnd_audio.timeline.syncqa import run_sync_qa
from dnd_audio.timeline.warp import IdentityWarp, TimeWarp
from dnd_audio.timeline.wavwrite import WavWriter

__all__ = ["IngestResult", "ingest_outputs", "run_ingest"]

#: The stages `ingest` does not run, and why (INV-13).
_SKIPPED_STAGES: Final = (
    (StageName.ACTIVITY, "`ingest` builds the timeline; interpreting audio content is M3"),
    (StageName.TRANSCRIBE, "`ingest` does not run ASR"),
    (StageName.RENDER, "there is no transcript to render"),
    (StageName.MIX, "`ingest` does not produce a mix"),
)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """What one `ingest` run produced."""

    timeline: Timeline | None
    timeline_path: Path
    report: IngestReport
    report_path: Path
    exit_code: ExitCode
    #: False only when writing the report would itself have violated INV-01.
    report_written: bool = True


def ingest_outputs(session_dir: Path) -> dict[str, Path]:
    """Everything `ingest` writes, for the INV-01 output check.

    A superset of `inspect`'s, because `ingest` performs an inspection as its first step.
    Declared as data so that adding an output and forgetting to protect it is a visible
    omission from one list.
    """
    return {
        "the manifest": session_dir / MANIFEST_RELATIVE_PATH,
        "the FFprobe sidecars": session_dir / PROBE_DIRNAME,
        "the inspection cache": session_dir / CACHE_DIRNAME,
        "the timeline": session_dir / TIMELINE_RELATIVE_PATH,
        "the working-audio cache": session_dir / TIMELINE_DIRNAME,
        "the report": session_dir / OUTPUT_DIRNAME / REPORT_FILENAME,
    }


def run_ingest(
    session_dir: Path,
    *,
    now: dt.datetime | None = None,
    use_cache: bool = True,
    materialize_48k: bool = False,
    warp: TimeWarp | None = None,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
) -> IngestResult:
    """Build the session timeline and its working audio.

    Never raises for an expected failure: a fatal condition becomes a failed stage, a
    structured error, a written report, and a nonzero exit code (INV-13).

    Args:
        now: Injectable clock for the report's telemetry. Nothing deterministic reads it.
        use_cache: False re-derives everything without deleting anything.
        materialize_48k: Also write contiguous float32 RF64 working audio. Off by default:
            the segment map is the working path, and these are disposable cache artifacts
            for debugging and interoperability (ADR-0011).
        warp: The affine drift-correction seam. The MVP passes the identity (OQ-006).
        window_samples: Bound on every audio read and write (INV-07).
    """
    started_at = now or dt.datetime.now(dt.UTC)
    timeline_path = session_dir / TIMELINE_RELATIVE_PATH
    report_path = session_dir / OUTPUT_DIRNAME / REPORT_FILENAME
    builder = _builder(session_dir.name, None, started_at)

    try:
        config = load_session_config(session_dir / "session.yaml")
        builder = _builder(config.session_id, config_hash(config), started_at)
        timeline = _ingest(
            session_dir,
            config,
            builder=builder,
            use_cache=use_cache,
            materialize_48k=materialize_48k,
            warp=warp or IdentityWarp(),
            window_samples=window_samples,
        )
        write_json_atomic(timeline_path, timeline.model_dump(mode="json"))
        builder.stage_complete(StageName.RECONSTRUCT, warnings=_report_warnings(timeline))
        builder.add_deliverable(timeline_path, relative_to=session_dir)
    except Exception as exc:
        # Every failure, not only the ones raised on purpose: an operator whose run died
        # on an OSError needs a report more than anyone, and a traceback with no report is
        # the worst of both.
        _remove_stale(timeline_path)
        error = StructuredError(code=_code_of(exc), message=str(exc) or type(exc).__name__)
        # A failure before or during inspection leaves that stage unrecorded, and
        # `build()` refuses a report with a gap in it — so without this, the earliest
        # failures produced no report at all, which is precisely what INV-13 forbids.
        if not builder.recorded(StageName.INSPECT):
            builder.stage_failed(StageName.INSPECT, [error])
        builder.stage_failed(StageName.RECONSTRUCT, [error])
        finished = dt.datetime.now(dt.UTC) if now is None else now
        if isinstance(exc, DiscoveryError) and exc.code == "output_inside_raw":
            # INV-01 outranks INV-13 here: writing the failure report would commit the
            # very violation being reported. A report is regenerable; a source directory
            # written into is not.
            return IngestResult(
                timeline=None,
                timeline_path=timeline_path,
                report=builder.build(finished),
                report_path=report_path,
                report_written=False,
                exit_code=ExitCode.FATAL,
            )
        report = builder.write(report_path, finished)
        return IngestResult(
            timeline=None,
            timeline_path=timeline_path,
            report=report,
            report_path=report_path,
            exit_code=report.exit_code(),
        )

    report = builder.write(report_path, dt.datetime.now(dt.UTC) if now is None else now)
    return IngestResult(
        timeline=timeline,
        timeline_path=timeline_path,
        report=report,
        report_path=report_path,
        exit_code=report.exit_code(),
    )


def _builder(session_id: str, hash_: str | None, started_at: dt.datetime) -> ReportBuilder:
    builder = ReportBuilder(session_id=session_id, config_hash=hash_, started_at=started_at)
    for stage, reason in _SKIPPED_STAGES:
        builder.stage_skipped(stage, reason)
    return builder


def _ingest(
    session_dir: Path,
    config: SessionConfig,
    *,
    builder: ReportBuilder,
    use_cache: bool,
    materialize_48k: bool,
    warp: TimeWarp,
    window_samples: int,
) -> Timeline:
    """The run itself. Raises :class:`DndAudioError` on any fatal condition."""
    roots = raw_roots(config)
    before = snapshot(session_dir, roots)
    reject_outputs_inside_raw(session_dir, config, roots, ingest_outputs(session_dir))

    inspection_cache, manifest = _inspect(session_dir, config, builder=builder, use_cache=use_cache)
    reject_unusable_sources(manifest)

    origin = determine_origin(manifest, config, warp=warp)
    layouts, layout_decisions, layout_warnings = build_layout(manifest, config, origin)
    tracks = [_as_track(layout) for layout in layouts]
    duration = max((track.end_sample for track in tracks), default=0)

    notes = [*origin.warnings, *layout_warnings]
    notes.extend(
        preflight(
            estimate(
                session_dir,
                duration_samples=duration,
                track_count=sum(1 for track in tracks if track.segments),
                materialize_48k=materialize_48k,
            )
        )
    )

    cache = DerivativeCache(session_dir=session_dir, read_enabled=use_cache)
    tracks = [
        _with_derivatives(
            session_dir,
            track,
            config=config,
            duration=duration,
            cache=cache,
            materialize_48k=materialize_48k,
            window_samples=window_samples,
        )
        for track in tracks
    ]

    notes.extend(run_sync_qa(session_dir, config, tracks, builder=builder))

    # Verify first, publish second. A source that changed under the pipeline must not be
    # able to leave behind a cache entry describing bytes that no longer exist — M1's
    # inspection cache stages for exactly this reason, and the derivative cache now does
    # too. Until commit() runs, a derivative's audio is on disk with no sidecar naming it,
    # which reads as a miss.
    verify_unchanged(session_dir, roots, before)
    inspection_cache.commit()
    cache.commit()
    builder.record_cache(hits=cache.hits, misses=cache.misses)
    builder.record_package_version("dnd_audio.timeline", str(TIMELINE_SEMANTICS_VERSION))
    builder.record_tool_version("numpy", np.__version__)
    builder.record_tool_version("scipy", scipy.__version__)
    for decision in [*origin.decisions, *layout_decisions]:
        builder.record_decision(
            Decision(code=decision.code, subject=decision.subject, detail=decision.detail)
        )

    frame_rate = parse_frame_rate(config.timecode.frame_rate)
    return Timeline(
        session_id=config.session_id,
        config_hash=config_hash(config),
        manifest_sha256=sha256_file(session_dir / MANIFEST_RELATIVE_PATH),
        provenance=TimelineProvenance(
            timeline_semantics_version=TIMELINE_SEMANTICS_VERSION,
            inspection_semantics_version=INSPECTION_SEMANTICS_VERSION,
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
        ),
        sample_rate=CANONICAL_SAMPLE_RATE,
        duration_samples=duration,
        session_zero=origin.zero,
        frame_rate_label=frame_rate.label,
        frame_rate=RationalRate(
            numerator=frame_rate.rate.numerator, denominator=frame_rate.rate.denominator
        ),
        tracks=tracks,
        warnings=notes,
        decisions=[*origin.decisions, *layout_decisions],
    )


def _inspect(
    session_dir: Path, config: SessionConfig, *, builder: ReportBuilder, use_cache: bool
) -> tuple[InspectionCache, Manifest]:
    """Run inspection and write a current manifest.

    Returns the cache **uncommitted**. Its entries are published by the caller once INV-01
    has been re-verified, so a run that discovers a source changed under it cannot leave a
    record describing bytes that are gone.

    Unconditional, and cheap when warm. See the module docstring's second step for why a
    configuration-hash match is not sufficient evidence that a manifest still describes
    what is on disk.
    """
    cache = InspectionCache(directory=session_dir / CACHE_DIRNAME, read_enabled=use_cache)
    manifest = inspect_session(session_dir, config, cache=cache, builder=builder, verify_raw=False)
    manifest_path = session_dir / MANIFEST_RELATIVE_PATH
    write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
    cache.commit()

    builder.stage_complete(
        StageName.INSPECT,
        warnings=_manifest_warnings(manifest),
        origin=StageOrigin.REUSED if cache.misses == 0 else StageOrigin.EXECUTED,
    )
    builder.add_deliverable(manifest_path, relative_to=session_dir)
    builder.record_cache(hits=cache.hits, misses=cache.misses)
    return cache, manifest


def _as_track(layout: TrackLayout) -> TimelineTrack:
    return TimelineTrack(
        track_id=layout.track_id,
        speaker_id=layout.speaker_id,
        speaker_name=layout.speaker_name,
        start_sample=layout.start_sample,
        end_sample=layout.end_sample,
        segments=list(layout.segments),
        warnings=list(layout.warnings),
    )


def _with_derivatives(
    session_dir: Path,
    track: TimelineTrack,
    *,
    config: SessionConfig,
    duration: int,
    cache: DerivativeCache,
    materialize_48k: bool,
    window_samples: int,
) -> TimelineTrack:
    """Build (or reuse) this track's working audio and attach the records."""
    if not track.segments:
        return track

    records = [
        _derive(
            session_dir,
            track,
            config=config,
            duration=duration,
            cache=cache,
            target_rate=DERIVATIVE_SAMPLE_RATE,
            window_samples=window_samples,
        )
    ]
    if materialize_48k:
        records.append(
            _derive(
                session_dir,
                track,
                config=config,
                duration=duration,
                cache=cache,
                target_rate=CANONICAL_SAMPLE_RATE,
                window_samples=window_samples,
            )
        )
    return track.model_copy(update={"derivatives": records})


def _derive(
    session_dir: Path,
    track: TimelineTrack,
    *,
    config: SessionConfig,
    duration: int,
    cache: DerivativeCache,
    target_rate: int,
    window_samples: int,
) -> DerivativeRecord:
    """One track at one rate: reuse if the identity matches, otherwise stream it out."""
    decimation_filter = load_decimation_filter()
    resampling = target_rate != CANONICAL_SAMPLE_RATE
    key = derivative_identity(
        track,
        config_hash=config_hash(config),
        target_rate=target_rate,
        filter_identity=decimation_filter.identity if resampling else None,
    )
    expected = output_length(duration, decimation_filter.decimation) if resampling else duration

    found = cache.get(key, target_rate, expected_samples=expected)
    if found is None:
        audio_path = cache.audio_path(key, target_rate)
        with TrackReader(session_dir, track, duration) as reader:
            blocks = (block for _, block in reader.windows(window_samples=window_samples))
            with WavWriter(audio_path, sample_rate=target_rate, n_samples=expected) as writer:
                if resampling:
                    for produced in decimate_stream(
                        blocks, duration, decimation_filter=decimation_filter
                    ):
                        writer.write(produced)
                else:
                    for block in blocks:
                        writer.write(block)
        found = cache.publish(key, target_rate=target_rate, n_samples=expected)

    return DerivativeRecord(
        sample_rate=target_rate,
        relative_path=derivative_relative_path(target_rate, key),
        cache_key=key,
        size_bytes=found.size_bytes,
        input_samples=duration,
        output_samples=expected,
        decimation=decimation_filter.decimation if resampling else 1,
        filter_name=decimation_filter.name if resampling else "none",
        filter_identity=decimation_filter.identity if resampling else "0" * 64,
        group_delay_input_samples=decimation_filter.group_delay_input if resampling else 0,
        group_delay_output_samples=decimation_filter.group_delay_output if resampling else 0,
    )


def _manifest_warnings(manifest: Manifest) -> list[ReportWarning]:
    """Session-level and per-source warnings, flattened for the report."""
    flattened = [
        ReportWarning(code=note.code, message=note.message, path=note.path)
        for note in manifest.warnings
    ]
    for source in [*(s for t in manifest.tracks for s in t.sources), *manifest.unassigned]:
        flattened.extend(
            ReportWarning(code=note.code, message=note.message, path=source.relative_path)
            for note in source.warnings
        )
    return sorted(flattened, key=lambda note: (note.code, note.path or "", note.message))


def _report_warnings(timeline: Timeline) -> list[ReportWarning]:
    flattened = [
        ReportWarning(code=note.code, message=note.message, path=note.path)
        for note in timeline.warnings
    ]
    for track in timeline.tracks:
        flattened.extend(
            ReportWarning(code=note.code, message=note.message, path=note.path or track.track_id)
            for note in track.warnings
        )
    return sorted(flattened, key=lambda note: (note.code, note.path or "", note.message))


def _remove_stale(path: Path) -> None:
    """Delete a timeline left by an earlier successful run.

    A failed run that leaves the previous timeline in place is worse than one that leaves
    none: the file looks current, describes a session that no longer ingests, and nothing
    in it says so. The report records the failure; the stale artifact would contradict it.
    """
    path.unlink(missing_ok=True)


def _code_of(exc: BaseException) -> str:
    if isinstance(exc, DndAudioError):
        return exc.code
    return "internal_error"

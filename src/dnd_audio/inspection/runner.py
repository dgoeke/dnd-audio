"""`dnd-audio inspect`: discover, capture, write the manifest, write the report.

The ordering here is load-bearing, so it is stated rather than left to be inferred:

1. **Snapshot every file under the raw roots** before anything else. That snapshot is
   what INV-01 is verified against at the end, and it covers *every* file — including
   ones inspection never selected, because "we did not touch what we did not read" is a
   weaker claim than the invariant makes.
2. **Refuse to run if an output path would land inside a raw root.** The spec lists it
   among the fatal errors, and it has to be checked before the first write rather than
   after.
3. **Discover, then capture.** Only selected sources are probed: a duplicate and an
   ignored `edit` are recorded and left alone.
4. **Persist FFprobe's bytes before parsing them.** A document this code cannot read is
   exactly the document a human will want (OQ-001).
5. **Verify the snapshot, then commit the cache.** In that order, so a source that
   changed under the pipeline cannot leave behind a cache entry describing bytes that
   no longer exist.
6. **Write the report whichever way it went.** INV-13: a failed inspection still
   produces a report, `inspect` is marked failed with a structured error, the five
   stages that did not run are `skipped` with reasons, and the exit code is nonzero.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from dnd_audio.artifacts.manifest import (
    BwfSampleReferenceRecord,
    ContainerRecord,
    DeclinedStrategyRecord,
    FilenameHintsRecord,
    InspectionProvenance,
    Manifest,
    ManifestDecision,
    ManifestNote,
    ManifestSource,
    ManifestTrack,
    ProbeRecord,
    RationalRate,
    RiffChunkRecord,
    RiffRecord,
    SessionOffsetRecord,
    StartTimeRecord,
    TimecodeRecord,
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
from dnd_audio.artifacts.roster import RosterSummary
from dnd_audio.config import SessionConfig, config_hash, load_session_config
from dnd_audio.determinism import sha256_file, write_atomic, write_json_atomic
from dnd_audio.errors import DiscoveryError, DndAudioError, ExitCode
from dnd_audio.inspection import INSPECTION_SEMANTICS_VERSION
from dnd_audio.inspection.cache import CACHE_DIRNAME, InspectionCache, cache_key
from dnd_audio.inspection.discovery import DiscoveredSource, Discovery, discover
from dnd_audio.inspection.probe import (
    FFPROBE_ARGS,
    ToolVersions,
    exact_sample_count,
    format_tags,
    parse_probe,
    read_audio_properties,
    run_ffprobe,
    tool_versions,
)
from dnd_audio.inspection.riff import RiffInventory, read_inventory
from dnd_audio.inspection.starttime import (
    BwfSampleReference,
    SourceContext,
    StartTime,
    TimecodeReference,
    extract_start_time,
)
from dnd_audio.timecode import parse_frame_rate

__all__ = [
    "MANIFEST_RELATIVE_PATH",
    "PROBE_DIRNAME",
    "InspectionResult",
    "inspect_session",
    "run_inspect",
]

MANIFEST_RELATIVE_PATH: Final = "work/manifest.json"
OUTPUT_DIRNAME: Final = "output"

#: Where verbatim FFprobe captures are kept, content-hash-addressed, beside the manifest.
PROBE_DIRNAME: Final = "work/ffprobe"

#: The stages `inspect` does not run, and why. Required: `ReportBuilder.build()` refuses
#: to assemble a report with any stage unaccounted for, because a stage that is simply
#: absent is indistinguishable from one nobody remembered (INV-13).
_SKIPPED_STAGES: Final = (
    (StageName.RECONSTRUCT, "`inspect` captures sources; building the timeline is `ingest`"),
    (StageName.ACTIVITY, "`inspect` does not interpret audio content"),
    (StageName.TRANSCRIBE, "`inspect` does not run ASR"),
    (StageName.RENDER, "there is no transcript to render"),
    (StageName.MIX, "`inspect` does not produce audio"),
)


@dataclass(frozen=True, slots=True)
class InspectionResult:
    """What one `inspect` run produced."""

    manifest: Manifest
    manifest_path: Path
    report: IngestReport
    report_path: Path
    exit_code: ExitCode


def run_inspect(
    session_dir: Path,
    *,
    now: dt.datetime | None = None,
    use_cache: bool = True,
) -> InspectionResult:
    """Inspect a session, writing `work/manifest.json` and `output/ingest-report.json`.

    Never raises for an expected failure: a fatal condition becomes a failed `inspect`
    stage, a structured error, a written report, and a nonzero exit code (INV-13).

    Args:
        now: Injectable clock for the report's telemetry. Nothing deterministic reads
            it — passing a different value must not change a manifest byte (INV-03).
        use_cache: Set false to force re-inspection without deleting the cache.
    """
    started_at = now or dt.datetime.now(dt.UTC)
    builder = ReportBuilder(
        session_id=session_dir.name, config_hash="0" * 64, started_at=started_at
    )
    for stage, reason in _SKIPPED_STAGES:
        builder.stage_skipped(stage, reason)

    manifest_path = session_dir / MANIFEST_RELATIVE_PATH
    report_path = session_dir / OUTPUT_DIRNAME / REPORT_FILENAME
    cache = InspectionCache(directory=session_dir / CACHE_DIRNAME, enabled=use_cache)

    try:
        config = load_session_config(session_dir / "session.yaml")
        builder = ReportBuilder(
            session_id=config.session_id,
            config_hash=config_hash(config),
            started_at=started_at,
        )
        for stage, reason in _SKIPPED_STAGES:
            builder.stage_skipped(stage, reason)

        manifest = inspect_session(session_dir, config, cache=cache, builder=builder)
        write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
        cache.commit()
        builder.stage_complete(
            StageName.INSPECT,
            warnings=[
                ReportWarning(code=note.code, message=note.message, path=note.path)
                for note in manifest.warnings
            ],
        )
        builder.add_deliverable(manifest_path, relative_to=session_dir)
    except DndAudioError as exc:
        cache.discard()
        builder.stage_failed(
            StageName.INSPECT,
            [StructuredError(code=exc.code, message=str(exc))],
        )
        report = builder.write(report_path, dt.datetime.now(dt.UTC) if now is None else now)
        return InspectionResult(
            manifest=_empty_manifest(session_dir),
            manifest_path=manifest_path,
            report=report,
            report_path=report_path,
            exit_code=report.exit_code(),
        )

    builder.record_cache(hits=cache.hits, misses=cache.misses)
    report = builder.write(report_path, dt.datetime.now(dt.UTC) if now is None else now)
    return InspectionResult(
        manifest=manifest,
        manifest_path=manifest_path,
        report=report,
        report_path=report_path,
        exit_code=report.exit_code(),
    )


def inspect_session(
    session_dir: Path,
    config: SessionConfig,
    *,
    cache: InspectionCache,
    builder: ReportBuilder,
) -> Manifest:
    """The inspection itself. Raises :class:`DndAudioError` on any fatal condition."""
    roots = _raw_roots(config)
    before = _snapshot(session_dir, roots)
    _reject_outputs_inside_raw(roots)

    tools = tool_versions()
    builder.record_tool_version("ffmpeg", tools.ffmpeg)
    builder.record_tool_version("ffprobe", tools.ffprobe)

    found = discover(session_dir, config)

    tracks: list[ManifestTrack] = []
    for track in found.tracks:
        tracks.append(
            ManifestTrack(
                track_id=track.track_id,
                speaker_id=track.speaker_id,
                speaker_name=track.speaker_name,
                input_path=track.input_path,
                active=track.active,
                inactive_reason=track.inactive_reason,
                sources=[
                    _capture(session_dir, config, source, tools, cache) for source in track.sources
                ],
            )
        )

    unassigned = [
        _capture(session_dir, config, source, tools, cache, probe_it=False)
        for source in found.unassigned
    ]

    _verify_unchanged(session_dir, roots, before)

    roster = _roster_of(found)
    _contribute_to_report(builder, found, roster)

    return Manifest(
        session_id=config.session_id,
        config_hash=config_hash(config),
        inspection=InspectionProvenance(
            ffmpeg_version=tools.ffmpeg,
            ffprobe_version=tools.ffprobe,
            ffprobe_args=list(FFPROBE_ARGS),
            semantics_version=INSPECTION_SEMANTICS_VERSION,
        ),
        roster=roster,
        tracks=tracks,
        unassigned=unassigned,
        warnings=[
            ManifestNote(code=note.code, message=note.message, path=note.path)
            for note in found.warnings
        ],
        decisions=[
            ManifestDecision(code=item.code, subject=item.subject, detail=item.detail)
            for item in found.decisions
        ],
    )


def _capture(
    session_dir: Path,
    config: SessionConfig,
    source: DiscoveredSource,
    tools: ToolVersions,
    cache: InspectionCache,
    *,
    probe_it: bool = True,
) -> ManifestSource:
    """Probe and walk one source, or reuse an identical capture.

    Only selected sources are probed. A duplicate and an ignored `edit` are recorded
    with everything discovery knows and left alone: reading them would cost time to
    learn something no stage will consume.
    """
    item = source.file
    hints = FilenameHintsRecord(
        recognized=item.hints.recognized,
        variant=item.hints.variant,
        tx_label=item.hints.tx_label,
        sequence=item.hints.sequence,
        named_date=item.hints.named_date,
        named_time=item.hints.named_time,
    )
    base = {
        "relative_path": item.relative_path,
        "sha256": item.sha256,
        "size_bytes": item.size_bytes,
        "role": source.role,
        "reason_code": source.reason_code,
        "detail": source.detail,
        "associated_with": source.associated_with,
        "filename": hints,
    }
    if not probe_it or source.role != "selected":
        return ManifestSource(**base)  # type: ignore[arg-type]

    key = cache_key(
        relative_path=item.relative_path,
        source_sha256=item.sha256,
        config_hash=config_hash(config),
        tools=tools,
        ffprobe_args=FFPROBE_ARGS,
    )
    cached = cache.get(key)
    if cached is not None:
        return ManifestSource(**base, **cached)  # type: ignore[arg-type]

    payload = _inspect_one(session_dir, config, source)
    cache.stage(key, {name: value.model_dump(mode="json") for name, value in payload.items()})
    return ManifestSource(**base, **payload)  # type: ignore[arg-type]


def _inspect_one(
    session_dir: Path, config: SessionConfig, source: DiscoveredSource
) -> dict[str, ContainerRecord | ProbeRecord | RiffRecord | StartTimeRecord]:
    """The expensive part: FFprobe, the chunk walk, and the strategy chain."""
    item = source.file
    probe = run_ffprobe(session_dir, item.relative_path)

    # Persisted before parsing, so a capture this code cannot read still survives.
    sidecar = session_dir / PROBE_DIRNAME / probe.sidecar_name
    write_atomic(sidecar, probe.raw)

    document = parse_probe(probe.raw)
    properties = read_audio_properties(document)
    inventory = read_inventory(session_dir / item.relative_path)
    data = inventory.find("data")
    count = exact_sample_count(
        data_size=data.size if data else None,
        block_align=properties.block_align,
        duration_ts=properties.duration_ts,
    )

    start = extract_start_time(
        SourceContext(
            relative_path=item.relative_path,
            sha256=item.sha256,
            sample_rate=properties.sample_rate,
            tags=format_tags(document),
            frame_rate=parse_frame_rate(config.timecode.frame_rate),
            override=config.recovery.source_time_overrides.get(item.relative_path),
        )
    )

    return {
        "container": ContainerRecord(
            codec_name=properties.codec_name,
            sample_format=properties.sample_format,
            bits_per_sample=properties.bits_per_sample,
            sample_rate=properties.sample_rate,
            channels=properties.channels,
            duration_ts=properties.duration_ts,
            time_base=properties.time_base,
            duration_text=properties.duration_text,
            sample_count=count.samples,
            sample_count_source=count.source,
            sample_count_agrees=count.agrees,
        ),
        "probe": ProbeRecord(
            sidecar_path=f"{PROBE_DIRNAME}/{probe.sidecar_name}",
            sha256=probe.sha256,
            command=["ffprobe", *FFPROBE_ARGS, "-i", item.relative_path],
        ),
        "riff": _riff_record(inventory),
        "start_time": _start_time_record(start),
    }


def _riff_record(inventory: RiffInventory) -> RiffRecord:
    return RiffRecord(
        form=inventory.form,
        form_type=inventory.form_type,
        declared_size=inventory.declared_size,
        file_size=inventory.file_size,
        truncated=inventory.truncated,
        chunks=[
            RiffChunkRecord(
                chunk_id=chunk.chunk_id,
                offset=chunk.offset,
                size=chunk.size,
                container=chunk.container,
                sha256=chunk.sha256,
                text=chunk.text,
            )
            for chunk in inventory.chunks
        ],
        warnings=[
            ManifestNote(code=note.code, message=note.message, path=str(note.offset))
            for note in inventory.warnings
        ],
    )


def _start_time_record(start: StartTime) -> StartTimeRecord:
    evidence = start.evidence
    if isinstance(evidence, BwfSampleReference):
        recorded: BwfSampleReferenceRecord | TimecodeRecord | SessionOffsetRecord = (
            BwfSampleReferenceRecord(
                samples=evidence.samples,
                sample_rate=evidence.sample_rate,
                origination_date=evidence.origination_date,
            )
        )
    elif isinstance(evidence, TimecodeReference):
        recorded = TimecodeRecord(
            text=evidence.text,
            frames=evidence.frames,
            frame_rate_label=evidence.frame_rate_label,
            frame_rate=RationalRate(
                numerator=evidence.frame_rate.numerator,
                denominator=evidence.frame_rate.denominator,
            ),
            drop_frame=evidence.drop_frame,
            recording_date=evidence.recording_date,
        )
    else:
        recorded = SessionOffsetRecord(
            samples=evidence.samples,
            sample_rate=evidence.sample_rate,
            recording_date=evidence.recording_date,
        )

    return StartTimeRecord(
        strategy=start.strategy,
        evidence=recorded,
        assumptions=list(start.assumptions),
        declined=[
            DeclinedStrategyRecord(strategy=item.name, reason=item.reason)
            for item in start.declined
        ],
        override_reason=start.override_reason,
    )


def _roster_of(found: Discovery) -> RosterSummary:
    return RosterSummary(
        known_tracks=[track.track_id for track in found.tracks],
        active_tracks=[track.track_id for track in found.tracks if track.active],
        inactive_tracks=[track.track_id for track in found.tracks if not track.active],
        file_counts={track.track_id: len(track.sources) for track in found.tracks},
        missing_directories=list(found.missing_directories),
        empty_directories=list(found.empty_directories),
        extra_directories=list(found.extra_directories),
    )


def _contribute_to_report(builder: ReportBuilder, found: Discovery, roster: RosterSummary) -> None:
    """Everything the report carries that the manifest does not carry for it."""
    builder.record_roster(roster)
    builder.record_package_version("dnd_audio.inspection", str(INSPECTION_SEMANTICS_VERSION))
    for item in found.decisions:
        builder.record_decision(Decision(code=item.code, subject=item.subject, detail=item.detail))


def _raw_roots(config: SessionConfig) -> tuple[str, ...]:
    """The directories a session's sources live under, as session-relative paths.

    Derived from the configured inputs rather than hardcoded to ``raw/``: the spec's
    layout is canonical, not mandatory, and INV-01 protects wherever the sources
    actually are.
    """
    roots = {str(PurePosixPath(track.input).parent) for track in config.tracks}
    return tuple(sorted(root for root in roots if root not in ("", ".")))


def _snapshot(session_dir: Path, roots: tuple[str, ...]) -> dict[str, tuple[str, int]]:
    """Hash and size every file under the raw roots.

    Every file, not only the selected sources: INV-01 says nothing under ``raw/`` is
    written, renamed, deleted, or normalized, and a check that looked only at what
    inspection read would miss exactly the accidental rename it exists to catch.
    """
    snapshot: dict[str, tuple[str, int]] = {}
    for root in roots:
        directory = session_dir / root
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                relative = path.relative_to(session_dir).as_posix()
                snapshot[relative] = (sha256_file(path), path.stat().st_size)
    return snapshot


def _verify_unchanged(
    session_dir: Path, roots: tuple[str, ...], before: dict[str, tuple[str, int]]
) -> None:
    """INV-01, verified rather than asserted."""
    after = _snapshot(session_dir, roots)
    if after == before:
        return

    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    details = []
    if changed:
        details.append(f"modified: {', '.join(changed)}")
    if removed:
        details.append(f"removed: {', '.join(removed)}")
    if added:
        details.append(f"appeared: {', '.join(added)}")
    message = (
        "the session's raw sources changed during inspection, which no stage of this "
        "pipeline is permitted to do (INV-01). " + "; ".join(details)
    )
    raise DiscoveryError(message, code="raw_sources_modified")


def _reject_outputs_inside_raw(roots: tuple[str, ...]) -> None:
    """The spec's "output paths would overwrite raw inputs" fatal error.

    Checked before the first write rather than after, which is the only order in which
    it is a check rather than a postmortem.
    """
    outputs = {
        "the manifest": PurePosixPath(MANIFEST_RELATIVE_PATH),
        "the FFprobe sidecars": PurePosixPath(PROBE_DIRNAME),
        "the inspection cache": PurePosixPath(CACHE_DIRNAME),
        "the report": PurePosixPath(OUTPUT_DIRNAME) / REPORT_FILENAME,
    }
    for label, target in outputs.items():
        for root in roots:
            if target == PurePosixPath(root) or PurePosixPath(root) in target.parents:
                message = (
                    f"{label} would be written to {target}, inside the source directory "
                    f"{root}. Nothing under a session's raw sources may be written to "
                    f"(INV-01); move the tracks' input directories out of {root}."
                )
                raise DiscoveryError(message, code="output_inside_raw")


def _empty_manifest(session_dir: Path) -> Manifest:
    """A placeholder for a run that failed before producing one.

    Never written to disk. It exists so :class:`InspectionResult` has one shape, and a
    caller reading `.manifest` after a failure gets an empty inventory rather than a
    partial one that looks complete.
    """
    return Manifest(
        session_id=session_dir.name,
        config_hash="0" * 64,
        inspection=InspectionProvenance(
            ffmpeg_version="unknown",
            ffprobe_version="unknown",
            semantics_version=INSPECTION_SEMANTICS_VERSION,
        ),
        roster=RosterSummary(),
    )

"""`dnd-audio inspect`: discover, capture, write the manifest, write the report.

The ordering here is load-bearing, so it is stated rather than left to be inferred:

1. **Snapshot every file under the raw roots** before anything else. That snapshot is
   what INV-01 is verified against at the end, and it covers *every* file — including
   ones inspection never selected, because "we did not touch what we did not read" is a
   weaker claim than the invariant makes.
2. **Refuse to run if an output path would land inside a raw root**, comparing
   *resolved* paths so a symlink cannot walk around it. The spec lists this among the
   fatal errors, and it has to be checked before the first write rather than after. When
   the report's own location is the offending one, no report is written: INV-01 outranks
   INV-13 there, because a report is regenerable and a source directory written into is
   not.
3. **Discover, then capture every candidate.** The spec says "for every candidate audio
   file, run `ffprobe`", and that includes the ones nothing will consume: the operator
   asking "why was this ignored" is asking about exactly those. Timing is the one thing
   that differs — missing timing is fatal for a source the pipeline will use (INV-12) and
   recorded as a warning for one it will not.
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
from collections.abc import Sequence
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
from dnd_audio.inspection import (
    INSPECTION_SEMANTICS_VERSION,
    OUTPUT_DIRNAME,
    WORK_DIRNAME,
)
from dnd_audio.inspection.cache import CACHE_DIRNAME, InspectionCache, cache_key
from dnd_audio.inspection.discovery import DiscoveredSource, Discovery, discover
from dnd_audio.inspection.probe import (
    FFPROBE_ARGS,
    AudioProperties,
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

MANIFEST_RELATIVE_PATH: Final = f"{WORK_DIRNAME}/manifest.json"

#: Where verbatim FFprobe captures are kept, content-hash-addressed, beside the manifest.
PROBE_DIRNAME: Final = f"{WORK_DIRNAME}/ffprobe"

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
    #: False only when writing the report would itself have violated INV-01 — see the
    #: carve-out in :func:`run_inspect`. The report object is still built and returned,
    #: so the caller can print the error it carries.
    report_written: bool = True


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
    # No config_hash yet, and `None` rather than a plausible-looking string of zeroes:
    # a run that never resolved a configuration has no configuration hash, and a
    # syntactically valid fabrication is worse than an absence a consumer can branch on.
    builder = ReportBuilder(session_id=session_dir.name, config_hash=None, started_at=started_at)
    for stage, reason in _SKIPPED_STAGES:
        builder.stage_skipped(stage, reason)

    manifest_path = session_dir / MANIFEST_RELATIVE_PATH
    report_path = session_dir / OUTPUT_DIRNAME / REPORT_FILENAME
    cache = InspectionCache(directory=session_dir / CACHE_DIRNAME, read_enabled=use_cache)

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
        builder.stage_complete(StageName.INSPECT, warnings=_report_warnings(manifest))
        builder.add_deliverable(manifest_path, relative_to=session_dir)
    except Exception as exc:
        # Every failure, not only the ones this project raises on purpose. INV-13 says
        # the report is written even on partial failure, and an unreadable source file
        # raising OSError is exactly the case where an operator most needs one — a
        # traceback and no report is the worst of both. A bug reaching here still
        # produces a structured error rather than nothing.
        cache.discard()
        _remove_stale_manifest(manifest_path)
        builder.stage_failed(
            StageName.INSPECT,
            [StructuredError(code=_code_of(exc), message=str(exc) or type(exc).__name__)],
        )
        finished = dt.datetime.now(dt.UTC) if now is None else now
        # The one failure that must not write the report is the one about where writing
        # is allowed. When `output/` resolves inside a track's source directory, writing
        # the failure report there commits the very violation being reported, so INV-01
        # wins over INV-13: nothing is written, the structured error still reaches the
        # caller, and the CLI prints it. INV-13's report is regenerable; a source
        # directory written into is not.
        if isinstance(exc, DiscoveryError) and exc.code == "output_inside_raw":
            return InspectionResult(
                manifest=_empty_manifest(session_dir),
                manifest_path=manifest_path,
                report=builder.build(finished),
                report_path=report_path,
                report_written=False,
                exit_code=ExitCode.FATAL,
            )
        report = builder.write(report_path, finished)
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
    _reject_outputs_inside_raw(session_dir, config, roots)

    tools = tool_versions()
    builder.record_tool_version("ffmpeg", tools.ffmpeg)
    builder.record_tool_version("ffprobe", tools.ffprobe)
    builder.record_command("ffprobe " + " ".join(FFPROBE_ARGS) + " -i <source>")

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
        _capture(session_dir, config, source, tools, cache) for source in found.unassigned
    ]

    _verify_unchanged(session_dir, roots, before)

    roster = _roster_of(found)
    manifest_tracks = tracks
    _contribute_to_report(builder, found, roster, manifest_tracks, unassigned)

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
) -> ManifestSource:
    """Probe and walk one source, or reuse an identical capture.

    **Every candidate is probed**, not only the selected ones. The spec says "for every
    candidate audio file, run `ffprobe` and retain…", and an earlier version skipped
    ignored edits, duplicates, and unassigned files on the grounds that no stage would
    consume them. That reasoning is wrong for a milestone whose product is a description
    of what is there: the operator asking "why was this ignored" is asking precisely
    about a file nothing consumed, and answering needs its container facts.

    Timing is the one thing that differs. A missing start time is fatal for a source the
    pipeline will use (INV-12) and merely worth recording for one it will not — a stray
    WAV in an unconfigured directory must not be able to fail the session.
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
    key = cache_key(
        relative_path=item.relative_path,
        source_sha256=item.sha256,
        config_hash=config_hash(config),
        tools=tools,
        ffprobe_args=FFPROBE_ARGS,
    )
    cached = cache.get(key)
    if cached is not None and _sidecar_exists(session_dir, cached):
        return ManifestSource(**base, **cached)  # type: ignore[arg-type]

    try:
        capture = _inspect_one(session_dir, config, source)
    except DndAudioError as exc:
        if source.role == "selected":
            raise
        # A file the pipeline will not use must not be able to fail the session. One
        # corrupt stray `.wav` in a directory nobody configured is a thing to report,
        # not a reason to refuse to inspect five good transmitters — and probing every
        # candidate would otherwise have *introduced* that failure while fixing a
        # different one.
        return ManifestSource(
            **base,  # type: ignore[arg-type]
            warnings=[
                ManifestNote(
                    code="capture_failed",
                    message=f"could not be inspected, which is recorded rather than fatal "
                    f"because this source is {source.role!r} and no stage will use it: {exc}",
                )
            ],
        )
    cache.stage(key, capture.as_payload())
    return ManifestSource(**base, **capture.as_payload())  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class _Capture:
    """Everything reading one source produced, and the only thing the cache stores.

    Serialized through one method rather than by iterating a dict of models: the fields
    have different shapes, and a generic ``model_dump`` loop over them silently stopped
    working the moment one of them became a list.
    """

    container: ContainerRecord
    probe: ProbeRecord
    riff: RiffRecord
    #: Absent only for a source no stage will use, which carried no timing evidence.
    start_time: StartTimeRecord | None
    warnings: list[ManifestNote]

    def as_payload(self) -> dict[str, object]:
        return {
            "container": self.container.model_dump(mode="json"),
            "probe": self.probe.model_dump(mode="json"),
            "riff": self.riff.model_dump(mode="json"),
            "start_time": None
            if self.start_time is None
            else self.start_time.model_dump(mode="json"),
            "warnings": [note.model_dump(mode="json") for note in self.warnings],
        }


def _sidecar_exists(session_dir: Path, cached: dict[str, object]) -> bool:
    """Whether the verbatim FFprobe capture a cached record points at is still there.

    A cache entry outlives the sidecar it references — deleting ``work/ffprobe/`` while
    keeping ``work/cache/`` is an ordinary thing to do, and without this check the next
    run exits zero having written a manifest whose ``probe.sidecar_path`` names a file
    that does not exist. The gate requires the raw JSON to be *retained*, and a reference
    to a missing file retains nothing. Re-probing costs one subprocess.
    """
    probe = cached.get("probe")
    if not isinstance(probe, dict):
        return False
    sidecar = probe.get("sidecar_path")
    return isinstance(sidecar, str) and (session_dir / sidecar).is_file()


def _inspect_one(session_dir: Path, config: SessionConfig, source: DiscoveredSource) -> _Capture:
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

    context = SourceContext(
        relative_path=item.relative_path,
        sha256=item.sha256,
        sample_rate=properties.sample_rate,
        tags=format_tags(document),
        frame_rate=parse_frame_rate(config.timecode.frame_rate),
        override=config.recovery.source_time_overrides.get(item.relative_path),
    )
    warnings = _format_warnings(properties, count.agrees)
    start: StartTime | None
    if source.role == "selected":
        start = extract_start_time(context)
    else:
        # A file nothing will consume still gets its timing recorded when it has any.
        # What it must not do is fail the session: a stray WAV in a directory nobody
        # configured has no obligation to carry a timecode (INV-12 is about the sources
        # the pipeline uses).
        try:
            start = extract_start_time(context)
        except DndAudioError as exc:
            start = None
            warnings.append(
                ManifestNote(
                    code="no_timing_evidence",
                    message=f"no start time could be established, which is recorded rather "
                    f"than fatal because this source is {source.role!r} and no stage will "
                    f"use it: {exc}",
                )
            )

    return _Capture(
        warnings=warnings,
        container=ContainerRecord(
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
        probe=ProbeRecord(
            sidecar_path=f"{PROBE_DIRNAME}/{probe.sidecar_name}",
            sha256=probe.sha256,
            command=["ffprobe", *FFPROBE_ARGS, "-i", item.relative_path],
        ),
        riff=_riff_record(inventory),
        start_time=None if start is None else _start_time_record(start),
    )


#: What a DJI transmitter recording is expected to be. Anything else is a warning here
#: and a fatal error in M2: the spec lists a non-48 kHz selected source among the fatal
#: errors, but placing it there is timeline construction's job, and refusing to *record*
#: a file we can describe would lose the diagnostic that explains the later failure.
_EXPECTED_SAMPLE_RATE: Final = 48000
_EXPECTED_CODEC: Final = "pcm_f32le"
_EXPECTED_CHANNELS: Final = 1


def _format_warnings(
    properties: AudioProperties, sample_count_agrees: bool | None
) -> list[ManifestNote]:
    """Flag a source that is not the shape the rest of the pipeline assumes."""
    notes: list[ManifestNote] = []
    if properties.sample_rate != _EXPECTED_SAMPLE_RATE:
        notes.append(
            ManifestNote(
                code="unexpected_sample_rate",
                message=f"{properties.sample_rate} Hz, not the {_EXPECTED_SAMPLE_RATE} Hz this "
                f"pipeline mixes at. Resampling a lossless timeline silently is not on offer, "
                f"so building the timeline will reject this source (M2).",
            )
        )
    if properties.codec_name != _EXPECTED_CODEC:
        notes.append(
            ManifestNote(
                code="unexpected_codec",
                message=f"{properties.codec_name}, not {_EXPECTED_CODEC}. Dual-file mode's "
                f"`orig` is 32-bit float; anything else suggests a processed or "
                f"reconverted file (OQ-007).",
            )
        )
    if properties.channels != _EXPECTED_CHANNELS:
        notes.append(
            ManifestNote(
                code="unexpected_channel_count",
                message=f"{properties.channels} channels; a transmitter records one. Two "
                f"suggests a receiver mixdown rather than a transmitter recording.",
            )
        )
    if sample_count_agrees is False:
        notes.append(
            ManifestNote(
                code="sample_count_disagreement",
                message="the RIFF data chunk and FFprobe's duration_ts imply different "
                "sample counts. The data chunk is used; the disagreement is evidence for "
                "OQ-011 and is worth looking at before trusting either.",
            )
        )
    return notes


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


def _contribute_to_report(
    builder: ReportBuilder,
    found: Discovery,
    roster: RosterSummary,
    tracks: Sequence[ManifestTrack],
    unassigned: Sequence[ManifestSource],
) -> None:
    """Everything the report carries that the manifest does not carry for it.

    An applied recovery override is recorded here as well as in the manifest, because the
    gate asks for it "prominently in manifest **and** report" and an earlier version
    satisfied only the first half — with a test whose name claimed both and which never
    opened the report.
    """
    builder.record_roster(roster)
    builder.record_package_version("dnd_audio.inspection", str(INSPECTION_SEMANTICS_VERSION))
    for item in found.decisions:
        builder.record_decision(Decision(code=item.code, subject=item.subject, detail=item.detail))

    for source in [*(s for track in tracks for s in track.sources), *unassigned]:
        start = source.start_time
        if start is None or start.override_reason is None:
            continue
        builder.record_decision(
            Decision(
                code="recovery_override_applied",
                subject=source.relative_path,
                detail=(
                    f"timing came from {start.strategy}, not from the file: {start.override_reason}"
                ),
                details={"strategy": start.strategy, "evidence": start.evidence.kind},
            )
        )


def _raw_roots(config: SessionConfig) -> tuple[str, ...]:
    """The directories a session's sources live under, as session-relative paths.

    Derived from the configured inputs rather than hardcoded to ``raw/``: the spec's
    layout is canonical, not mandatory, and INV-01 protects wherever the sources
    actually are.

    ``"."`` is kept. An earlier version dropped it — reasonably, since every relative
    path is under ``"."`` and the output check would fire on all of them — but dropping
    it also emptied the snapshot, so for a session configured as ``input: "tx-a"`` the
    INV-01 verification compared two empty dicts and passed no matter what happened to
    the sources. The false-positive problem belongs to the output check alone, and is
    handled there; the snapshot excludes this pipeline's own directories by name.
    """
    roots = {str(PurePosixPath(track.input).parent) or "." for track in config.tracks}
    return tuple(sorted("." if root == "" else root for root in roots))


def _snapshot(session_dir: Path, roots: tuple[str, ...]) -> dict[str, tuple[str, int]]:
    """Hash and size every file under the raw roots.

    Every file, not only the selected sources: INV-01 says nothing under ``raw/`` is
    written, renamed, deleted, or normalized, and a check that looked only at what
    inspection read would miss exactly the accidental rename it exists to catch.

    ``work/`` and ``output/`` are excluded, because when a track's input sits directly in
    the session root they are inside a scanned root and are the two directories this run
    is *supposed* to write.
    """
    generated = {WORK_DIRNAME, OUTPUT_DIRNAME}
    snapshot: dict[str, tuple[str, int]] = {}
    for root in roots:
        directory = session_dir if root == "." else session_dir / root
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(session_dir).as_posix()
            if generated.intersection(PurePosixPath(relative).parts):
                continue
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


def _reject_outputs_inside_raw(
    session_dir: Path, config: SessionConfig, roots: tuple[str, ...]
) -> None:
    """The spec's "output paths would overwrite raw inputs" fatal error.

    Checked before the first write rather than after, which is the only order in which
    it is a check rather than a postmortem.

    **Compared after resolution, not lexically.** A lexical comparison is defeated by one
    symlink: with ``output -> raw/tx-a``, ``output/ingest-report.json`` does not look like
    it is inside ``raw/``, and the run cheerfully writes a report into a track's source
    directory. The snapshot cannot catch it either, because the report is written after
    the snapshot is verified. So every candidate path and every protected directory is
    resolved to a real filesystem location first.

    Protected: each configured track's input directory, always; and each scan root, except
    when the root *is* the session directory, where ``work/`` and ``output/`` are
    legitimately siblings of the track directories.
    """
    protected: dict[str, Path] = {
        track.input: _resolve(session_dir / track.input) for track in config.tracks
    }
    for root in roots:
        if root != ".":
            protected[root] = _resolve(session_dir / root)

    outputs = {
        "the manifest": session_dir / MANIFEST_RELATIVE_PATH,
        "the FFprobe sidecars": session_dir / PROBE_DIRNAME,
        "the inspection cache": session_dir / CACHE_DIRNAME,
        "the report": session_dir / OUTPUT_DIRNAME / REPORT_FILENAME,
    }
    for label, target in outputs.items():
        resolved = _resolve(target)
        for name, directory in sorted(protected.items()):
            if resolved == directory or directory in resolved.parents:
                message = (
                    f"{label} would be written to {resolved}, inside the source directory "
                    f"{name} ({directory}). Nothing under a session's raw sources may be "
                    f"written to (INV-01). If a symlink put it there, that counts."
                )
                raise DiscoveryError(message, code="output_inside_raw")


def _resolve(path: Path) -> Path:
    """The real location a path names, following symlinks as far as they exist.

    ``strict=False`` because most of these paths have not been created yet; what matters
    is where they *would* land, and that is decided by the symlinks that already exist on
    the way there.
    """
    return path.resolve()


def _report_warnings(manifest: Manifest) -> list[ReportWarning]:
    """Every warning the run produced, flattened for the report.

    Session-level discovery warnings *and* per-source ones — an unexpected sample rate, a
    sample-count disagreement, a truncated RIFF chunk. Those were previously reachable
    only by opening the manifest and walking into a track's sources, which is not what
    the spec means by a structured report of warnings.
    """
    flattened = [
        ReportWarning(code=note.code, message=note.message, path=note.path)
        for note in manifest.warnings
    ]
    for source in [*(s for t in manifest.tracks for s in t.sources), *manifest.unassigned]:
        flattened.extend(
            ReportWarning(code=note.code, message=note.message, path=source.relative_path)
            for note in source.warnings
        )
        if source.riff is not None:
            flattened.extend(
                ReportWarning(
                    code=f"riff_{note.code}",
                    message=note.message,
                    path=source.relative_path,
                )
                for note in source.riff.warnings
            )
    return sorted(flattened, key=lambda note: (note.code, note.path or "", note.message))


def _remove_stale_manifest(manifest_path: Path) -> None:
    """Delete a manifest left by an earlier successful run.

    A failed run that leaves the previous manifest in place is worse than one that
    leaves none: the file looks current, describes a session that no longer inspects,
    and nothing in it says so. The report records the failure; the stale artifact would
    contradict it.
    """
    manifest_path.unlink(missing_ok=True)


def _code_of(exc: BaseException) -> str:
    """The structured error code for a failure (INV-13).

    Errors this project raises carry their own; anything else is `internal_error`, which
    is honest — an unexpected exception is a bug, and labelling it as one of the known
    failure modes would send an operator looking in the wrong place.
    """
    if isinstance(exc, DndAudioError):
        return exc.code
    return "internal_error"


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

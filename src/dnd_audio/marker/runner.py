"""`dnd-audio marker analyze`: find the marker on every track, and say what it means.

The ordering is load-bearing and stated rather than left to be inferred:

1. **Declare both outputs and refuse any that resolves inside a source directory** — before
   anything is read, written, or deleted (INV-01).
2. **Snapshot every file under the source roots**, once, around the whole run.
3. **Validate the existing artifacts read-only.** This command never rebuilds and never
   rewrites `timeline.json` or `ingest-report.json`; see :mod:`dnd_audio.marker.inputs` for
   why that departure from ADR-0015 has to be paid for by checking more rather than less.
4. **Canonicalize the searched intervals** into a disjoint half-open set, so two overlapping
   configured windows cannot detect one occurrence twice.
5. **Stream each track through the detector** in bounded blocks.
6. **Form occurrences on the reference track first**, then associate other tracks one-to-one
   inside a bounded lag interval. Never by list index — one missed detection would otherwise
   shift every later pairing.
7. **Verify the sources are unchanged** after the last read and before publication.
8. **Write the analysis, then the report**, atomically. Cleanup on failure runs *after* the
   INV-01 carve-out, because on that path the unlink is itself the violation (ADR-0021).

Nothing here corrects anything. The timeline is not touched, no sample moves, and a
start-to-end change is differential acoustic arrival unless the event log asserts fixed
geometry (ADR-0040).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy
import scipy

from dnd_audio.artifacts.manifest import StartEvidenceRecord
from dnd_audio.artifacts.timeline import TimelineTrack
from dnd_audio.config import SessionConfig, load_session_config
from dnd_audio.determinism import sha256_bytes, to_milliseconds, write_json_atomic
from dnd_audio.errors import DndAudioError, ExitCode
from dnd_audio.marker import (
    ANALYSIS_RELATIVE_PATH,
    DEFAULT_WINDOW_SECONDS,
    DETECTOR_SEMANTICS_VERSION,
    MARKER_ANALYSIS_SEMANTICS_VERSION,
    MARKER_REPORT_RELATIVE_PATH,
    MARKER_SAMPLE_RATE,
    MARKER_SEMANTICS_VERSION,
)
from dnd_audio.marker.analysis import (
    AnalysisIdentity,
    ArrivalComparison,
    ArrivalOutcome,
    DetectedOccurrence,
    DetectionOutcome,
    GroupMember,
    OccurrenceGroup,
    SyncMarkerAnalysis,
    TimecodeComparison,
)
from dnd_audio.marker.detect import DetectorThresholds, MarkerOccurrence, detect_occurrences
from dnd_audio.marker.eventlog import MarkerEventLog, load_event_log
from dnd_audio.marker.inputs import SessionArtifacts, read_session_artifacts
from dnd_audio.marker.report import (
    AnalysisStatus,
    MarkerReport,
    MarkerReportError,
    MarkerReportWarning,
    OverallStatus,
    ReportDeliverable,
    write_marker_report,
)
from dnd_audio.marker.spec import MarkerSpec, resolve
from dnd_audio.marker.wav import marker_wav_bytes
from dnd_audio.raw_guard import raw_roots, reject_outputs_inside_raw, snapshot, verify_unchanged
from dnd_audio.timeline.reader import TrackReader
from dnd_audio.timeline.syncqa import offset_floor_samples

__all__ = ["MarkerAnalysisResult", "marker_analyze_outputs", "run_marker_analyze"]


@dataclass(frozen=True, slots=True)
class MarkerAnalysisResult:
    """What the command produced, for the CLI to report and a test to inspect."""

    report: MarkerReport
    report_path: Path
    analysis_path: Path
    report_written: bool
    analysis: SyncMarkerAnalysis | None = None
    exit_code: ExitCode = ExitCode.OK


@dataclass
class _Accumulator:
    """Warnings and errors collected while the run proceeds."""

    warnings: list[MarkerReportWarning] = field(default_factory=list)
    errors: list[MarkerReportError] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def warn(self, code: str, message: str, *, path: str | None = None) -> None:
        self.warnings.append(MarkerReportWarning(code=code, message=message, path=path))
        self.notes.append(f"{code}: {message}")


def marker_analyze_outputs(session_dir: Path) -> dict[str, Path]:
    """Every path this command may write, declared before it writes any of them.

    The signature `raw_guard.reject_outputs_inside_raw` takes, and the reason it takes one: a
    stage that adds an output and forgets to declare it here is the failure this makes
    visible.
    """
    return {
        "the marker analysis": session_dir / ANALYSIS_RELATIVE_PATH,
        "the marker report": session_dir / MARKER_REPORT_RELATIVE_PATH,
    }


def _canonical_intervals(
    windows: list[tuple[int, int]], *, duration: int, halo: int
) -> list[tuple[int, int]]:
    """Merge overlapping half-open windows into a disjoint ascending set.

    Each window is widened by ``halo`` — the marker's own length — so an occurrence whose
    anchor sits near an edge has its whole waveform inside what is read. Without that, an
    event logged a second before the marker actually sounded would be half-scanned and
    silently missed.

    Merging is what stops one occurrence being detected twice when an operator logs two
    overlapping windows, which the charter calls out explicitly.
    """
    widened = sorted(
        (max(0, start - halo), min(duration, end + halo)) for start, end in windows if end > start
    )
    merged: list[tuple[int, int]] = []
    for start, end in widened:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _default_windows(
    duration: int, *, start_seconds: int, end_seconds: int
) -> list[tuple[int, int]]:
    """A window at each end of the session, when no event log says otherwise.

    Each span is clamped to half the session so the two windows cannot swallow the middle and
    turn the default into an accidental whole-session scan — the charter wants that explicit.
    On a session shorter than twice the requested span they meet in the middle, and
    :func:`_canonical_intervals` then merges them into one, which is why
    :func:`_assign_roles` checks the *default windows* rather than counting occurrences.
    """
    opening = min(start_seconds * MARKER_SAMPLE_RATE, max(duration // 2, 1))
    closing = min(end_seconds * MARKER_SAMPLE_RATE, max(duration // 2, 1))
    return [(0, opening), (max(0, duration - closing), duration)]


def _source_coordinate(track: TimelineTrack, sample: int) -> tuple[str | None, int | None]:
    """Where a session sample falls in a real recording, if it falls in one.

    Silence is a real answer — before the transmitter started, inside a gap, or after it
    stopped — and returning ``None`` for it is deliberate: inventing a source position for an
    anchor that landed in a gap would be inventing timing (INV-12).
    """
    for segment in track.segments:
        if segment.kind != "audio":
            continue
        if segment.session_start_sample <= sample < segment.session_end_sample:
            offset = sample - segment.session_start_sample
            return segment.source_relative_path, (segment.source_start_sample or 0) + offset
    return None, None


def _usable(item: MarkerOccurrence) -> bool:
    """Whether an occurrence can supply a trusted integer-sample arrival."""
    return not (item.clipped or item.weak or item.ambiguous)


def _choose_reference(configured: str | None, detections: dict[str, list[MarkerOccurrence]]) -> str:
    """Which track anchors every group.

    Configured wins. Otherwise the track with the **most** accepted occurrences — most
    likely the one nearest the phone, and therefore the one whose detections are sharpest —
    tie-broken lexically so the answer never depends on dictionary order. Stated here rather
    than left implicit because the reference decides the sign of every lag in the analysis.
    """
    if configured is not None:
        return configured
    ranked = sorted(
        detections,
        key=lambda track: (-sum(_usable(item) for item in detections[track]), track),
    )
    return ranked[0]


def _assign_roles(
    groups: list[tuple[int, list[MarkerOccurrence]]],
    log: MarkerEventLog | None,
    spec: MarkerSpec,
    *,
    default_windows: list[tuple[int, int]],
    accumulator: _Accumulator,
) -> list[tuple[str | None, str, str | None]]:
    """``(role, role_source, geometry_id)`` per group, one-to-one against logged events.

    Role assignment is a **second** matching problem, and the charter did not say so. Once
    configured windows are unioned for scanning, one reference occurrence can sit inside both
    a `start` and an `end` interval; canonicalizing the searched set stops it being detected
    twice but does nothing about labelling it arbitrarily.

    So events claim occurrences one-to-one, in playback order, and an occurrence that two
    events could equally claim is left **unassigned** rather than given the nearer one's role.
    Never by peak strength: a louder detection is not a more start-like one.
    """
    assigned: list[tuple[str | None, str, str | None]] = [
        (None, "unassigned", None) for _ in groups
    ]
    if log is None:
        # The one-event-per-default-window rule, checked against the windows themselves
        # rather than against the total count. Those two are not the same thing: on a short
        # session the two default windows merge into one searched interval, and "there
        # happen to be two occurrences somewhere" is not the charter's condition.
        opening, closing = default_windows[0], default_windows[-1]
        inside_open = [
            i for i, (anchor, _) in enumerate(groups) if opening[0] <= anchor < opening[1]
        ]
        inside_close = [
            i for i, (anchor, _) in enumerate(groups) if closing[0] <= anchor < closing[1]
        ]
        if len(inside_open) == 1 and len(inside_close) == 1 and inside_open != inside_close:
            assigned[inside_open[0]] = ("start", "default_window", None)
            assigned[inside_close[0]] = ("end", "default_window", None)
            return assigned
        if groups:
            accumulator.warn(
                "marker_roles_unassigned",
                f"{len(groups)} occurrence(s) were accepted and no event log was supplied. A "
                f"start/end pair is named only when each default window holds exactly one; "
                f"here they hold {len(inside_open)} and {len(inside_close)}. Every occurrence "
                f"is reported and none is labelled — supply --event-log to label them.",
            )
        return assigned

    events = log.for_marker(spec.name)
    if not events:
        accumulator.warn(
            "marker_event_log_names_no_occurrence",
            f"the event log records no playback of {spec.name!r}; every occurrence below is "
            f"unassigned. Analysing a take against a marker it was not recorded with is the "
            f"one confusion the bench's three candidates make easy.",
        )
        return assigned

    claimed: set[int] = set()
    for event in events:
        start, end = event.interval_samples(MARKER_SAMPLE_RATE)
        inside = [
            index
            for index, (anchor, _) in enumerate(groups)
            if start <= anchor < end and index not in claimed
        ]
        if not inside:
            accumulator.warn(
                "marker_event_without_occurrence",
                f"the event log records a {event.role.value} at {event.start_ms} ms and no "
                f"marker was accepted inside that window.",
            )
            continue
        if len(inside) > 1:
            accumulator.warn(
                "marker_event_ambiguous",
                f"the {event.role.value} logged at {event.start_ms} ms contains "
                f"{len(inside)} accepted occurrences, so which one it names is not "
                f"decidable. All are reported and none is labelled.",
            )
            continue
        index = inside[0]
        claimed.add(index)
        assigned[index] = (event.role.value, "event_log", event.geometry_id)
    return assigned


def _associate(
    reference_anchor: int,
    detections: dict[str, list[MarkerOccurrence]],
    reference_track: str,
    *,
    settings: DetectorThresholds,
    used: dict[str, set[int]],
) -> list[GroupMember]:
    """One member per track, one-to-one, inside the association window.

    Greedy by absolute lag, and each occurrence may be claimed once — so a track that heard
    the marker twice close together cannot supply the same arrival to two groups. Two
    candidates inside the window make the member `ambiguous` rather than silently taking the
    nearer: ADR-0041 requires ambiguity to be reported, not resolved.
    """
    members: list[GroupMember] = []
    for track_id, occurrences in sorted(detections.items()):
        if track_id == reference_track:
            found = next(
                (item for item in occurrences if item.anchor_sample == reference_anchor), None
            )
            if found is not None and _usable(found):
                members.append(
                    GroupMember(
                        track_id=track_id,
                        outcome=DetectionOutcome.DETECTED,
                        anchor_sample=found.anchor_sample,
                        relative_lag_samples=0,
                        score_permille=found.score_permille,
                    )
                )
            continue

        nearby = [
            (index, item)
            for index, item in enumerate(occurrences)
            if index not in used.setdefault(track_id, set())
            and abs(item.anchor_sample - reference_anchor) <= settings.association_lag_samples
        ]
        if not nearby:
            members.append(GroupMember(track_id=track_id, outcome=DetectionOutcome.MISSING))
            continue
        if len(nearby) > 1:
            members.append(GroupMember(track_id=track_id, outcome=DetectionOutcome.AMBIGUOUS))
            continue

        index, item = nearby[0]
        used[track_id].add(index)
        if item.ambiguous:
            outcome = DetectionOutcome.AMBIGUOUS
        elif item.clipped:
            outcome = DetectionOutcome.CLIPPED
        elif item.weak:
            outcome = DetectionOutcome.WEAK
        else:
            outcome = DetectionOutcome.DETECTED
        members.append(
            GroupMember(
                track_id=track_id,
                outcome=outcome,
                anchor_sample=item.anchor_sample,
                relative_lag_samples=(
                    item.anchor_sample - reference_anchor
                    if outcome is DetectionOutcome.DETECTED
                    else None
                ),
                score_permille=item.score_permille,
            )
        )
    return members


def _compare_arrival(
    groups: list[OccurrenceGroup],
    *,
    settings: DetectorThresholds,
    accumulator: _Accumulator,
) -> list[ArrivalComparison]:
    """Start-to-end change per track, and what ADR-0040 allows it to mean.

    A change is `clock_drift_evidence` **only** when both groups carry the same non-null
    geometry ID — the operator's written assertion that the phone and every transmitter
    stayed put. Otherwise it is differential acoustic arrival, and says why.
    """
    start = next((group for group in groups if group.role == "start"), None)
    end = next((group for group in groups if group.role == "end"), None)
    if start is None or end is None:
        return []

    fixed = (
        start.geometry_id is not None
        and end.geometry_id is not None
        and start.geometry_id == end.geometry_id
    )
    by_track = {member.track_id: member for member in start.members}
    comparisons: list[ArrivalComparison] = []

    for member in end.members:
        opening = by_track.get(member.track_id)
        if (
            opening is None
            or opening.relative_lag_samples is None
            or member.relative_lag_samples is None
        ):
            comparisons.append(
                ArrivalComparison(
                    track_id=member.track_id,
                    start_lag_samples=opening.relative_lag_samples if opening else None,
                    end_lag_samples=member.relative_lag_samples,
                    outcome=ArrivalOutcome.INCONCLUSIVE,
                    detail=(
                        "one end of the pair has no clean detection on this track, so there "
                        "is no change to report. The analyzer does not fabricate a lag "
                        "because the report has a field for one."
                    ),
                )
            )
            continue

        change = member.relative_lag_samples - opening.relative_lag_samples
        if fixed:
            outcome = ArrivalOutcome.CLOCK_DRIFT_EVIDENCE
            detail = (
                f"the event log asserts geometry {start.geometry_id!r} for both ends, so the "
                f"phone and this transmitter did not move between them. A change of "
                f"{change} samples is therefore evidence about the recorders' clocks. No "
                f"correction was applied: drift correction is post-MVP (INV-12)."
            )
            if abs(change) >= settings.material_arrival_change_samples:
                accumulator.warn(
                    "marker_material_clock_drift",
                    f"{member.track_id} changed by {change} samples under asserted fixed "
                    f"geometry {start.geometry_id!r}, meeting the material "
                    f"{settings.material_arrival_change_samples}-sample threshold. This is "
                    f"recorder-drift evidence only; no correction was applied.",
                )
        else:
            outcome = ArrivalOutcome.DIFFERENTIAL_ARRIVAL
            detail = (
                f"this track's arrival moved by {change} samples between the two markers. "
                f"Geometry is not asserted unchanged, so this is differential acoustic "
                f"arrival and **not** clock drift: a wearer leaning back moves the acoustic "
                f"term by milliseconds, which is the same size as the drift being looked for "
                f"(ADR-0040)."
            )
        comparisons.append(
            ArrivalComparison(
                track_id=member.track_id,
                start_lag_samples=opening.relative_lag_samples,
                end_lag_samples=member.relative_lag_samples,
                change_samples=change,
                change_ms=to_milliseconds(Fraction(change, MARKER_SAMPLE_RATE)),
                outcome=outcome,
                detail=detail,
            )
        )
    return comparisons


def _compare_timecode(
    groups: list[OccurrenceGroup],
    evidence: tuple[StartEvidenceRecord, ...],
    config: SessionConfig,
) -> list[TimecodeComparison]:
    """Measured lag against what the metadata predicted, with M8's quantization floor.

    The timeline has *already* placed every track by its own timing evidence, so after
    placement two tracks hearing one sound should show zero relative lag apart from
    propagation. A nonzero measured lag therefore **is** the disagreement, and the only
    question is whether it exceeds what this session's own evidence could express.

    The floor comes from `syncqa.offset_floor_samples` rather than from a constant, for the
    reason M8 recorded: a receiver set to 60 fps still wrote 1600-sample boundaries, so
    deriving the threshold from the configured frame rate reinstates the false alarm it
    exists to remove. A matched filter resolves single samples; a healthy within-one-quantum
    offset must not become a failed jam because the instrument improved.
    """
    floor = offset_floor_samples(evidence, config, rate=MARKER_SAMPLE_RATE)

    first = next((group for group in groups if group.role in ("start", None)), None)
    if first is None:
        return []
    return [
        TimecodeComparison(
            track_id=member.track_id,
            measured_lag_samples=member.relative_lag_samples or 0,
            predicted_lag_samples=0,
            disagreement_samples=abs(member.relative_lag_samples or 0),
            quantum_floor_samples=floor,
            beyond_quantum=abs(member.relative_lag_samples or 0) > floor,
        )
        for member in first.members
        if member.relative_lag_samples is not None
    ]


def run_marker_analyze(
    session_dir: Path,
    *,
    marker: str | None = None,
    reference_track: str | None = None,
    event_log: Path | None = None,
    start_window_seconds: int = DEFAULT_WINDOW_SECONDS,
    end_window_seconds: int = DEFAULT_WINDOW_SECONDS,
    thresholds: DetectorThresholds | None = None,
) -> MarkerAnalysisResult:
    """Find the marker on every track and write the analysis and the report.

    The two window spans apply only when no event log is supplied: a log states its own
    intervals, and silently widening them would search audio the operator did not ask about.
    """
    started = dt.datetime.now(dt.UTC)
    settings = thresholds if thresholds is not None else DetectorThresholds()
    accumulator = _Accumulator()

    analysis_path = session_dir / ANALYSIS_RELATIVE_PATH
    report_path = session_dir / MARKER_REPORT_RELATIVE_PATH

    try:
        config = load_session_config(session_dir / "session.yaml")
    except DndAudioError as exc:
        return _failed(
            session_dir, started, exc, analysis_path=analysis_path, report_path=report_path
        )

    roots = raw_roots(config)
    try:
        reject_outputs_inside_raw(session_dir, config, roots, marker_analyze_outputs(session_dir))
    except DndAudioError:
        # INV-01 outranks INV-13: writing the failure report here would commit the very
        # violation being reported, and nothing is unlinked either (ADR-0021).
        return MarkerAnalysisResult(
            report=_skeleton(config.session_id, started, marker, failed=True),
            report_path=report_path,
            analysis_path=analysis_path,
            report_written=False,
            exit_code=ExitCode.FATAL,
        )

    before = snapshot(session_dir, roots)

    try:
        spec = resolve(marker)
        artifacts = read_session_artifacts(session_dir, config)
        log = load_event_log(event_log) if event_log is not None else None
        analysis = _analyze(
            session_dir,
            config,
            spec,
            artifacts,
            log,
            reference_track,
            settings,
            (start_window_seconds, end_window_seconds),
            accumulator,
        )
        verify_unchanged(session_dir, roots, before)
    except DndAudioError as exc:
        analysis_path.unlink(missing_ok=True)
        return _failed(
            session_dir,
            started,
            exc,
            analysis_path=analysis_path,
            report_path=report_path,
            session_id=config.session_id,
            marker=marker,
            warnings=accumulator.warnings,
        )

    write_json_atomic(analysis_path, analysis.model_dump(mode="json"))

    payload = analysis_path.read_bytes()
    report = MarkerReport(
        session_id=config.session_id,
        marker_name=spec.name,
        overall_status=OverallStatus.COMPLETE,
        analysis_status=AnalysisStatus.COMPLETE,
        inconclusive=not analysis.conclusive,
        occurrences_found=len(analysis.occurrences),
        groups_formed=len(analysis.groups),
        warnings=accumulator.warnings,
        deliverables=[
            ReportDeliverable(
                relative_path=ANALYSIS_RELATIVE_PATH,
                sha256=sha256_bytes(payload),
                size_bytes=len(payload),
            )
        ],
        started_at=started,
        finished_at=dt.datetime.now(dt.UTC),
    )
    write_marker_report(report, report_path)
    return MarkerAnalysisResult(
        report=report,
        report_path=report_path,
        analysis_path=analysis_path,
        report_written=True,
        analysis=analysis,
        exit_code=report.exit_code(),
    )


def _analyze(
    session_dir: Path,
    config: SessionConfig,
    spec: MarkerSpec,
    artifacts: SessionArtifacts,
    log: MarkerEventLog | None,
    configured_reference: str | None,
    settings: DetectorThresholds,
    windows_seconds: tuple[int, int],
    accumulator: _Accumulator,
) -> SyncMarkerAnalysis:
    """Detect, group, compare. Every read is bounded; nothing is written here."""
    if min(windows_seconds) < 1:
        message = (
            f"--start-window-s and --end-window-s must each be at least 1 second, got "
            f"{windows_seconds[0]} and {windows_seconds[1]}. A zero-length window would "
            f"report `marker_not_found` while never having looked."
        )
        raise DndAudioError(message, code="invalid_search_window")

    timeline = artifacts.timeline
    duration = timeline.duration_samples
    windows = (
        [event.interval_samples(MARKER_SAMPLE_RATE) for event in log.for_marker(spec.name)]
        if log is not None
        else []
    )
    defaults = _default_windows(
        duration, start_seconds=windows_seconds[0], end_seconds=windows_seconds[1]
    )
    intervals = _canonical_intervals(
        windows or defaults, duration=duration, halo=spec.total_samples
    )

    usable = [track for track in timeline.tracks if track.segments]
    detections: dict[str, list[MarkerOccurrence]] = {}
    for track in usable:
        found: list[MarkerOccurrence] = []
        with TrackReader(session_dir, track, duration) as reader:
            for interval in intervals:
                found.extend(
                    detect_occurrences(reader, spec, interval=interval, thresholds=settings)
                )
        detections[track.track_id] = sorted(found, key=lambda item: item.anchor_sample)

    if not detections:
        accumulator.warn(
            "marker_no_usable_track",
            "no track in this session has any audio, so there was nothing to search.",
        )

    reference = _choose_reference(configured_reference, detections) if detections else ""
    if configured_reference is not None and configured_reference not in detections:
        message = (
            f"--reference-track {configured_reference!r} is not a track with audio in this session"
        )
        raise DndAudioError(message, code="unknown_reference_track")

    by_track = {track.track_id: track for track in timeline.tracks}
    occurrences = [
        _describe_occurrence(by_track[track_id], item)
        for track_id, items in sorted(detections.items())
        for item in items
    ]

    reference_hits = [item for item in detections.get(reference, []) if _usable(item)]
    if not reference_hits and detections:
        accumulator.warn(
            "marker_not_found",
            "no unambiguous usable marker sequence was accepted on any track inside the "
            "searched windows. That is a measurement about the room, not a failure of the "
            "command.",
        )

    raw_groups = [(item.anchor_sample, [item]) for item in reference_hits]
    roles = _assign_roles(raw_groups, log, spec, default_windows=defaults, accumulator=accumulator)

    used: dict[str, set[int]] = {}
    groups: list[OccurrenceGroup] = []
    for index, ((anchor, _), (role, source, geometry)) in enumerate(
        zip(raw_groups, roles, strict=True)
    ):
        groups.append(
            OccurrenceGroup(
                group_index=index,
                reference_anchor_sample=anchor,
                role=role,
                role_source=source,  # type: ignore[arg-type]
                geometry_id=geometry,
                members=_associate(anchor, detections, reference, settings=settings, used=used),
            )
        )

    unmatched = [
        _describe_occurrence(by_track[track_id], item)
        for track_id, items in sorted(detections.items())
        if track_id != reference
        for index, item in enumerate(items)
        if index not in used.get(track_id, set())
    ]

    identity = AnalysisIdentity(
        marker_semantics_version=MARKER_SEMANTICS_VERSION,
        detector_semantics_version=DETECTOR_SEMANTICS_VERSION,
        marker_analysis_semantics_version=MARKER_ANALYSIS_SEMANTICS_VERSION,
        marker_name=spec.name,
        marker_wav_sha256=sha256_bytes(marker_wav_bytes(spec)),
        timeline_schema_version=timeline.schema_version,
        event_log_schema_version=log.schema_version if log is not None else None,
        event_log_sha256=log.digest() if log is not None else None,
        config_hash=artifacts.config_hash,
        manifest_sha256=artifacts.manifest_sha256,
        timeline_config_hash=timeline.config_hash,
        reference_track=reference or "(none)",
        thresholds=settings.identity(),
        searched_intervals=intervals,
        numpy_version=numpy.__version__,
        scipy_version=scipy.__version__,
    )

    return SyncMarkerAnalysis(
        session_id=config.session_id,
        identity=identity,
        occurrences=occurrences,
        groups=groups,
        unmatched=unmatched,
        timecode=_compare_timecode(groups, artifacts.start_evidence(), config),
        arrival=_compare_arrival(groups, settings=settings, accumulator=accumulator),
        notes=accumulator.notes,
    )


def _describe_occurrence(track: TimelineTrack, item: MarkerOccurrence) -> DetectedOccurrence:
    path, sample = _source_coordinate(track, item.anchor_sample)
    return DetectedOccurrence(
        track_id=track.track_id,
        anchor_sample=item.anchor_sample,
        anchor_ms=to_milliseconds(Fraction(item.anchor_sample, MARKER_SAMPLE_RATE)),
        score_permille=item.score_permille,
        runner_up_permille=item.runner_up_permille,
        gap_errors_samples=list(item.gap_errors_samples),
        source_relative_path=path,
        source_sample=sample,
        clipped=item.clipped,
        weak=item.weak,
        ambiguous=item.ambiguous,
    )


def _skeleton(
    session_id: str, started: dt.datetime, marker: str | None, *, failed: bool
) -> MarkerReport:
    """A report for a run that could not produce one on disk."""
    return MarkerReport(
        session_id=session_id,
        marker_name=marker,
        overall_status=OverallStatus.FAILED if failed else OverallStatus.COMPLETE,
        analysis_status=AnalysisStatus.FAILED if failed else AnalysisStatus.COMPLETE,
        errors=(
            [
                MarkerReportError(
                    code="output_inside_raw",
                    message=(
                        "an output path resolves inside this session's own sources, so "
                        "nothing was written — INV-01 outranks INV-13 here, because a "
                        "report is regenerable and a source recording is not."
                    ),
                )
            ]
            if failed
            else []
        ),
        started_at=started,
        finished_at=dt.datetime.now(dt.UTC),
    )


def _failed(
    session_dir: Path,
    started: dt.datetime,
    exc: DndAudioError,
    *,
    analysis_path: Path,
    report_path: Path,
    session_id: str | None = None,
    marker: str | None = None,
    warnings: list[MarkerReportWarning] | None = None,
) -> MarkerAnalysisResult:
    """Write the failure report atomically. INV-13's whole point."""
    report = MarkerReport(
        session_id=session_id or session_dir.name,
        marker_name=marker,
        overall_status=OverallStatus.FAILED,
        analysis_status=AnalysisStatus.FAILED,
        errors=[MarkerReportError(code=exc.code, message=str(exc))],
        warnings=warnings or [],
        started_at=started,
        finished_at=dt.datetime.now(dt.UTC),
    )
    write_marker_report(report, report_path)
    return MarkerAnalysisResult(
        report=report,
        report_path=report_path,
        analysis_path=analysis_path,
        report_written=True,
        exit_code=ExitCode.FATAL,
    )

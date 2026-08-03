"""`dnd-audio activity`: who was speaking, written to `work/activity.json`.

The spec calls `activity` the shared cached operation that `transcribe`, `mix`, and
`process` all invoke. ADR-0015 makes it a command as well, because a milestone whose only
demonstration is a test is exactly the work that only appears done — and because M4 and M5
then call this function rather than reimplementing the composition.

The ordering is load-bearing and is stated rather than left to be inferred:

1. **Snapshot every file under the raw roots, and refuse outputs that would land inside
   them.** Once, around the whole composed run — inspection, reconstruction, and attribution
   together — so the sources are hashed once rather than three times (INV-01).
2. **Rebuild the timeline.** Every run, never validated-and-reused. M2 settled the
   equivalent question for the manifest: a configuration-hash match is not evidence that an
   artifact still describes what is on disk, because a replaced file keeps every hash
   internally consistent. Warm, it costs no FFprobe and no resampling.
3. **Detect per track, from cache where possible.** The detector belongs to one track and
   sees contiguous windows in order; a recurrent model's state is only meaningful under that
   contract (ADR-0013).
4. **Attribute across tracks, from cache where possible.** A separate identity, because
   tuning a bleed threshold must not re-run inference (ADR-0016).
5. **Verify INV-01, then commit every cache, then write the artifacts and one report.**
   Whichever way it went (INV-13), with the same carve-out `inspect` and `ingest` have: when
   the report's own location resolves inside a source directory, nothing is written and
   INV-01 wins.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np
import scipy

from dnd_audio.activity import (
    ACTIVITY_RELATIVE_PATH,
    ACTIVITY_SEMANTICS_VERSION,
    ATTRIBUTION_DIRNAME,
    DETECTION_DIRNAME,
    DETECTOR_FRAME_SAMPLES,
)
from dnd_audio.activity.band import load_speech_band_filter
from dnd_audio.activity.bleed import (
    Attribution,
    AttributionResult,
    CandidateInput,
    attribute,
)
from dnd_audio.activity.cache import (
    AttributionCache,
    DetectionCache,
    attribution_identity,
    detection_identity,
    probability_relative_path,
)
from dnd_audio.activity.detect import SpeechRegion, detect_track
from dnd_audio.artifacts.activity import (
    ActivityCandidate,
    ActivityDecision,
    ActivityGraph,
    ActivityNote,
    ActivityProvenance,
    ActivityTrack,
    CandidateEvidence,
    DetectorIdentity,
    candidate_id,
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
from dnd_audio.artifacts.timeline import DerivativeRecord, Timeline, TimelineTrack
from dnd_audio.config import (
    SessionConfig,
    config_hash,
    load_session_config,
    stage_config_hash,
)
from dnd_audio.determinism import sha256_file, write_json_atomic
from dnd_audio.errors import DiscoveryError, DndAudioError, ExitCode
from dnd_audio.inspection import OUTPUT_DIRNAME
from dnd_audio.interfaces import ActivityDetector
from dnd_audio.raw_guard import raw_roots, reject_outputs_inside_raw, snapshot, verify_unchanged
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE, TIMELINE_RELATIVE_PATH
from dnd_audio.timeline.reader import DEFAULT_WINDOW_SAMPLES, DerivativeReader
from dnd_audio.timeline.resample import to_derivative_interval, to_source_sample
from dnd_audio.timeline.runner import build_timeline, ingest_outputs

__all__ = [
    "ActivityResult",
    "ActivityWork",
    "DetectorBundle",
    "activity_outputs",
    "perform_activity",
    "remove_activity_artifacts",
    "run_activity",
]

#: The stages `activity` does not run, and why (INV-13).
_SKIPPED_STAGES: Final = (
    (StageName.TRANSCRIBE, "`activity` stops before ASR; the transcript branch is M4"),
    (StageName.RENDER, "there is no transcript to render"),
    (StageName.MIX, "`activity` produces the graph the mix consumes, not the mix"),
)

#: One second of derivative audio per read. Small enough that six tracks cost megabytes,
#: large enough that per-window overhead is irrelevant (INV-07).
DEFAULT_DETECT_WINDOW: Final = DERIVATIVE_SAMPLE_RATE


class _Committable(Protocol):
    """A staged cache: publish it, or drop it. Every cache in this project is one.

    A protocol rather than a base class so `ActivityWork` can hold M1's, M2's and M3's caches
    in one tuple without any of them knowing about the others.
    """

    def commit(self) -> int: ...

    def discard(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DetectorBundle:
    """A detector implementation, its identity, and how to build one per track.

    The identity is needed *before* any detector is constructed, because it is part of the
    cache key that decides whether one has to be constructed at all. Bundling the two means
    a caller cannot supply a factory whose identity describes a different detector — which
    would serve one model's answers under another's key (INV-08).
    """

    identity: DetectorIdentity
    make: Callable[[str], ActivityDetector]
    runtime_version: str | None = None


@dataclass(frozen=True, slots=True)
class ActivityWork:
    """The activity stages' results, with every cache **staged and uncommitted**.

    What `perform_activity` returns so a longer run can compose it (M4's `transcribe` does).
    The caller owns the INV-01 verification and the commit: an entry may only be published
    once the sources it was computed from have been re-checked, and a composed run has one
    place where that is true for everything at once.
    """

    graph: ActivityGraph
    timeline: Timeline
    timeline_sha256: str
    caches: tuple[_Committable, ...]

    def commit(self) -> None:
        """Publish every staged cache entry. Call only after INV-01 has been re-verified."""
        for cache in self.caches:
            cache.commit()

    def discard(self) -> None:
        """Drop everything staged. The data files remain, and without sidecars are inert."""
        for cache in self.caches:
            cache.discard()


@dataclass(frozen=True, slots=True)
class ActivityResult:
    """What one `activity` run produced."""

    graph: ActivityGraph | None
    graph_path: Path
    timeline: Timeline | None
    report: IngestReport
    report_path: Path
    exit_code: ExitCode
    #: False only when writing the report would itself have violated INV-01.
    report_written: bool = True


def activity_outputs(session_dir: Path) -> dict[str, Path]:
    """Everything `activity` writes, for the INV-01 output check.

    A superset of `ingest`'s, because a composed run performs both. Declared as data so that
    adding an output and forgetting to protect it is a visible omission from one list.
    """
    return {
        **ingest_outputs(session_dir),
        "the activity graph": session_dir / ACTIVITY_RELATIVE_PATH,
        "the detection cache": session_dir / DETECTION_DIRNAME,
        "the attribution cache": session_dir / ATTRIBUTION_DIRNAME,
    }


def run_activity(
    session_dir: Path,
    *,
    detector: DetectorBundle | None = None,
    now: dt.datetime | None = None,
    use_cache: bool = True,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
    detect_window_samples: int = DEFAULT_DETECT_WINDOW,
) -> ActivityResult:
    """Reconstruct the session, detect speech, and attribute it.

    Never raises for an expected failure: a fatal condition becomes a failed stage, a
    structured error, a written report, and a nonzero exit code (INV-13).

    Args:
        detector: The detector to use. Defaults to the pinned Silero model, which is
            resolved — and can fail for an absent or wrong-hashed artifact — at the point
            detection starts, not at import.
        use_cache: False re-detects and re-attributes without deleting anything.
        window_samples: Bound on every 48 kHz read while the timeline is built (INV-07).
        detect_window_samples: Bound on every 16 kHz read while detecting.
    """
    started_at = now or dt.datetime.now(dt.UTC)
    graph_path = session_dir / ACTIVITY_RELATIVE_PATH
    report_path = session_dir / OUTPUT_DIRNAME / REPORT_FILENAME
    timeline_path = session_dir / TIMELINE_RELATIVE_PATH
    builder = _builder(session_dir.name, None, started_at)
    timeline: Timeline | None = None

    try:
        config = load_session_config(session_dir / "session.yaml")
        builder = _builder(config.session_id, config_hash(config), started_at)
        roots = raw_roots(config)
        before = snapshot(session_dir, roots)
        reject_outputs_inside_raw(session_dir, config, roots, activity_outputs(session_dir))

        work = perform_activity(
            session_dir,
            config,
            builder=builder,
            detector=detector,
            use_cache=use_cache,
            window_samples=window_samples,
            detect_window_samples=detect_window_samples,
        )
        timeline = work.timeline
        graph = work.graph

        # Verify first, publish second — every cache from this composed run commits at one
        # moment, after INV-01 has been re-checked. A run that correctly failed on a changed
        # source must not leave an entry keyed on the bytes it read (M2's closeout).
        verify_unchanged(session_dir, roots, before)
        work.commit()

        write_json_atomic(graph_path, graph.model_dump(mode="json"))
        builder.stage_complete(StageName.RECONSTRUCT, warnings=_notes(timeline.warnings))
        builder.add_deliverable(timeline_path, relative_to=session_dir)
        builder.stage_complete(StageName.ACTIVITY, warnings=_notes(graph.warnings))
        builder.add_deliverable(graph_path, relative_to=session_dir)
    except Exception as exc:
        # Every failure, not only the ones raised on purpose: an operator whose run died on
        # an OSError needs a report more than anyone.
        #
        remove_activity_artifacts(session_dir)
        error = StructuredError(code=_code_of(exc), message=str(exc) or type(exc).__name__)
        for stage in (StageName.INSPECT, StageName.RECONSTRUCT, StageName.ACTIVITY):
            if not builder.recorded(stage):
                builder.stage_failed(stage, [error])
        finished = dt.datetime.now(dt.UTC) if now is None else now
        if isinstance(exc, DiscoveryError) and exc.code == "output_inside_raw":
            # INV-01 outranks INV-13 here: writing the failure report would commit the very
            # violation being reported. A report is regenerable; a source directory written
            # into is not.
            return ActivityResult(
                graph=None,
                graph_path=graph_path,
                timeline=timeline,
                report=builder.build(finished),
                report_path=report_path,
                report_written=False,
                exit_code=ExitCode.FATAL,
            )
        report = builder.write(report_path, finished)
        return ActivityResult(
            graph=None,
            graph_path=graph_path,
            timeline=timeline,
            report=report,
            report_path=report_path,
            exit_code=report.exit_code(),
        )

    report = builder.write(report_path, dt.datetime.now(dt.UTC) if now is None else now)
    return ActivityResult(
        graph=graph,
        graph_path=graph_path,
        timeline=timeline,
        report=report,
        report_path=report_path,
        exit_code=report.exit_code(),
    )


def perform_activity(
    session_dir: Path,
    config: SessionConfig,
    *,
    builder: ReportBuilder,
    detector: DetectorBundle | None = None,
    use_cache: bool = True,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
    detect_window_samples: int = DEFAULT_DETECT_WINDOW,
) -> ActivityWork:
    """Reconstruct, detect and attribute, leaving every cache staged.

    The composable half of `activity`, so a longer run — `transcribe`, and `process` in M5 —
    performs these stages exactly once and exactly the way the `activity` command does, rather
    than reimplementing the composition beside it (ADR-0015's argument, one milestone later).

    What it deliberately does **not** do: snapshot `raw/`, verify it, commit anything, write
    the graph, or write a report. Those belong to whoever owns the whole run, because INV-01
    verification has to happen once, around everything, and a cache entry may only be
    committed after it.

    `timeline.json` *is* written here, before attribution, because the attribution cache key
    is keyed on its hash and so it has to exist first. A failed run therefore has to remove it
    — see :func:`remove_activity_artifacts`.

    Raises:
        Exception: any fatal condition, for the caller to turn into a failed stage and a
            report (INV-13).
    """
    timeline_path = session_dir / TIMELINE_RELATIVE_PATH
    build = build_timeline(
        session_dir,
        config,
        builder=builder,
        use_cache=use_cache,
        window_samples=window_samples,
    )
    write_json_atomic(timeline_path, build.timeline.model_dump(mode="json"))
    timeline_sha256 = sha256_file(timeline_path)

    bundle = detector or _silero_bundle(config)
    graph, caches = _attribute(
        session_dir,
        config,
        build.timeline,
        bundle=bundle,
        builder=builder,
        use_cache=use_cache,
        detect_window_samples=detect_window_samples,
        timeline_sha256=timeline_sha256,
    )
    return ActivityWork(
        graph=graph,
        timeline=build.timeline,
        timeline_sha256=timeline_sha256,
        caches=(
            build.inspection_cache,
            build.derivative_cache,
            caches.detection,
            caches.attribution,
        ),
    )


def remove_activity_artifacts(session_dir: Path) -> None:
    """Delete the graph and the timeline a failed run may have left behind.

    Both, not only the graph. `timeline.json` is written *before* attribution, so a run that
    failed during attribution had already overwritten it — and leaving it there publishes a
    timeline the report simultaneously calls `reconstruct: failed` and does not hash as a
    deliverable, which M4 and M5 both read (INV-13). A stale artifact that looks current is
    worse than none: the file describes attributions that no longer hold and nothing in it
    says so.
    """
    (session_dir / ACTIVITY_RELATIVE_PATH).unlink(missing_ok=True)
    (session_dir / TIMELINE_RELATIVE_PATH).unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _Caches:
    """The two activity caches, staged and committed together with everything else."""

    detection: DetectionCache
    attribution: AttributionCache

    def commit(self) -> None:
        self.detection.commit()
        self.attribution.commit()


def _attribute(
    session_dir: Path,
    config: SessionConfig,
    timeline: Timeline,
    *,
    bundle: DetectorBundle,
    builder: ReportBuilder,
    use_cache: bool,
    detect_window_samples: int,
    timeline_sha256: str,
) -> tuple[ActivityGraph, _Caches]:
    """Detect on every track, then decide across them. Raises on any fatal condition."""
    caches = _Caches(
        detection=DetectionCache(session_dir=session_dir, read_enabled=use_cache),
        attribution=AttributionCache(session_dir=session_dir, read_enabled=use_cache),
    )
    band = load_speech_band_filter()
    detection_scope = stage_config_hash(config, "detection")
    attribution_scope = stage_config_hash(config, "attribution")

    # Paired with its derivative rather than filtered and looked up again: a track and the
    # audio it was detected on have to travel together, and re-deriving the record would
    # make "usable" and "what was read" two facts that can disagree.
    usable = [
        (track, record)
        for track in timeline.tracks
        if track.segments and (record := _derivative(track)) is not None
    ]
    detectable = {track.track_id for track, _ in usable}
    warnings = [
        ActivityNote(
            code="activity_track_skipped",
            message=(
                f"{track.track_id} has no 16 kHz working audio, so nothing could be detected "
                f"on it. Re-run `dnd-audio ingest` if this is unexpected."
            ),
            path=track.track_id,
        )
        for track in timeline.tracks
        if track.track_id not in detectable
    ]

    detections: dict[str, tuple[SpeechRegion, ...]] = {}
    keys: dict[str, str] = {}
    frames: dict[str, tuple[int, bool]] = {}
    for track, record in usable:
        key = detection_identity(
            track_id=track.track_id,
            derivative_cache_key=record.cache_key,
            detector=bundle.identity,
            stage_config_hash=detection_scope,
        )
        keys[track.track_id] = key
        found = caches.detection.get(key)
        if found is None:
            result = detect_track(
                session_dir / record.relative_path,
                track_id=track.track_id,
                detector=bundle.make(track.track_id),
                settings=config.activity.vad,
                window_samples=detect_window_samples,
            )
            found = caches.detection.publish(key, result)
        detections[track.track_id] = found.regions
        frames[track.track_id] = (found.frame_count, found.from_detector)

    attribution_key = attribution_identity(
        detection_keys=sorted(keys.values()),
        timeline_sha256=timeline_sha256,
        speech_band_identity=band.identity,
        stage_config_hash=attribution_scope,
    )
    builder.record_cache(hits=caches.detection.hits, misses=caches.detection.misses)
    builder.record_package_version("dnd_audio.activity", str(ACTIVITY_SEMANTICS_VERSION))
    _record_detector(builder, bundle)

    cached = caches.attribution.get(attribution_key)
    builder.record_cache(hits=caches.attribution.hits, misses=caches.attribution.misses)
    if cached is not None:
        _record_decisions(builder, cached)
        return cached, caches

    candidates = _candidates(timeline, detections)
    with DerivativeReader(session_dir, derivative_paths(timeline)) as audio:
        decided = attribute(candidates, read=audio.read, config=config.activity)

    graph = _graph(
        config,
        timeline,
        candidates,
        decided,
        bundle=bundle,
        band_name=band.name,
        band_identity=band.identity,
        keys=keys,
        frames=frames,
        attribution_key=attribution_key,
        timeline_sha256=timeline_sha256,
        warnings=warnings,
    )
    caches.attribution.publish(attribution_key, graph)
    _record_decisions(builder, graph)
    return graph, caches


def _candidates(
    timeline: Timeline, detections: dict[str, tuple[SpeechRegion, ...]]
) -> list[CandidateInput]:
    """Place every detected region on the session grid, in canonical order.

    A region is clamped to the session's aligned duration: the derivative is `ceil`-padded
    to a whole number of output samples, so its last frame can reach past the audio it
    describes, and a candidate that did would fail the artifact's own bounds check.

    Both directions go through M2's helpers rather than being spelled out again here. They
    agreed when this was written, which is exactly why a second copy is dangerous: the
    floor/ceil asymmetry on the way back is the documented trap (rounding both ends the same
    way costs a word its first phoneme), and INV-04 names a second conversion as how that
    rule dies.
    """
    decimation = timeline.sample_rate // DERIVATIVE_SAMPLE_RATE
    found: list[CandidateInput] = []
    for track_id, regions in sorted(detections.items()):
        for region in regions:
            start = to_source_sample(region.start_sample, decimation)
            end = min(to_source_sample(region.end_sample, decimation), timeline.duration_samples)
            if end <= start:
                continue
            derivative_start, derivative_end = to_derivative_interval(start, end, decimation)
            found.append(
                CandidateInput(
                    track_id=track_id,
                    start_sample=start,
                    end_sample=end,
                    derivative_start_sample=derivative_start,
                    derivative_end_sample=derivative_end,
                    probability_permille=region.probability_permille,
                    peak_probability_permille=max(
                        region.peak_probability_permille, region.probability_permille
                    ),
                )
            )
    return sorted(found, key=lambda item: (item.start_sample, item.track_id))


def _graph(
    config: SessionConfig,
    timeline: Timeline,
    candidates: list[CandidateInput],
    decided: AttributionResult,
    *,
    bundle: DetectorBundle,
    band_name: str,
    band_identity: str,
    keys: dict[str, str],
    frames: dict[str, tuple[int, bool]],
    attribution_key: str,
    timeline_sha256: str,
    warnings: list[ActivityNote],
) -> ActivityGraph:
    """Assemble the frozen document from the decisions (ADR-0012)."""
    ids = [candidate_id(item.track_id, item.start_sample) for item in candidates]

    tracks = [
        ActivityTrack(
            track_id=track.track_id,
            speaker_id=track.speaker_id,
            speaker_name=track.speaker_name,
            detection_cache_key=keys[track.track_id],
            probability_relative_path=probability_relative_path(keys[track.track_id]),
            probability_frames=frames[track.track_id][0],
            frame_samples=DETECTOR_FRAME_SAMPLES,
            speech_reference_mbfs=decided.speech_references.get(track.track_id),
        )
        for track in timeline.tracks
        if track.track_id in keys
    ]

    # Only for a track that found *something*. A track with no candidates at all has nothing
    # to veto, and warning that its veto is inactive would put six lines of noise in front of
    # an operator whose session simply had no speech in it.
    found = {candidate.track_id for candidate in candidates}
    notes = list(warnings)
    notes.extend(
        ActivityNote(
            code="no_speech_reference",
            message=(
                f"{track.track_id} has fewer than "
                f"{config.activity.bleed.min_reference_candidates} candidates, so no speech "
                f"reference could be estimated and its bleed veto is inactive. A reference "
                f"from one or two regions is as likely to measure bleed as speech (OQ-017)."
            ),
            path=track.track_id,
        )
        for track in tracks
        if track.speech_reference_mbfs is None and track.track_id in found
    )

    return ActivityGraph(
        session_id=config.session_id,
        config_hash=config_hash(config),
        timeline_sha256=timeline_sha256,
        attribution_cache_key=attribution_key,
        provenance=ActivityProvenance(
            activity_semantics_version=ACTIVITY_SEMANTICS_VERSION,
            timeline_semantics_version=timeline.provenance.timeline_semantics_version,
            inspection_semantics_version=timeline.provenance.inspection_semantics_version,
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            onnxruntime_version=bundle.runtime_version,
            detector=bundle.identity,
            speech_band_filter_name=band_name,
            speech_band_filter_identity=band_identity,
        ),
        sample_rate=timeline.sample_rate,
        derivative_sample_rate=DERIVATIVE_SAMPLE_RATE,
        duration_samples=timeline.duration_samples,
        tracks=tracks,
        candidates=[
            _candidate(candidates[found.index], ids, found) for found in decided.attributions
        ],
        warnings=notes,
        decisions=_decisions(candidates, ids, decided.attributions),
    )


def _candidate(source: CandidateInput, ids: list[str], found: Attribution) -> ActivityCandidate:
    return ActivityCandidate(
        candidate_id=ids[found.index],
        track_id=source.track_id,
        start_sample=source.start_sample,
        end_sample=source.end_sample,
        derivative_start_sample=source.derivative_start_sample,
        derivative_end_sample=source.derivative_end_sample,
        probability_permille=source.probability_permille,
        peak_probability_permille=source.peak_probability_permille,
        band_level_mbfs=found.band_level_mbfs,
        relative_level_mb=found.relative_level_mb,
        score_permille=found.terms.total_permille,
        score_level_permille=found.terms.level_permille,
        score_confidence_permille=found.terms.confidence_permille,
        score_dominance_permille=found.terms.dominance_permille,
        score_correlation_permille=found.terms.correlation_permille,
        decision=found.decision,
        ambiguous=found.ambiguous,
        suppressed_by_candidate_id=None
        if found.suppressed_by is None
        else ids[found.suppressed_by],
        evidence=[
            CandidateEvidence(
                other_candidate_id=ids[item.other_index],
                other_track_id=_track_of(item.other_index, ids),
                overlap_start_sample=item.overlap_start_sample,
                overlap_end_sample=item.overlap_end_sample,
                compared_derivative_samples=item.compared_derivative_samples,
                correlation_permille=item.correlation_permille,
                lag_derivative_samples=item.lag_derivative_samples,
                score_margin_permille=item.score_margin_permille,
                level_delta_mb=item.level_delta_mb,
                outcome=item.outcome,
            )
            for item in found.evidence
        ],
    )


def _track_of(index: int, ids: list[str]) -> str:
    """Recover a track id from a candidate id, which encodes it by construction."""
    return ids[index].removeprefix("cand_").rsplit("_", 1)[0]


def _decisions(
    candidates: list[CandidateInput], ids: list[str], attributions: tuple[Attribution, ...]
) -> list[ActivityDecision]:
    """One auditable record per suppression and per veto-only retention."""
    found: list[ActivityDecision] = []
    for item in attributions:
        source = candidates[item.index]
        if item.suppressed_by is not None:
            winner = ids[item.suppressed_by]
            found.append(
                ActivityDecision(
                    code="bleed_suppressed",
                    subject=ids[item.index],
                    detail=(
                        f"{source.track_id} at sample {source.start_sample} was suppressed in "
                        f"favour of {winner}: it scored {item.terms.total_permille}/1000, the "
                        f"other track's candidate scored more by the configured margin, the "
                        f"two signals correlate, and this track's own level sits far below "
                        f"what its wearer speaks at."
                    ),
                )
            )
        elif item.ambiguous:
            found.append(
                ActivityDecision(
                    code="bleed_vetoed",
                    subject=ids[item.index],
                    detail=(
                        f"{source.track_id} at sample {source.start_sample} was kept although "
                        f"another track scored better and the two correlate: its level is "
                        f"within the veto of what this wearer normally speaks at, so this is "
                        f"more likely simultaneous speech than bleed (ADR-0014)."
                    ),
                )
            )
    return found


def _silero_bundle(config: SessionConfig) -> DetectorBundle:
    """The pinned Silero detector, resolved from the local model store.

    Imported lazily so that neither `onnxruntime` nor a model file is needed to import this
    module — the default test suite drives a scripted detector and must stay free of both
    (INV-05).
    """
    from dnd_audio.activity.silero import silero_bundle

    return silero_bundle(silence_threshold=config.activity.vad.silence_threshold)


def derivative_paths(timeline: Timeline) -> dict[str, str]:
    """Each track's 16 kHz derivative, session-relative, for :class:`DerivativeReader`.

    Only tracks that have one. A track with no derivative has nothing to detect on and
    nothing to transcribe from, and both stages report that as a warning rather than reading
    a file that is not there.
    """
    return {
        track.track_id: record.relative_path
        for track in timeline.tracks
        if (record := _derivative(track)) is not None
    }


def _derivative(track: TimelineTrack) -> DerivativeRecord | None:
    """This track's 16 kHz record, or ``None`` when it has none."""
    for record in track.derivatives:
        if record.sample_rate == DERIVATIVE_SAMPLE_RATE:
            return record
    return None


def _record_detector(builder: ReportBuilder, bundle: DetectorBundle) -> None:
    """Put the detector's identity in the report, as INV-08 and the gate require."""
    identity = bundle.identity
    builder.record_model_identity(
        "vad", identity.commit or identity.variant_digest or identity.name
    )
    if identity.model_sha256 is not None:
        builder.record_model_identity("vad_sha256", identity.model_sha256)
    if bundle.runtime_version is not None:
        builder.record_tool_version("onnxruntime", bundle.runtime_version)


def _record_decisions(builder: ReportBuilder, graph: ActivityGraph) -> None:
    """Copy the graph's decisions into the report, with the numbers behind them.

    The gate requires the scoring function's diagnostics to be visible in
    `ingest-report.json`, and prose is not a diagnostic: an operator asking why a speaker
    vanished needs the four terms, the correlation, the lag, and the level difference — the
    same values the graph carries, in the artifact a human opens first.
    """
    scored = {candidate.candidate_id: candidate for candidate in graph.candidates}
    for decision in graph.decisions:
        candidate = scored.get(decision.subject)
        builder.record_decision(
            Decision(
                code=decision.code,
                subject=decision.subject,
                detail=decision.detail,
                details={} if candidate is None else _diagnostics(candidate),
            )
        )


def _first_vetoed(candidate: ActivityCandidate) -> CandidateEvidence | None:
    """The comparison the veto overrode, for a candidate nothing suppressed."""
    return next(
        (item for item in candidate.evidence if item.outcome == "vetoed_by_track_level"), None
    )


def _diagnostics(candidate: ActivityCandidate) -> dict[str, str]:
    """Every number that produced one attribution, as report details.

    Strings because that is what the report's `details` is: a consumer reads them, and
    keeping them integral in the graph while rendering them here means the two cannot drift
    into disagreeing about a rounding.
    """
    details = {
        "score_permille": str(candidate.score_permille),
        "score_level_permille": str(candidate.score_level_permille),
        "score_confidence_permille": str(candidate.score_confidence_permille),
        "score_dominance_permille": str(candidate.score_dominance_permille),
        "score_correlation_permille": str(candidate.score_correlation_permille),
        "band_level_mbfs": str(candidate.band_level_mbfs),
        "relative_level_mb": (
            "unknown" if candidate.relative_level_mb is None else str(candidate.relative_level_mb)
        ),
        "track_id": candidate.track_id,
    }
    decisive = next(
        (
            item
            for item in candidate.evidence
            if item.other_candidate_id == candidate.suppressed_by_candidate_id
        ),
        # A retained-but-ambiguous candidate has no suppressor; the record that nearly was
        # one is the one worth showing, and it is the only vetoed pair by construction.
        _first_vetoed(candidate),
    )
    if decisive is not None:
        details.update(
            {
                "against_candidate_id": decisive.other_candidate_id,
                "correlation_permille": str(decisive.correlation_permille),
                "lag_derivative_samples": str(decisive.lag_derivative_samples),
                "score_margin_permille": str(decisive.score_margin_permille),
                "level_delta_mb": str(decisive.level_delta_mb),
                "outcome": decisive.outcome,
            }
        )
    return details


def _builder(session_id: str, hash_: str | None, started_at: dt.datetime) -> ReportBuilder:
    builder = ReportBuilder(session_id=session_id, config_hash=hash_, started_at=started_at)
    for stage, reason in _SKIPPED_STAGES:
        builder.stage_skipped(stage, reason)
    return builder


class _Note(Protocol):
    """What the report needs from a warning, whichever artifact it came from.

    `TimelineNote` and `ActivityNote` are separate models on purpose — each belongs to the
    document that carries it — and this is the one place both are flattened into the report.
    A protocol rather than a union so a third artifact's note needs no edit here. Declared
    read-only — properties rather than attributes — because a bare annotation would make the
    protocol *settable*, and a frozen pydantic model does not satisfy that.
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

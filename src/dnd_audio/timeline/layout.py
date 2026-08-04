"""Order a track's chunks, preserve its gaps, and resolve its overlaps (ADR-0010).

Once :mod:`~dnd_audio.timeline.origin` has given every chunk a position, a track is just a
sequence of intervals — and the only interesting question is what to do where two of them
disagree.

**Nothing here is relative to the previous chunk.** Each chunk is placed by its own
evidence, so a transmitter switched off and back on cannot pull later audio earlier: the
hole becomes an explicit silence segment and everything after it stays where its own
timecode says. Placing chunks by accumulating durations is the bug this shape prevents,
and it is the one the spec calls out by name.

**A gap and an overlap are the same comparison** — this chunk's start against the previous
chunk's end — read in two directions. A gap is preserved. An overlap within the
quantization tolerance is resolved by moving the later chunk, never by trimming it; a
larger one is governed by `timecode.chunk_overlap_policy`, whose default refuses rather
than guessing.

The refusals live here too, and they run **before** any of the above. A 44.1 kHz source or
a track whose chunks disagree about their sample rate cannot be placed on a 48 kHz grid
without resampling a lossless path silently, which is not on offer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dnd_audio.artifacts.manifest import Manifest, ManifestSource, ManifestTrack
from dnd_audio.artifacts.timeline import TimelineDecision, TimelineNote, TimelineSegment
from dnd_audio.config import SessionConfig
from dnd_audio.errors import DndAudioError
from dnd_audio.timecode import FrameRate, parse_frame_rate
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE
from dnd_audio.timeline.origin import SessionOrigin, SourceStart, selected_sources
from dnd_audio.timeline.pcm import ACCEPTED_CODECS, ACCEPTED_FORMATS, refusal_reason
from dnd_audio.timeline.rasterize import quantization_tolerance_samples

__all__ = [
    "REQUIRED_CHANNELS",
    "LayoutError",
    "TrackLayout",
    "build_layout",
    "reject_unusable_sources",
]

#: A transmitter records one channel. Two suggests a receiver mixdown.
REQUIRED_CHANNELS = 1


class LayoutError(DndAudioError):
    """A source cannot be placed on the session timeline."""

    default_code = "timeline_layout_failed"


@dataclass(frozen=True, slots=True)
class TrackLayout:
    """One track's segment map, before it becomes an artifact."""

    track_id: str
    speaker_id: str
    speaker_name: str
    start_sample: int
    end_sample: int
    segments: tuple[TimelineSegment, ...]
    warnings: tuple[TimelineNote, ...] = ()


@dataclass
class _Notes:
    """Collected while laying out, emitted once."""

    decisions: list[TimelineDecision] = field(default_factory=list)
    warnings: list[TimelineNote] = field(default_factory=list)


def reject_unusable_sources(manifest: Manifest) -> None:
    """Refuse a session whose selected sources cannot be placed, before placing any.

    The spec lists both of these among the fatal errors, and M1 recorded them as warnings
    on purpose: refusing to *describe* a file we can read would have lost the diagnostic
    that explains this failure. Here is where the refusal belongs.

    Checked in this order, because the first gives a better message than the second when
    both apply. A track holding a 44.1 kHz chunk and a 48 kHz chunk is a capture-procedure
    problem; a track that is entirely 44.1 kHz is a settings problem.

    Raises:
        LayoutError: naming the source, what is wrong with it, and — for a rate
            disagreement — every rate the track's chunks claim.
    """
    for track in manifest.tracks:
        sources = [source for source in track.sources if source.role == "selected"]
        _reject_inconsistent_rates(track, sources)
        for source in sources:
            _reject_unusable(track, source)


def _reject_inconsistent_rates(track: ManifestTrack, sources: list[ManifestSource]) -> None:
    rates = {source.container.sample_rate for source in sources if source.container is not None}
    if len(rates) <= 1:
        return
    detail = ", ".join(
        f"{source.relative_path} at {source.container.sample_rate} Hz"
        for source in sorted(sources, key=lambda s: s.relative_path)
        if source.container is not None
    )
    message = (
        f"track {track.track_id} has chunks that disagree about their sample rate "
        f"({detail}). One person's recording cannot be reconstructed from pieces sampled "
        f"at different rates without resampling part of a lossless path, so this is fatal "
        f"before the timeline is built."
    )
    raise LayoutError(message, code="inconsistent_sample_rate")


def _reject_unusable(track: ManifestTrack, source: ManifestSource) -> None:
    container = source.container
    if container is None:
        message = (
            f"{source.relative_path} is selected for track {track.track_id} but was never "
            f"successfully inspected, so nothing is known about its format. Re-run "
            f"`dnd-audio inspect`; if it still fails, the file cannot be used."
        )
        raise LayoutError(message, code="source_not_inspected")

    if container.sample_rate != CANONICAL_SAMPLE_RATE:
        message = (
            f"{source.relative_path} is {container.sample_rate} Hz, and this pipeline's "
            f"working path is {CANONICAL_SAMPLE_RATE} Hz. Silently resampling a lossless "
            f"timeline is not on offer; re-record at {CANONICAL_SAMPLE_RATE} Hz, or "
            f"exclude the track."
        )
        raise LayoutError(message, code="unsupported_sample_rate")

    if container.channels != REQUIRED_CHANNELS:
        message = (
            f"{source.relative_path} is {container.channels}-channel, and a transmitter "
            f"records one. Two suggests a receiver mixdown rather than a transmitter "
            f"recording, which is a different file than the one this track needs."
        )
        raise LayoutError(message, code="undecodable_source")

    if container.codec_name not in ACCEPTED_CODECS:
        accepted = ", ".join(fmt.codec_name for fmt in ACCEPTED_FORMATS)
        message = (
            f"{source.relative_path} is {container.codec_name}, and the working path "
            f"reads {accepted} — every format that converts to float32 exactly "
            f"(ADR-0030). {refusal_reason(container.codec_name)}"
        )
        raise LayoutError(message, code="undecodable_source")

    if container.sample_count is None:
        message = (
            f"{source.relative_path} has no established sample count, so where it ends is "
            f"unknown and the next chunk's start cannot be validated against it. Timing is "
            f"never invented (INV-12)."
        )
        raise LayoutError(message, code="unknown_sample_count")


def build_layout(
    manifest: Manifest, config: SessionConfig, origin: SessionOrigin
) -> tuple[tuple[TrackLayout, ...], tuple[TimelineDecision, ...], tuple[TimelineNote, ...]]:
    """Turn placed starts into per-track segment maps.

    Assumes :func:`reject_unusable_sources` has already run — it is a separate function so
    the runner can call it before anything is built, which is what "fails *before* timeline
    construction" means.
    """
    frame_rate = parse_frame_rate(config.timecode.frame_rate)
    lengths = {
        source.relative_path: source.container.sample_count
        for _, source in selected_sources(manifest)
        if source.container is not None and source.container.sample_count is not None
    }

    notes = _Notes()
    layouts = tuple(
        _lay_out_track(track, origin.by_track(track.track_id), lengths, config, frame_rate, notes)
        for track in manifest.tracks
    )
    return layouts, tuple(notes.decisions), tuple(notes.warnings)


def _lay_out_track(
    track: ManifestTrack,
    starts: tuple[SourceStart, ...],
    lengths: dict[str, int],
    config: SessionConfig,
    frame_rate: FrameRate,
    notes: _Notes,
) -> TrackLayout:
    """One track: sort by parsed start time, then walk the boundaries.

    Sorted by *time*, not by filename. DJI's `MIC###` counter is a secondary hint at best
    (OQ-003) and INV-12 forbids deriving timing from a name; the tie-break on path exists
    only so two chunks claiming the identical start produce a stable order rather than
    depending on dictionary iteration.
    """
    ordered = sorted(starts, key=lambda item: (item.session_start_sample, item.relative_path))
    if not ordered:
        return TrackLayout(
            track_id=track.track_id,
            speaker_id=track.speaker_id,
            speaker_name=track.speaker_name,
            start_sample=0,
            end_sample=0,
            segments=(),
        )

    segments: list[TimelineSegment] = []
    previous: SourceStart | None = None
    placed_end = ordered[0].session_start_sample
    track_start = placed_end

    for item in ordered:
        evidence_start = item.session_start_sample
        placed = evidence_start

        if previous is not None:
            if evidence_start > placed_end:
                segments.append(
                    TimelineSegment(
                        kind="silence",
                        session_start_sample=placed_end,
                        n_samples=evidence_start - placed_end,
                    )
                )
                notes.decisions.append(
                    TimelineDecision(
                        code="chunk_gap_preserved",
                        subject=item.relative_path,
                        detail=(
                            f"{evidence_start - placed_end} samples of silence precede this "
                            f"chunk, because the previous one ended at {placed_end} and this "
                            f"one's own evidence starts at {evidence_start}. The transmitter "
                            f"was not recording in between."
                        ),
                    )
                )
            elif evidence_start < placed_end:
                placed = _resolve_overlap(
                    item, previous, placed_end, evidence_start, config, frame_rate, notes
                )

        length = lengths[item.relative_path]
        segments.append(
            TimelineSegment(
                kind="audio",
                session_start_sample=placed,
                n_samples=length,
                source_relative_path=item.relative_path,
                source_sha256=item.sha256,
                source_start_sample=0,
                evidence_start_sample=evidence_start,
                shift_samples=placed - evidence_start,
            )
        )
        placed_end = placed + length
        previous = item

    return TrackLayout(
        track_id=track.track_id,
        speaker_id=track.speaker_id,
        speaker_name=track.speaker_name,
        start_sample=track_start,
        end_sample=placed_end,
        segments=tuple(segments),
    )


def _resolve_overlap(
    item: SourceStart,
    previous: SourceStart,
    placed_end: int,
    evidence_start: int,
    config: SessionConfig,
    frame_rate: FrameRate,
    notes: _Notes,
) -> int:
    """Two chunks claim the same samples. Move the later one, or refuse.

    Never trim. The spec's sentence ends "rather than silently discarding audio", and a
    trimmed head is discarded audio whether or not anyone notices.
    """
    overlap = placed_end - evidence_start
    tolerance = quantization_tolerance_samples(
        previous.evidence,
        item.evidence,
        frame_rate,
        CANONICAL_SAMPLE_RATE,
        bwf_quantum_samples=config.timecode.bwf_reference_quantum_samples,
    )

    if overlap <= tolerance:
        notes.decisions.append(
            TimelineDecision(
                code="chunk_overlap_quantization",
                subject=item.relative_path,
                detail=(
                    f"it overlapped the previous chunk by {overlap} sample(s), within the "
                    f"{tolerance}-sample tolerance the two chunks' evidence allows, so it "
                    f"was moved to start immediately after. No audio was trimmed."
                ),
            )
        )
        return placed_end

    if config.timecode.chunk_overlap_policy == "reject":
        message = (
            f"{item.relative_path} overlaps the previous chunk on track {item.track_id} by "
            f"{overlap} samples, which exceeds the {tolerance}-sample tolerance their "
            f"timing evidence allows. That is a real disagreement, not rounding. Set "
            f"timecode.chunk_overlap_policy to 'nudge_later' to place it immediately after "
            f"the previous chunk — no audio is discarded either way — or correct the "
            f"source's timing with a recovery override."
        )
        raise LayoutError(message, code="chunk_overlap")

    notes.warnings.append(
        TimelineNote(
            code="chunk_overlap_nudged",
            message=(
                f"overlapped the previous chunk by {overlap} samples, far beyond the "
                f"{tolerance}-sample quantization tolerance, and was moved later under "
                f"chunk_overlap_policy: nudge_later. Everything after it on this track "
                f"carries the same shift, so this track's alignment against the others now "
                f"differs from its own timecode by that much."
            ),
            path=item.relative_path,
        )
    )
    notes.decisions.append(
        TimelineDecision(
            code="chunk_overlap_nudged",
            subject=item.relative_path,
            detail=(
                f"moved {overlap} samples later so it starts immediately after the previous "
                f"chunk, under chunk_overlap_policy: nudge_later. Every sample is kept."
            ),
        )
    )
    return placed_end

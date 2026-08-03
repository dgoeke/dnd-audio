"""From a track's 16 kHz audio to padded, merged speech regions.

The detector answers one question per frame — *is someone talking here* — and this module
turns a sequence of those answers into the regions M4 will transcribe and M5 will mix. The
turning is where the judgement lives, and every step of it is configurable because none of
the numbers are knowable in advance (OQ-017).

**Everything happens on frames, whatever the detector is.** Silero has a 512-sample frame
and no choice about it; a scripted detector has spans and no frames at all. Rather than two
assemblers, a span-based detector's answer is *rasterized* onto the same frame grid by
coverage — a frame half inside a span reads 500 per-mille — and one assembler runs over
both. That keeps the fake and the real detector on the same code path, which is the only way
the fake proves anything about the real one.

**Hysteresis, not a threshold.** A single threshold cuts a word in half every time a
probability wobbles across it mid-syllable. Speech opens above `speech_threshold` and closes
below `silence_threshold`, and the gap between them is what stops a candidate flickering.

**The five reshaping steps are separate on purpose**, because they answer different
questions and a real session will want them tuned independently:

1. merge across a dip shorter than `min_silence_ms` — that is a stop consonant, not a turn;
2. drop what is left shorter than `min_speech_ms` — that is a cough, not a word;
3. merge across a gap shorter than `merge_gap_ms` — one sentence, not eight fragments;
4. pad both ends by `pad_ms` so the region does not clip the word it contains;
5. merge anything padding just made overlap, so a track's candidates stay disjoint.

Collapsing 1 and 3 into one knob looks like a simplification and removes the ability to say
"keep fragments apart for the detector, join them for the transcriber".

**Memory is bounded by the window, not the session** (INV-07). Audio is read in bounded
windows and handed straight to the detector; the only thing that grows with the session is
the per-frame probability array, at two bytes per 32 ms — under a megabyte for four hours,
and the artifact that makes a bad result debuggable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from dnd_audio.activity import (
    DETECTOR_FRAME_SAMPLES,
    PERMILLE,
    to_permille,
    to_permille_array,
)
from dnd_audio.config import VadConfig
from dnd_audio.interfaces import ActivityDetector, AudioWindow, SpeechSpan
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE
from dnd_audio.timeline.pcm import PcmReader, open_pcm

__all__ = [
    "PERMILLE",
    "DetectionResult",
    "FrameProbabilities",
    "SpeechRegion",
    "assemble_regions",
    "detect_track",
    "frame_count",
    "rasterize_spans",
]


@runtime_checkable
class FrameProbabilities(Protocol):
    """A detector that can report the per-frame probabilities behind its spans.

    Optional, and deliberately *not* part of :class:`~dnd_audio.interfaces.ActivityDetector`.
    The protocol INV-10 froze returns candidates, which is the model-independent thing every
    detector has; per-frame probabilities are an artifact of detectors that happen to have
    frames. A detector without them still works here — its spans are rasterized instead —
    and the graph records which of the two it was, so a reader of the probability file knows
    whether they are looking at a measurement or at a reconstruction.
    """

    def frame_probabilities(self) -> npt.NDArray[np.uint16]: ...


@dataclass(frozen=True, slots=True)
class SpeechRegion:
    """One assembled candidate, in **derivative** samples, half-open."""

    start_sample: int
    end_sample: int
    #: Mean and peak per-mille over the frames the region covers.
    probability_permille: int
    peak_probability_permille: int

    @property
    def n_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """What one track's detection pass produced."""

    track_id: str
    regions: tuple[SpeechRegion, ...]
    #: One per-mille value per frame, for the whole track. Persisted beside the cache entry.
    frame_probabilities: npt.NDArray[np.uint16]
    #: Whether the probabilities came from the detector or were rasterized from its spans.
    from_detector: bool

    @property
    def frame_samples(self) -> int:
        return DETECTOR_FRAME_SAMPLES


def frame_count(n_samples: int) -> int:
    """Frames covering ``n_samples`` derivative samples.

    `ceil`, matching the resampler's own length rule: a partial final frame is zero-padded
    rather than dropped, because dropping it would silently lose the last 32 ms of every
    track whose length is not a multiple of the frame.
    """
    return -(-n_samples // DETECTOR_FRAME_SAMPLES)


def rasterize_spans(
    spans: tuple[SpeechSpan, ...], *, n_frames: int, offset_samples: int = 0
) -> npt.NDArray[np.uint16]:
    """Turn spans into per-frame probabilities by **coverage**, not by containment.

    A frame is credited with the fraction of it a span covers, scaled by that span's own
    probability. Containment would round every region outward to whole frames and make a
    scripted fixture's carefully chosen sample positions unrecoverable; coverage puts a
    boundary frame at the value that decides it — half covered reads 500, which is exactly
    the threshold's own tie point.

    Args:
        spans: In derivative samples, on the same timeline as ``offset_samples``.
        n_frames: Length of the returned array.
        offset_samples: Where frame zero of this array sits on that timeline.
    """
    covered = np.zeros(n_frames, dtype=np.float64)
    for span in spans:
        start = max(span.start_sample - offset_samples, 0)
        end = min(span.end_sample - offset_samples, n_frames * DETECTOR_FRAME_SAMPLES)
        if end <= start:
            continue
        first = start // DETECTOR_FRAME_SAMPLES
        last = (end - 1) // DETECTOR_FRAME_SAMPLES
        for frame in range(first, last + 1):
            edge = frame * DETECTOR_FRAME_SAMPLES
            overlap = min(end, edge + DETECTOR_FRAME_SAMPLES) - max(start, edge)
            covered[frame] += span.probability * overlap / DETECTOR_FRAME_SAMPLES
    return to_permille_array(covered * PERMILLE)


def detect_track(
    derivative_path: Path,
    *,
    track_id: str,
    detector: ActivityDetector,
    settings: VadConfig,
    window_samples: int,
) -> DetectionResult:
    """Run one detector over one track's derivative and assemble its candidates.

    The detector is handed contiguous, in-order windows and belongs to this track alone —
    a recurrent detector's state is only meaningful under that contract, and
    :class:`~dnd_audio.activity.silero.SileroActivityDetector` raises when it is broken
    rather than quietly resetting (ADR-0013).

    Args:
        window_samples: Bound on every read (INV-07). Rounded up to a whole number of
            frames, because a window that splits a frame would make the frame grid depend
            on the window size — and then so would the answer.
    """
    per_window = max(1, -(-window_samples // DETECTOR_FRAME_SAMPLES)) * DETECTOR_FRAME_SAMPLES
    source = open_pcm(derivative_path)
    if source.sample_rate != DERIVATIVE_SAMPLE_RATE:
        message = (
            f"{derivative_path} is at {source.sample_rate} Hz; activity detection runs on "
            f"the {DERIVATIVE_SAMPLE_RATE} Hz derivative"
        )
        raise ValueError(message)

    total = source.n_samples
    n_frames = frame_count(total)
    rasterized = np.zeros(n_frames, dtype=np.uint16)

    with PcmReader(source) as handle:
        position = 0
        while position < total:
            length = min(per_window, total - position)
            window = AudioWindow(
                track_id=track_id,
                sample_rate=DERIVATIVE_SAMPLE_RATE,
                start_sample=position,
                samples=handle.read(position, length),
            )
            spans = detector.detect(window)
            if spans:
                first = position // DETECTOR_FRAME_SAMPLES
                covered = frame_count(position + length) - first
                chunk = rasterize_spans(
                    spans, n_frames=covered, offset_samples=first * DETECTOR_FRAME_SAMPLES
                )
                rasterized[first : first + covered] = np.maximum(
                    rasterized[first : first + covered], chunk
                )
            position += length

    if isinstance(detector, FrameProbabilities):
        probabilities = detector.frame_probabilities()
        from_detector = True
    else:
        probabilities = rasterized
        from_detector = False
    if probabilities.shape[0] != n_frames:
        message = (
            f"the detector reported {probabilities.shape[0]} frames for {track_id}, but "
            f"{total} derivative samples is {n_frames} frames. A probability array that "
            f"does not cover the track would silence whatever it is short by."
        )
        raise ValueError(message)

    return DetectionResult(
        track_id=track_id,
        regions=assemble_regions(probabilities, settings=settings, n_samples=total),
        frame_probabilities=probabilities,
        from_detector=from_detector,
    )


def assemble_regions(
    probabilities: npt.NDArray[np.uint16], *, settings: VadConfig, n_samples: int
) -> tuple[SpeechRegion, ...]:
    """Per-frame probabilities to candidates, through the five steps in the module docstring."""
    speech = round(settings.speech_threshold * PERMILLE)
    silence = round(settings.silence_threshold * PERMILLE)
    regions = _hysteresis(probabilities, speech=speech, silence=silence)

    regions = _merge(regions, _ms_to_frames(settings.min_silence_ms))
    regions = [
        span for span in regions if span[1] - span[0] >= _ms_to_frames(settings.min_speech_ms)
    ]
    regions = _merge(regions, _ms_to_frames(settings.merge_gap_ms))

    pad = _ms_to_samples(settings.pad_ms)
    padded: list[tuple[int, int]] = []
    for start, end in regions:
        first = max(start * DETECTOR_FRAME_SAMPLES - pad, 0)
        last = min(end * DETECTOR_FRAME_SAMPLES + pad, n_samples)
        if last > first:
            padded.append((first, last))

    return tuple(_summarize(start, end, probabilities) for start, end in _merge_samples(padded))


def _hysteresis(
    probabilities: npt.NDArray[np.uint16], *, speech: int, silence: int
) -> list[tuple[int, int]]:
    """Frame runs that opened above ``speech`` and stayed above ``silence``."""
    found: list[tuple[int, int]] = []
    start: int | None = None
    for frame, value in enumerate(probabilities.tolist()):
        if start is None:
            if value >= speech:
                start = frame
        elif value < silence:
            found.append((start, frame))
            start = None
    if start is not None:
        found.append((start, int(probabilities.shape[0])))
    return found


def _merge(regions: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Join neighbours separated by fewer than ``gap`` units. Units are the caller's."""
    merged: list[tuple[int, int]] = []
    for start, end in regions:
        if merged and start - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _merge_samples(regions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Join regions that padding pushed into each other, so candidates stay disjoint."""
    merged: list[tuple[int, int]] = []
    for start, end in regions:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _summarize(start: int, end: int, probabilities: npt.NDArray[np.uint16]) -> SpeechRegion:
    """Mean and peak probability over the frames a sample interval touches.

    Padding can push a region's edges into frames the detector called silence, so the mean
    is taken over the frames the *padded* region covers. Quoting the unpadded mean would
    make a candidate look more confident than the audio it actually contains.
    """
    first = start // DETECTOR_FRAME_SAMPLES
    last = max(first + 1, -(-end // DETECTOR_FRAME_SAMPLES))
    values = probabilities[first : min(last, probabilities.shape[0])]
    if values.size == 0:
        return SpeechRegion(start, end, 0, 0)
    return SpeechRegion(
        start_sample=start,
        end_sample=end,
        probability_permille=to_permille(float(values.mean())),
        peak_probability_permille=int(values.max()),
    )


def _ms_to_frames(milliseconds: int) -> int:
    return -(-_ms_to_samples(milliseconds) // DETECTOR_FRAME_SAMPLES)


def _ms_to_samples(milliseconds: int) -> int:
    return milliseconds * DERIVATIVE_SAMPLE_RATE // 1000

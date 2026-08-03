"""Who is heard, and how loudly, at every moment (ADR-0022).

This is the part of M5 the charter calls the real gate. A mix that picks the wrong speaker
passes every loudness test there is, so the criteria here are written as *bounds* rather than
as behaviours that happen to hold on the fixture:

**A control grid, validated to be exact.** Gains are computed per control frame and linearly
interpolated to samples, so the applied gain is continuous by construction rather than by a
smoothing filter's good behaviour. `EnvelopeConfig` refuses a rate that does not divide the
session grid *and* an attack or release that is not a whole number of frames — the second does
not follow from the first (800 Hz divides 48000; an 11 ms attack is 8.8 frames of it).

**Two weight floors, because one cannot prove the criteria.** An active channel never falls
below `min_active_share` however badly it scored; an inactive one never falls below
`room_tone_share`. Worst-case solo dominance is therefore their ratio, computable before a
sample is read — with a single floor it would scale with the winner's score, and the gate
criterion would be a property of the fixture rather than of the rule.

**Suppressed candidates sit at the room-tone share; every retained candidate, `ambiguous`
included, is eligible.** That is the whole of this milestone's reading of the graph.
`ambiguous` marks a candidate the *track-level veto* kept — a lav hearing its wearer at that
wearer's normal level is probably not hearing someone else (ADR-0014) — so it is the least
obvious bleed case there is, not the most.

**A slew-limited linear ramp, not a one-pole.** An exponential never reaches its target and its
"attack time" is a time constant rather than a bound, which would make "respect attack, release
and maximum-slew limits" unassertable: there would be no frame at which it is true or false.

**The invariant is stated over what reaches a sample.** `sum(shares) == 1` exactly, and
`c_min <= sum(shares * corrections) <= c_max`. The first alone bounds nothing audible, which is
the finding M5's plan review opened with. Both are checked here as frames are produced, not
only in tests — and :class:`EnvelopeError` is what a violation raises, so the check can fail.

**Nothing is session-length.** 1 kHz over six tracks and four hours is 690 MB of gains, so this
is an iterator carrying its slew state across chunks, the way the 3:1 decimator carries filter
state across windows (INV-07). `tests/test_memory.py` asserts a write happens before the last
chunk is *produced*: a proof over the audio path alone is passed by a renderer that materializes
all of it first and only then interleaves reads and writes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from dnd_audio.activity import PERMILLE
from dnd_audio.artifacts.activity import ActivityGraph
from dnd_audio.config import EnvelopeConfig
from dnd_audio.errors import DndAudioError
from dnd_audio.mix.levels import MILLIBELS_PER_DB, LevelCorrections

__all__ = [
    "SHARE_TOLERANCE",
    "ActiveSpan",
    "EnvelopeChunk",
    "EnvelopeError",
    "EnvelopeStream",
    "active_spans",
    "expand",
    "frame_interval",
]

#: How far the normalized share may drift from 1.0 before the check fires. Six float64
#: divisions of a sum of six float64 values; anything above this is a bug, not rounding.
SHARE_TOLERANCE: Final = 1e-9

#: The same slack on the slew and applied-sum comparisons, so a check cannot fire on the last
#: bit of a division that is exactly at the limit.
_SLACK: Final = 1e-9


class EnvelopeError(DndAudioError):
    """The envelope produced a frame that violates a bound it exists to guarantee."""

    default_code = "envelope_invariant_violated"


@dataclass(frozen=True, slots=True)
class ActiveSpan:
    """One track being eligible, over a half-open range of control frames, at one weight."""

    track_index: int
    start_frame: int
    end_frame: int
    weight: float


@dataclass(frozen=True, slots=True)
class EnvelopeChunk:
    """A bounded run of control frames, with both the share and what reaches a sample.

    ``previous`` is the applied coefficient of the frame *before* ``start_frame`` — the value
    each track's ramp starts from. Carrying it here rather than looking ahead is what lets
    :func:`expand` interpolate without the stream ever holding a frame it has not produced.
    """

    start_frame: int
    #: ``(n_frames, n_tracks)``. The smoothed control signal the slew limit applies to,
    #: before the room-tone floor and before normalization. Exposed because it is the only
    #: place attack and release are a *per-frame bound*: the share is a nonlinear function of
    #: all six presences at once, so one track's share legitimately moves faster than its own
    #: ramp when another track's collapses.
    presence: npt.NDArray[np.float64]
    #: ``(n_frames, n_tracks)``. Sums to 1.0 across tracks at every frame.
    shares: npt.NDArray[np.float64]
    #: ``(n_frames, n_tracks)``. ``shares`` times each track's level correction: the number
    #: that multiplies a sample, and the one every gate criterion is about.
    applied: npt.NDArray[np.float64]
    #: ``(n_tracks,)``. The applied coefficients immediately before this chunk.
    previous: npt.NDArray[np.float64]

    @property
    def n_frames(self) -> int:
        return int(self.shares.shape[0])


def frame_interval(start_sample: int, end_sample: int, samples_per_frame: int) -> tuple[int, int]:
    """The control frames covering ``[start_sample, end_sample)``.

    The start floors and the end **ceils**, so the frame interval always covers the sample
    interval. The same rule as `resample.to_derivative_interval`, and for the same reason M2's
    closeout gives: rounding both ends alike shrinks a speech region, which is how a word loses
    its first phoneme. Here it would clip the first and last few milliseconds of every
    utterance, which the slew limit would then smear into a fade-in over the word.
    """
    return start_sample // samples_per_frame, -(-end_sample // samples_per_frame)


def active_spans(
    graph: ActivityGraph, *, settings: EnvelopeConfig, track_ids: tuple[str, ...]
) -> tuple[ActiveSpan, ...]:
    """Every retained candidate, as a weighted span of control frames.

    Retained and nothing else: a suppressed candidate is another track's voice arriving late
    and quiet, and promoting it is the failure the gate criterion names. `ambiguous` is not a
    reason to exclude one — see the module docstring.

    Spans are returned sorted by ``(track_index, start_frame)``, which is what lets
    :class:`EnvelopeStream` walk them with one cursor per track instead of scanning.
    """
    samples_per_frame = graph.sample_rate // settings.control_rate_hz
    index_of = {track_id: index for index, track_id in enumerate(track_ids)}
    spans: list[ActiveSpan] = []
    for candidate in graph.retained():
        track_index = index_of.get(candidate.track_id)
        if track_index is None:
            continue
        start, end = frame_interval(candidate.start_sample, candidate.end_sample, samples_per_frame)
        weight = settings.min_active_share + (1.0 - settings.min_active_share) * (
            candidate.score_permille / PERMILLE
        )
        spans.append(
            ActiveSpan(track_index=track_index, start_frame=start, end_frame=end, weight=weight)
        )
    return tuple(sorted(spans, key=lambda span: (span.track_index, span.start_frame)))


class EnvelopeStream:
    """The gain envelope, produced in bounded chunks with carried slew state.

    One instance produces one pass over the session. Chunks arrive in order and the slew state
    crosses their boundaries, so the envelope is identical however the caller partitions it —
    the property the resampler's own boundary handling has, proved the same way.
    """

    def __init__(
        self,
        graph: ActivityGraph,
        *,
        settings: EnvelopeConfig,
        corrections: LevelCorrections,
        track_ids: tuple[str, ...],
    ) -> None:
        if not track_ids:
            message = "an envelope needs at least one track to share gain between"
            raise ValueError(message)

        self._settings = settings
        self._track_ids = track_ids
        self._samples_per_frame = graph.sample_rate // settings.control_rate_hz
        self._total_frames = -(-graph.duration_samples // self._samples_per_frame)
        self._corrections = corrections.gains(track_ids)

        # Grouped by track rather than kept as one flat list with a shared cursor. The flat
        # version is how this was first written and it was wrong in a way that looked right:
        # every track's walk saw every track's spans, so one speaker lifted all six weights,
        # the shares stayed at 1/N, and the envelope came out perfectly flat. Six equal gains
        # are also exactly what a *correct* silent session produces, which is why it survived
        # a reading and not the first assertion.
        self._by_track: list[list[ActiveSpan]] = [[] for _ in track_ids]
        for span in active_spans(graph, settings=settings, track_ids=track_ids):
            self._by_track[span.track_index].append(span)

        # One frame's worth of the slew limit, in presence units. Full scale over the
        # configured time, so the bound a test asserts is exactly `1 / attack_frames`.
        self._rise = 1.0 / (settings.attack_ms * settings.control_rate_hz / 1000)
        self._fall = 1.0 / (settings.release_ms * settings.control_rate_hz / 1000)

        clamp = 10.0 ** (
            settings.max_level_correction_db * MILLIBELS_PER_DB / (20.0 * MILLIBELS_PER_DB)
        )
        self._applied_min = 1.0 / clamp
        self._applied_max = clamp

        self._presence = np.zeros(len(track_ids), dtype=np.float64)
        self._previous_applied = self._share(self._presence) * self._corrections
        self._cursors = [0] * len(track_ids)
        self._position = 0

    @property
    def total_frames(self) -> int:
        """Control frames in the whole session, `ceil` of the aligned duration."""
        return self._total_frames

    @property
    def samples_per_frame(self) -> int:
        return self._samples_per_frame

    @property
    def track_ids(self) -> tuple[str, ...]:
        return self._track_ids

    @property
    def corrections(self) -> npt.NDArray[np.float64]:
        """Each track's linear level correction, in `track_ids` order."""
        return self._corrections.copy()

    def chunks(self, *, chunk_frames: int) -> Iterator[EnvelopeChunk]:
        """Yield the whole session, at most ``chunk_frames`` control frames at a time.

        A generator rather than a list, and the difference between a bounded mixer and a
        690 MB one is exactly this keyword (INV-07).
        """
        if chunk_frames <= 0:
            message = f"chunk_frames must be positive, got {chunk_frames}"
            raise ValueError(message)
        while self._position < self._total_frames:
            yield self._next(min(chunk_frames, self._total_frames - self._position))

    def _next(self, n_frames: int) -> EnvelopeChunk:
        """Produce the next ``n_frames`` frames, advancing the slew state across them."""
        start = self._position
        targets = self._targets(start, n_frames)

        presence = np.empty((n_frames, len(self._track_ids)), dtype=np.float64)
        carried = self._presence
        for row in range(n_frames):
            carried = np.clip(targets[row], carried - self._fall, carried + self._rise)
            presence[row] = carried

        shares = self._share(presence)
        applied = shares * self._corrections
        previous = self._previous_applied

        self._check(presence, shares, applied)

        self._presence = carried
        self._previous_applied = applied[-1].copy()
        self._position = start + n_frames
        return EnvelopeChunk(
            start_frame=start,
            presence=presence,
            shares=shares,
            applied=applied,
            previous=previous,
        )

    def _targets(self, start: int, n_frames: int) -> npt.NDArray[np.float64]:
        """The unsmoothed active weight of every track over ``[start, start + n_frames)``.

        Zero where no retained candidate covers the frame. The room-tone floor is applied
        *after* smoothing, in :meth:`_share`, so that a channel decays toward the floor rather
        than toward it plus itself.
        """
        end = start + n_frames
        targets = np.zeros((n_frames, len(self._track_ids)), dtype=np.float64)
        for track_index, spans in enumerate(self._by_track):
            cursor = self._cursors[track_index]
            while cursor < len(spans):
                span = spans[cursor]
                if span.end_frame <= start:
                    cursor += 1
                    continue
                if span.start_frame >= end:
                    break
                lower = max(span.start_frame, start) - start
                upper = min(span.end_frame, end) - start
                # `maximum` rather than assignment: a track's retained candidates are disjoint
                # after M3's merge, but the artifact does not promise it and an overlap here
                # would otherwise let the later span's weight win by arriving second.
                targets[lower:upper, track_index] = np.maximum(
                    targets[lower:upper, track_index], span.weight
                )
                if span.end_frame > end:
                    break
                cursor += 1
            self._cursors[track_index] = cursor
        return targets

    def _share(self, presence: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Floor every weight, then normalize (ADR-0022's Dugan-style share).

        Accepts one frame or many; the sum is taken over the last axis either way.
        """
        weights = np.maximum(presence, self._settings.room_tone_share)
        return weights / np.sum(weights, axis=-1, keepdims=True)

    def _check(
        self,
        presence: npt.NDArray[np.float64],
        shares: npt.NDArray[np.float64],
        applied: npt.NDArray[np.float64],
    ) -> None:
        """Three bounds this exists to guarantee, checked where they are produced.

        Not in a test: a bound that only a test enforces is a bound a caller can construct a
        session to break. These are cheap — three reductions over a bounded chunk — and they
        cover the frames a fixture never reaches.
        """
        totals = np.sum(shares, axis=-1)
        if not np.all(np.abs(totals - 1.0) <= SHARE_TOLERANCE):
            worst = int(np.argmax(np.abs(totals - 1.0)))
            message = (
                f"the normalized gain share sums to {totals[worst]!r} at control frame "
                f"{self._position + worst}, not 1. Every frame's shares are one whole signal "
                f"divided between the tracks (ADR-0022)."
            )
            raise EnvelopeError(message)

        applied_totals = np.sum(applied, axis=-1)
        low = self._applied_min - _SLACK
        high = self._applied_max + _SLACK
        if not np.all((applied_totals >= low) & (applied_totals <= high)):
            worst = int(np.argmax(np.abs(applied_totals - 1.0)))
            message = (
                f"the applied coefficients sum to {applied_totals[worst]!r} at control frame "
                f"{self._position + worst}, outside the level-correction clamp "
                f"[{self._applied_min!r}, {self._applied_max!r}]. The share summing to 1 "
                f"bounds nothing audible; this is the bound that does (ADR-0022)."
            )
            raise EnvelopeError(message)

        steps = np.diff(np.vstack([self._presence, presence]), axis=0)
        if not np.all((steps <= self._rise + _SLACK) & (steps >= -self._fall - _SLACK)):
            worst = int(np.unravel_index(int(np.argmax(np.abs(steps))), steps.shape)[0])
            message = (
                f"the envelope moved by {steps[worst].max()!r}/{steps[worst].min()!r} in one "
                f"control frame at {self._position + worst}, outside the configured attack "
                f"(+{self._rise!r}) and release (-{self._fall!r}) limits"
            )
            raise EnvelopeError(message)


def expand(
    chunk: EnvelopeChunk, *, samples_per_frame: int, n_samples: int
) -> npt.NDArray[np.float64]:
    """Interpolate one chunk's applied coefficients onto ``n_samples`` samples.

    Within control frame *k* the gain ramps linearly from frame *k-1*'s value to frame *k*'s,
    reaching it exactly on the frame's last sample — so consecutive frames join without a step
    and the whole envelope is continuous. Interpolating *backwards* like this is what lets the
    stream produce a chunk without looking at the frame after it, which is the same reason the
    envelope can be an iterator at all.

    ``n_samples`` is passed rather than derived because the session's final control frame may
    cover fewer than ``samples_per_frame`` samples, and inventing gain past the aligned
    duration would put a ramp on audio that does not exist.
    """
    starts = np.vstack([chunk.previous[None, :], chunk.applied[:-1]])
    ramp = (np.arange(samples_per_frame, dtype=np.float64) + 1.0) / samples_per_frame
    gains = starts[:, None, :] + (chunk.applied - starts)[:, None, :] * ramp[None, :, None]
    return gains.reshape(-1, gains.shape[2])[:n_samples]

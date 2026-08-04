"""Cross-correlation as synchronization QA. It never moves a sample.

The spec is unambiguous about the boundary: an optional clap correlation "should report
disagreement with timecode, not override valid timecode automatically". So nothing here
returns a correction, and nothing that calls it can apply one — it produces warnings and
report decisions and that is all.

**Measuring at both ends is the point.** A constant lag between two tracks is a constant
timecode offset: the receivers disagree about what time it is, which is a capture-procedure
problem. A lag that *changes* between the start and the end is something else entirely —
the two sample clocks are running at different rates, which is **OQ-006**, the assumption
the MVP rests on and has no evidence for. The MVP still does not correct it (INV-12 forbids
correcting by an unmeasured amount); it says so, loudly, and H2 is what will settle it.

**Low correlation is reported as low confidence, not as nothing.** Correlating two
independent noise floors produces a confident-looking peak at some arbitrary lag, so a weak
peak must not become a warning. Discarding it is the opposite error, and the one M8 found:
on the 2026-08-03 capture six measurements whose lags matched an independent hand
measurement were reported as "no shared transient found", because ordinary speech does not
correlate like a clap. The instrument had the right answer and threw it away. There are
three outcomes now — measured, low confidence, and no signal at all — and only the first
can raise a disagreement. "Nobody clapped" and "the jam failed" no longer look identical.

**Thresholds are compared in integer samples** (INV-04). Milliseconds are the operator's
unit and the report's; nothing here decides anything by comparing two floats.

Off by default. It costs two correlations per track and answers a question most sessions do
not have.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.signal import correlate

from dnd_audio.artifacts.manifest import StartEvidenceRecord
from dnd_audio.artifacts.report import Decision, ReportBuilder
from dnd_audio.artifacts.timeline import TimelineNote, TimelineTrack
from dnd_audio.config import SessionConfig, SyncQaConfig
from dnd_audio.timecode import parse_frame_rate
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE
from dnd_audio.timeline.pcm import PcmReader, open_pcm
from dnd_audio.timeline.rasterize import evidence_quantum_samples

__all__ = [
    "MINIMUM_WINDOW_SAMPLES",
    "LagMeasurement",
    "measure_lag",
    "offset_floor_samples",
    "run_sync_qa",
]

#: Below this there is not enough signal for a correlation to mean anything, whatever the
#: configured window says. A tenth of a second at 16 kHz.
MINIMUM_WINDOW_SAMPLES: Final = 1600


@dataclass(frozen=True, slots=True)
class LagMeasurement:
    """One track against the reference, over one window."""

    track_id: str
    #: ``start`` or ``end``. Both, always, because the difference is the interesting part.
    position: str
    #: Samples at the derivative rate. Positive means this track's audio arrives *later*
    #: than the reference's, so its timecode places it too early.
    lag_samples: int
    #: Normalized peak, in [0, 1]. Below the configured threshold the lag is kept and
    #: marked low-confidence rather than discarded.
    correlation: float
    #: Whether there was any energy to correlate at all. False means a window was silent,
    #: which is a different fact from a weak peak and gets a different outcome.
    has_signal: bool = True

    @property
    def lag_ms(self) -> float:
        """The lag as an operator reads it. A serialization boundary, never a comparison."""
        return self.lag_samples * 1000 / DERIVATIVE_SAMPLE_RATE


def measure_lag(
    reference: npt.NDArray[np.float32],
    other: npt.NDArray[np.float32],
    *,
    max_lag_samples: int,
) -> tuple[int, float]:
    """Peak normalized cross-correlation and the lag it occurred at.

    Normalized by both signals' energy, so a quiet track and a loud one are comparable —
    an unnormalized correlation would rank tracks by volume and call the loudest one the
    best match.
    """
    if reference.size == 0 or other.size == 0:
        return 0, 0.0
    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(other, dtype=np.float64)
    energy = float(np.linalg.norm(first) * np.linalg.norm(second))
    if energy == 0.0:
        return 0, 0.0

    full = correlate(second, first, mode="full", method="fft")
    centre = first.size - 1
    low = max(0, centre - max_lag_samples)
    high = min(full.size, centre + max_lag_samples + 1)
    window = full[low:high]
    peak = int(np.argmax(np.abs(window)))
    return low + peak - centre, float(abs(window[peak]) / energy)


def offset_floor_samples(
    evidence: Sequence[StartEvidenceRecord], config: SessionConfig, *, rate: int
) -> int:
    """The finest constant offset this session's own timing evidence could express.

    Read from the evidence rather than from `timecode.frame_rate`, and the difference is
    not academic: **OQ-024** showed a receiver set to 60 fps writing `TIMECODE_RATE 30/1`
    and references on 1600-sample boundaries anyway. Deriving the floor from the configured
    rate would give that session a 16.7 ms threshold against source timing that still moves
    in 33.3 ms steps — reinstating the false alarm this exists to remove.

    One sample when a session carries nothing but operator-stated offsets, which are exact.
    """
    frame_rate = parse_frame_rate(config.timecode.frame_rate)
    quantum = config.timecode.bwf_reference_quantum_samples
    return max(
        (
            evidence_quantum_samples(item, frame_rate, rate, bwf_quantum_samples=quantum)
            for item in evidence
        ),
        default=1,
    )


def _threshold_samples(milliseconds: int, rate: int) -> int:
    """An integer-millisecond threshold as whole samples, rounded up (INV-04).

    Rounded up so that a threshold is never made *stricter* by the conversion: an operator
    who writes 33 ms should not get a warning at 32.9.
    """
    return math.ceil(Fraction(milliseconds * rate, 1000))


def run_sync_qa(
    session_dir: Path,
    config: SessionConfig,
    tracks: list[TimelineTrack],
    *,
    builder: ReportBuilder,
    evidence: Sequence[StartEvidenceRecord] = (),
) -> list[TimelineNote]:
    """Correlate every track against a reference near both ends of the session.

    Returns the warnings; records the measurements as report decisions. Returns an empty
    list — and reads nothing — when `sync_qa.enabled` is false.

    Args:
        evidence: The session's own timing evidence, used only to derive the constant-offset
            threshold when none is configured. Empty means "assume exact", which is the
            conservative reading: it produces the tightest threshold rather than the widest.
    """
    settings = config.sync_qa
    if not settings.enabled:
        return []

    usable = [track for track in tracks if track.segments and track.derivatives]
    if len(usable) < 2:
        return [
            TimelineNote(
                code="sync_qa_skipped",
                message=(
                    "cross-correlation QA needs at least two tracks with working audio; "
                    f"this session has {len(usable)}"
                ),
            )
        ]

    duration = max(
        derivative.output_samples for track in usable for derivative in track.derivatives
    )
    window = min(settings.window_s * DERIVATIVE_SAMPLE_RATE, duration // 2)
    if window < MINIMUM_WINDOW_SAMPLES:
        return [
            TimelineNote(
                code="sync_qa_skipped",
                message=(
                    f"this session is too short for cross-correlation QA: a "
                    f"{window}-sample window at {DERIVATIVE_SAMPLE_RATE} Hz is below the "
                    f"{MINIMUM_WINDOW_SAMPLES} needed for a correlation to mean anything"
                ),
            )
        ]

    reference, *others = usable
    ends = {"start": 0, "end": duration - window}
    with _derivative(session_dir, reference) as handle:
        anchors = {name: handle.read(offset, window) for name, offset in ends.items()}

    max_lag = max(1, settings.max_lag_ms * DERIVATIVE_SAMPLE_RATE // 1000)
    if settings.offset_warn_ms is None:
        offset_threshold = offset_floor_samples(evidence, config, rate=DERIVATIVE_SAMPLE_RATE)
    else:
        offset_threshold = _threshold_samples(settings.offset_warn_ms, DERIVATIVE_SAMPLE_RATE)
    drift_threshold = _threshold_samples(settings.drift_warn_ms, DERIVATIVE_SAMPLE_RATE)

    notes: list[TimelineNote] = []
    for track in others:
        measurements = []
        with _derivative(session_dir, track) as handle:
            for name, offset in ends.items():
                against = handle.read(offset, window)
                lag, peak = measure_lag(anchors[name], against, max_lag_samples=max_lag)
                measurements.append(
                    LagMeasurement(
                        track_id=track.track_id,
                        position=name,
                        lag_samples=lag,
                        correlation=peak,
                        # Exactly the condition `measure_lag` returns nothing for. Recovered
                        # here rather than returned, so a track of digital silence stays
                        # distinguishable from a peak that happened to land at zero.
                        has_signal=_has_energy(anchors[name]) and _has_energy(against),
                    )
                )
        notes.extend(
            _assess(
                reference.track_id,
                measurements,
                settings=settings,
                builder=builder,
                offset_threshold=offset_threshold,
                drift_threshold=drift_threshold,
            )
        )
    return notes


def _has_energy(window: npt.NDArray[np.float32]) -> bool:
    return window.size > 0 and float(np.linalg.norm(np.asarray(window, dtype=np.float64))) > 0.0


def _outcome(found: LagMeasurement, minimum: float) -> str:
    """Which of the three things happened here.

    Kept as one function because the distinction is the substance of the change: before M8
    the second and third were one code, and a session where nobody clapped read exactly like
    a session where the jam had failed.
    """
    if not found.has_signal:
        return "sync_qa_no_signal"
    return "sync_qa_measured" if found.correlation >= minimum else "sync_qa_low_confidence"


def _assess(
    reference_id: str,
    measurements: list[LagMeasurement],
    *,
    settings: SyncQaConfig,
    builder: ReportBuilder,
    offset_threshold: int,
    drift_threshold: int,
) -> list[TimelineNote]:
    """Turn two measurements into decisions and, where warranted, warnings.

    ``offset_threshold`` and ``drift_threshold`` are in samples at the derivative rate, not
    in milliseconds: every comparison below is integer (INV-04), and the millisecond figures
    appear only in the prose an operator reads.
    """
    minimum = settings.min_correlation
    track_id = measurements[0].track_id
    notes: list[TimelineNote] = []

    for found in measurements:
        code = _outcome(found, minimum)
        if code == "sync_qa_no_signal":
            detail = (
                f"nothing to correlate against {reference_id}: one of the two windows is "
                f"silent. No transient was recorded here — which is not the same as a jam "
                f"that failed, and is why this is its own outcome."
            )
        elif code == "sync_qa_low_confidence":
            detail = (
                f"a peak against {reference_id} at {found.lag_ms:+.2f} ms, but its "
                f"correlation of {found.correlation:.3f} is below {minimum}. The lag is "
                f"reported because it is evidence — ordinary speech does not correlate like "
                f"a clap — and it raises no disagreement on its own."
            )
        else:
            detail = (
                f"peak correlation {found.correlation:.3f} against {reference_id} at a "
                f"lag of {found.lag_ms:+.2f} ms. QA only — the timeline was not adjusted."
            )
        builder.record_decision(
            Decision(
                code=code,
                subject=f"{track_id}:{found.position}",
                detail=detail,
                details={
                    "correlation": f"{found.correlation:.4f}",
                    "lag_ms": f"{found.lag_ms:+.3f}",
                    "lag_samples": str(found.lag_samples),
                    "position": found.position,
                    "reference": reference_id,
                },
            )
        )

    confident = [found for found in measurements if _outcome(found, minimum) == "sync_qa_measured"]
    weak = [found for found in measurements if _outcome(found, minimum) == "sync_qa_low_confidence"]
    silent = [found for found in measurements if _outcome(found, minimum) == "sync_qa_no_signal"]

    if weak:
        detail = ", ".join(
            f"{found.position} ({found.lag_ms:+.2f} ms at {found.correlation:.3f})"
            for found in weak
        )
        notes.append(
            TimelineNote(
                code="sync_qa_low_confidence",
                message=(
                    f"a lag against {reference_id} was measured but is below a normalized "
                    f"correlation of {minimum}, so it is reported rather than acted on: "
                    f"{detail}. A weak peak can be two noise floors agreeing by chance — and "
                    f"can equally be ordinary speech, which is what the 2026-08-03 capture "
                    f"turned out to be."
                ),
                path=track_id,
            )
        )

    if silent:
        detail = ", ".join(found.position for found in silent)
        notes.append(
            TimelineNote(
                code="sync_qa_no_signal",
                message=(
                    f"no audio to correlate against {reference_id} at: {detail}. One of the "
                    f"two windows is silent, so nothing was measured — as distinct from a "
                    f"measurement that came out weak."
                ),
                path=track_id,
            )
        )

    for found in confident:
        if abs(found.lag_samples) > offset_threshold:
            threshold_ms = offset_threshold * 1000 / DERIVATIVE_SAMPLE_RATE
            notes.append(
                TimelineNote(
                    code="timecode_disagreement",
                    message=(
                        f"its audio sits {found.lag_ms:+.2f} ms from {reference_id}'s at the "
                        f"{found.position} of the session, beyond the {threshold_ms:.2f} ms "
                        f"this session's own timing evidence could explain. The timeline "
                        f"still follows the timecode: valid timecode is never overridden by "
                        f"a correlation."
                    ),
                    path=track_id,
                )
            )

    if len(confident) == 2:
        drift_samples = confident[1].lag_samples - confident[0].lag_samples
        drift = confident[1].lag_ms - confident[0].lag_ms
        threshold_ms = float(settings.drift_warn_ms)
        if abs(drift_samples) > drift_threshold:
            notes.append(
                TimelineNote(
                    code="clock_drift_suspected",
                    message=(
                        f"its lag against {reference_id} changed by {drift:+.2f} ms between "
                        f"the start and the end of the session. A constant offset is a "
                        f"timecode disagreement; a changing one is evidence that the two "
                        f"sample clocks run at different rates (OQ-006). No correction was "
                        f"applied — drift correction is post-MVP."
                    ),
                    path=track_id,
                )
            )
    return notes


def _derivative(session_dir: Path, track: TimelineTrack) -> PcmReader:
    """A reader over this track's 16 kHz working audio, for the caller's ``with``.

    Deliberately *not* entered here. An earlier version called ``__enter__`` itself and
    every caller then entered it again, which opens a second handle and drops the first
    without closing it — relying on the garbage collector to clean up a file descriptor.
    """
    record = next(
        derivative
        for derivative in track.derivatives
        if derivative.sample_rate == DERIVATIVE_SAMPLE_RATE
    )
    return PcmReader(open_pcm(session_dir / record.relative_path))

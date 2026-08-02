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

**Low correlation is reported as low correlation.** Correlating two independent noise floors
produces a confident-looking peak at some arbitrary lag, and a QA step that reported it as
a measurement would be worse than no QA step. Below `sync_qa.min_correlation` the answer is
"no shared transient was found here", not a number.

Off by default. It costs two correlations per track and answers a question most sessions do
not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.signal import correlate

from dnd_audio.artifacts.report import Decision, ReportBuilder
from dnd_audio.artifacts.timeline import TimelineNote, TimelineTrack
from dnd_audio.config import SessionConfig, SyncQaConfig
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE
from dnd_audio.timeline.pcm import PcmReader, open_pcm

__all__ = ["MINIMUM_WINDOW_SAMPLES", "LagMeasurement", "measure_lag", "run_sync_qa"]

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
    #: Normalized peak, in [0, 1]. Below the configured threshold the lag is meaningless.
    correlation: float

    @property
    def lag_ms(self) -> float:
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


def run_sync_qa(
    session_dir: Path,
    config: SessionConfig,
    tracks: list[TimelineTrack],
    *,
    builder: ReportBuilder,
) -> list[TimelineNote]:
    """Correlate every track against a reference near both ends of the session.

    Returns the warnings; records the measurements as report decisions. Returns an empty
    list — and reads nothing — when `sync_qa.enabled` is false.
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
    notes: list[TimelineNote] = []
    for track in others:
        measurements = []
        with _derivative(session_dir, track) as handle:
            for name, offset in ends.items():
                lag, peak = measure_lag(
                    anchors[name], handle.read(offset, window), max_lag_samples=max_lag
                )
                measurements.append(
                    LagMeasurement(
                        track_id=track.track_id, position=name, lag_samples=lag, correlation=peak
                    )
                )
        notes.extend(_assess(reference.track_id, measurements, settings=settings, builder=builder))
    return notes


def _assess(
    reference_id: str,
    measurements: list[LagMeasurement],
    *,
    settings: SyncQaConfig,
    builder: ReportBuilder,
) -> list[TimelineNote]:
    """Turn two measurements into decisions and, where warranted, warnings."""
    threshold_ms = float(settings.drift_warn_ms)
    minimum = settings.min_correlation
    track_id = measurements[0].track_id
    notes: list[TimelineNote] = []

    for found in measurements:
        builder.record_decision(
            Decision(
                code="sync_qa_measured",
                subject=f"{track_id}:{found.position}",
                detail=(
                    f"peak correlation {found.correlation:.3f} against {reference_id} at a "
                    f"lag of {found.lag_ms:+.2f} ms. QA only — the timeline was not adjusted."
                ),
                details={
                    "correlation": f"{found.correlation:.4f}",
                    "lag_ms": f"{found.lag_ms:+.3f}",
                    "position": found.position,
                    "reference": reference_id,
                },
            )
        )

    confident = [found for found in measurements if found.correlation >= minimum]
    if len(confident) < len(measurements):
        weak = ", ".join(
            f"{found.position} ({found.correlation:.3f})"
            for found in measurements
            if found.correlation < minimum
        )
        notes.append(
            TimelineNote(
                code="sync_qa_inconclusive",
                message=(
                    f"no shared transient found against {reference_id} at: {weak}. Below "
                    f"a normalized correlation of {minimum}, a peak is as likely to be two "
                    f"noise floors agreeing by chance as a clap."
                ),
                path=track_id,
            )
        )

    for found in confident:
        if abs(found.lag_ms) > threshold_ms:
            notes.append(
                TimelineNote(
                    code="timecode_disagreement",
                    message=(
                        f"its audio sits {found.lag_ms:+.2f} ms from {reference_id}'s at the "
                        f"{found.position} of the session, beyond the {threshold_ms} ms "
                        f"threshold. The timeline still follows the timecode: valid timecode "
                        f"is never overridden by a correlation."
                    ),
                    path=track_id,
                )
            )

    if len(confident) == 2:
        drift = confident[1].lag_ms - confident[0].lag_ms
        if abs(drift) > threshold_ms:
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

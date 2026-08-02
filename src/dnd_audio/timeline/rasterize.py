"""Evidence → exact time → one integer sample position (ADR-0008, ADR-0009).

M1 captured three kinds of timing evidence and deliberately did not collapse them
(ADR-0006). This module is where they are finally reconciled, and the whole of its
correctness is in two rules.

**Exactness is preserved until the last step, and there is exactly one last step.** Every
conversion here returns a :class:`~fractions.Fraction`. The single rounding happens in
:func:`session_position`, which takes the source's time and session zero's time and
subtracts them *before* quantizing. Rounding an absolute position and then subtracting a
rounded origin rounds twice, doubles the worst-case error, and makes a chunk's placement
depend on where ``origin_timecode`` happens to sit inside a sample.

**A 24-hour cycle is counted in the evidence's own units.** This is the finding that a
plausible first draft got wrong. "Add a day" is not one operation:

===============================  ==========================  ===================
Evidence                         One cycle                   In real seconds
===============================  ==========================  ===================
``bwf_sample_reference``         ``86400 * sample_rate``     86 400 exactly
``timecode`` at 24F/25F/30F/…    2 592 000 … frames          86 400 exactly
``timecode`` at 23.98F / 29.97F  2 073 600 / 2 592 000       **86 486.4**
``timecode`` at 29.97DF          2 589 408 frames            86 399.9136
===============================  ==========================  ===================

Non-drop fractional timecode does not track wall time — that is what drop-frame exists to
fix — so unwrapping a wrapped 29.97F chunk by 86 400 seconds lands it 86.4 seconds *before*
the frame that preceded it, manufacturing a large false overlap out of correct evidence.
:func:`absolute_seconds` therefore takes whole cycles and adds them in frames or samples,
never in seconds.

The two absolute domains are compared against each other on the assumption that a
recorder's ``00:00:00:00`` is real midnight. That holds trivially at an integer rate and
is an assumption at a fractional non-drop one, which is **OQ-015**. A caller notices with
:func:`has_mixed_absolute_domains` and sizes it with
:func:`timecode_day_discrepancy_seconds`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from typing import Final

from dnd_audio.artifacts.manifest import (
    BwfSampleReferenceRecord,
    SessionOffsetRecord,
    StartEvidenceRecord,
    TimecodeRecord,
)
from dnd_audio.determinism import to_samples
from dnd_audio.timecode import FrameRate, frames_per_day

__all__ = [
    "SECONDS_PER_DAY",
    "absolute_seconds",
    "cycle_units",
    "has_mixed_absolute_domains",
    "is_absolute",
    "is_frame_quantized",
    "quantization_tolerance_samples",
    "relative_seconds",
    "session_position",
    "timecode_day_discrepancy_seconds",
]

#: A calendar day. Used for a BWF reference, whose samples-since-midnight really is
#: measured against wall time — and pointedly *not* for a timecode, which has its own
#: cycle (see the module docstring).
SECONDS_PER_DAY: Final = 86400


def is_absolute(evidence: StartEvidenceRecord) -> bool:
    """Whether this evidence names a time of day rather than a session-relative offset.

    Only absolute evidence can decide where session zero is: an offset is *defined*
    relative to zero, so using one to locate zero is circular (ADR-0009).
    """
    return not isinstance(evidence, SessionOffsetRecord)


def is_frame_quantized(evidence: StartEvidenceRecord) -> bool:
    """Whether this evidence's resolution is a frame rather than a sample.

    A recorder that writes ``19:00:00:00`` may have started anywhere inside that frame, so
    a timecode's own quantization — 1602 samples at 29.97 fps — dwarfs the single-sample
    rounding this module introduces. That difference is what makes the overlap tolerance a
    property of the evidence pair rather than a constant.
    """
    return isinstance(evidence, TimecodeRecord)


def cycle_units(evidence: StartEvidenceRecord, frame_rate: FrameRate) -> int | None:
    """How much one 24-hour wrap adds, **in this evidence's own units**.

    Returns ``None`` for a session-relative offset, which has no midnight in it: it is
    measured from session zero, and session zero does not wrap.

    Args:
        frame_rate: The configured rate. Used only for timecode evidence, whose record
            carries its own rate — they are cross-checked by the caller, not here.
    """
    if isinstance(evidence, BwfSampleReferenceRecord):
        return SECONDS_PER_DAY * evidence.sample_rate
    if isinstance(evidence, TimecodeRecord):
        return frames_per_day(frame_rate)
    return None


def absolute_seconds(
    evidence: StartEvidenceRecord, frame_rate: FrameRate, *, cycles: int = 0
) -> Fraction:
    """Exact seconds from this evidence's day origin, after ``cycles`` 24-hour wraps.

    The cycles are added in the evidence's own units *before* the conversion to seconds,
    which is the whole point — see the module docstring's table.

    Raises:
        ValueError: if handed a session-relative offset, which has no day origin, or a
            negative cycle count. Both mean a caller has confused the two coordinate
            systems ADR-0006 exists to keep apart, and inventing an answer would place
            audio somewhere plausible and wrong.
    """
    if cycles < 0:
        message = f"cycles must not be negative, got {cycles}; only forward rollover is inferred"
        raise ValueError(message)

    if isinstance(evidence, BwfSampleReferenceRecord):
        samples = evidence.samples + cycles * SECONDS_PER_DAY * evidence.sample_rate
        return Fraction(samples, evidence.sample_rate)

    if isinstance(evidence, TimecodeRecord):
        frames = evidence.frames + cycles * frames_per_day(frame_rate)
        # frames / (numerator/denominator), kept exact. At 30000/1001 fps one frame is
        # 1001/30000 s, which is where the 8008/5-samples-per-frame figure comes from.
        return Fraction(frames * evidence.frame_rate.denominator, evidence.frame_rate.numerator)

    message = (
        "a session_offset_samples override has no time of day: it is measured from session "
        "zero, which is what this function is being used to help determine (ADR-0006)"
    )
    raise ValueError(message)


def relative_seconds(evidence: SessionOffsetRecord) -> Fraction:
    """Exact seconds from session zero, for an operator-supplied offset.

    Signed, as the spec requires: a negative offset places audio before session zero, and
    ADR-0009 keeps that meaningful by shifting the whole timeline when no explicit origin
    was configured.
    """
    return Fraction(evidence.samples, evidence.sample_rate)


def session_position(source_seconds: Fraction, zero_seconds: Fraction, rate: int) -> int:
    """The integer sample position of ``source_seconds`` on a timeline starting at zero.

    **One subtraction, then one rounding.** The signature takes both times rather than a
    pre-computed difference so that the rule ADR-0008 states is structural: a caller
    cannot accidentally quantize an absolute position and subtract a quantized origin,
    because there is nowhere here to pass one.

    May return a negative value. Whether that is fatal or means the timeline needs
    shifting is ADR-0009's question, not this function's.
    """
    return to_samples(source_seconds - zero_seconds, rate)


def quantization_tolerance_samples(
    first: StartEvidenceRecord, second: StartEvidenceRecord, frame_rate: FrameRate, rate: int
) -> int:
    """The largest overlap between two chunks explainable by rounding rather than by audio.

    One sample when both starts came from sample-exact evidence — the only error available
    is :func:`session_position`'s single rounding. One whole frame, rounded up, when either
    came from a timecode, because the recorder's own quantization is then three orders of
    magnitude larger than ours.

    A property of the *pair*, not a global constant: a 1602-sample overlap between two
    BWF-timed chunks is a real overlap the operator should see, and calling it rounding
    because some other track uses timecodes would hide it (ADR-0008, ADR-0010).
    """
    tolerance = 1
    for evidence in (first, second):
        if not is_frame_quantized(evidence):
            continue
        per_frame = Fraction(rate) / frame_rate.rate
        tolerance = max(tolerance, math.ceil(per_frame))
    return tolerance


def timecode_day_discrepancy_seconds(frame_rate: FrameRate) -> Fraction:
    """How far a timecode day is from a calendar day, exactly.

    Zero at every integer rate, where a timecode day *is* 86 400 seconds and a timecode's
    day origin coincides with real midnight by construction. Non-zero at 23.98F and 29.97F
    (+86.4 s) and, much more mildly, at 29.97DF (-0.0864 s, drop-frame's residual).

    Returned as a signed exact value rather than a boolean so a caller can put the
    magnitude in its warning: 86 seconds and 86 milliseconds are the same *kind* of
    assumption and very different problems.
    """
    return Fraction(frames_per_day(frame_rate)) / frame_rate.rate - SECONDS_PER_DAY


def has_mixed_absolute_domains(evidence: Sequence[StartEvidenceRecord]) -> bool:
    """Whether both absolute domains appear among this session's evidence.

    A BWF reference counts from real midnight; a timecode counts from the recorder's
    ``00:00:00:00``. Relating the two requires assuming where the recorder was jammed
    (**OQ-015**). Whether that assumption costs anything depends on the rate — see
    :func:`timecode_day_discrepancy_seconds` — so the two questions are answered
    separately and the caller combines them.

    The canonical fixture mixes exactly these domains at 30F, where the discrepancy is
    zero and there is nothing to warn about.
    """
    kinds = {type(item) for item in evidence if is_absolute(item)}
    return {BwfSampleReferenceRecord, TimecodeRecord} <= kinds

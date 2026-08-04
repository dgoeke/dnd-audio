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

**Neither absolute domain's origin is midnight.** A ``bwf_sample_reference`` counts from the
recorder's own timecode origin, which on this hardware is where that receiver was last
jammed or powered on — OQ-004 measured a 19:26:55 file carrying 388 seconds, and OQ-023
measured a jam propagating into the field to within one frame. A ``timecode`` counts from a
recorder's ``00:00:00:00``, which is not real midnight at a fractional non-drop rate either
(**OQ-015**). None of that reaches placement, because :func:`session_position` is a
*subtraction* and a shared origin cancels out of it — but it is why nothing here says
"midnight", and why :class:`~dnd_audio.artifacts.timeline.SessionZero` records
``since_domain_origin_samples`` rather than a time of day (ADR-0031).

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

The two absolute domains are compared against each other on the assumption that they share
one origin. On this hardware they do — the reference *is* the receiver's jammed timecode
count, in samples instead of frames (OQ-023) — so what is left is that their 24-hour cycles
differ in length at a fractional non-drop rate, which is **OQ-015**. A caller notices with
:func:`has_mixed_absolute_domains` and sizes it with
:func:`timecode_day_discrepancy_seconds`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
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
    "evidence_quantum_samples",
    "has_mixed_absolute_domains",
    "is_absolute",
    "quantization_tolerance_samples",
    "relative_seconds",
    "session_position",
    "timecode_day_discrepancy_seconds",
]

#: One counter cycle for a BWF reference, in seconds — and pointedly *not* for a timecode,
#: which counts its own cycle in frames (see the module docstring).
#:
#: The name survives the reframe because the arithmetic does: 86 400 seconds of samples is
#: 86 400 seconds of samples whatever the origin is measured from, so **OQ-004** disproving
#: "since midnight" changed what the number means and not what it is. Whether a DJI counter
#: wraps at all, and with what period, is **OQ-026** — see :func:`cycle_units`.
SECONDS_PER_DAY: Final = 86400


def is_absolute(evidence: StartEvidenceRecord) -> bool:
    """Whether this evidence names a time of day rather than a session-relative offset.

    Only absolute evidence can decide where session zero is: an offset is *defined*
    relative to zero, so using one to locate zero is circular (ADR-0009).
    """
    return not isinstance(evidence, SessionOffsetRecord)


def evidence_quantum_samples(
    evidence: StartEvidenceRecord,
    frame_rate: FrameRate,
    rate: int,
    *,
    bwf_quantum_samples: int,
) -> int:
    """This evidence's own resolution, in samples at ``rate``, rounded up.

    A recorder that writes ``19:00:00:00`` may have started anywhere inside that frame, so a
    timecode's own quantization — 1602 samples at 29.97 fps — dwarfs the single-sample
    rounding this module introduces.

    **A BWF sample reference is not automatically finer.** It is a sample count and looks
    exact, but on this hardware it moves in steps of a whole frame: OQ-004 measured every
    `time_reference` in the sample captures as a multiple of 1600 samples. Believing the
    field's units rather than its measured behaviour is what makes an ordinary second chunk
    look like a material overlap, since it rounds *backward* into the chunk before it. The
    quantum cannot be read from the file (OQ-024), so it is configuration —
    ``timecode.bwf_reference_quantum_samples``.

    A session-relative offset is an operator's assertion and is exact by construction.
    """
    if isinstance(evidence, TimecodeRecord):
        return math.ceil(Fraction(rate) / frame_rate.rate)
    if isinstance(evidence, BwfSampleReferenceRecord):
        # Stated at the file's own rate, applied at the session's.
        return math.ceil(Fraction(bwf_quantum_samples * rate, evidence.sample_rate))
    return 1


def cycle_units(evidence: StartEvidenceRecord, frame_rate: FrameRate) -> int | None:
    """How much one 24-hour wrap adds, **in this evidence's own units**.

    Returns ``None`` for a session-relative offset, which has no cycle in it: it is measured
    from session zero, and session zero does not wrap.

    **That a BWF reference wraps every 24 hours is now an assumption rather than a
    definition** (**OQ-026**). It followed from "samples since midnight", which OQ-004
    disproved; a device-local counter need not have a 24-hour period, or any period a
    session would reach. The arithmetic stays because unwrapping is spec-required and
    tested, because a recorder whose reference *is* midnight-relative needs it, and because
    INV-12 keeps it safe meanwhile — the inference warns, refuses a tie rather than
    guessing, and no real DJI session has reached it.

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


def session_position(
    source_seconds: Fraction,
    zero_seconds: Fraction,
    rate: int,
    *,
    adjust: Callable[[Fraction], Fraction] | None = None,
) -> int:
    """The integer sample position of ``source_seconds`` on a timeline starting at zero.

    **One subtraction, then one rounding.** The signature takes both times rather than a
    pre-computed difference so that the rule ADR-0008 states is structural: a caller
    cannot accidentally quantize an absolute position and subtract a quantized origin,
    because there is nowhere here to pass one.

    ``adjust`` is where a time warp goes (:mod:`~dnd_audio.timeline.warp`) — applied to the
    exact elapsed time, inside the single rounding. A correction applied to an
    already-rounded index would accumulate the error it exists to remove.

    May return a negative value. Whether that is fatal or means the timeline needs
    shifting is ADR-0009's question, not this function's.
    """
    elapsed = source_seconds - zero_seconds
    if adjust is not None:
        elapsed = adjust(elapsed)
    return to_samples(elapsed, rate)


def quantization_tolerance_samples(
    first: StartEvidenceRecord,
    second: StartEvidenceRecord,
    frame_rate: FrameRate,
    rate: int,
    *,
    bwf_quantum_samples: int,
) -> int:
    """The largest overlap between two chunks explainable by rounding rather than by audio.

    The coarser of the two chunks' own quanta, and never less than one sample — the single
    rounding :func:`session_position` introduces is always available as an explanation.

    A property of the *pair*, not a global constant: an overlap larger than either chunk's
    evidence could have produced is a real overlap the operator should see, and calling it
    rounding because some other track uses coarser evidence would hide it (ADR-0008,
    ADR-0010).
    """
    return max(
        1,
        *(
            evidence_quantum_samples(
                evidence, frame_rate, rate, bwf_quantum_samples=bwf_quantum_samples
            )
            for evidence in (first, second)
        ),
    )


def timecode_day_discrepancy_seconds(frame_rate: FrameRate) -> Fraction:
    """How far a timecode day is from a calendar day, exactly.

    Zero at every integer rate, where a timecode cycle *is* 86 400 seconds and therefore
    the same length as a BWF reference's. Non-zero at 23.98F and 29.97F
    (+86.4 s) and, much more mildly, at 29.97DF (-0.0864 s, drop-frame's residual).

    Returned as a signed exact value rather than a boolean so a caller can put the
    magnitude in its warning: 86 seconds and 86 milliseconds are the same *kind* of
    assumption and very different problems.
    """
    return Fraction(frames_per_day(frame_rate)) / frame_rate.rate - SECONDS_PER_DAY


def has_mixed_absolute_domains(evidence: Sequence[StartEvidenceRecord]) -> bool:
    """Whether both absolute domains appear among this session's evidence.

    Both count from the recorder's own origin — the reference in samples, the tag in frames
    — so on this hardware they are one clock in two units (OQ-023) and relating them is
    sound. What differs is the length of their 24-hour cycles at a fractional non-drop rate
    (**OQ-015**), which matters only once a session is unwrapped across one. Whether that
    costs anything depends on the rate — see :func:`timecode_day_discrepancy_seconds` — so
    the two questions are answered separately and the caller combines them.

    The canonical fixture mixes exactly these domains at 30F, where the discrepancy is
    zero and there is nothing to warn about.
    """
    kinds = {type(item) for item in evidence if is_absolute(item)}
    return {BwfSampleReferenceRecord, TimecodeRecord} <= kinds

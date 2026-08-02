"""The timecode strategy chain: extract timing evidence, or fail saying why.

The spec is emphatic that no DJI metadata layout may be invented, so this is a short
ordered list of named strategies, each of which either produces evidence or declines
with a reason. Both outcomes are recorded — a manifest that says *which* strategies
declined and *why* is what makes H1 cheap, because settling OQ-001 then means reading
the recorded reasons rather than re-running an investigation.

**Evidence is a tagged union and is never collapsed into one number** (ADR-0006). The
three kinds do not share a coordinate system:

===========================  ========  =====================  ==============  ======
Evidence                     Unit      Rate                   Origin          Signed
===========================  ========  =====================  ==============  ======
:class:`BwfSampleReference`  samples   the file's own rate    midnight        no
:class:`TimecodeReference`   frames    the configured rate    midnight        no
:class:`SessionOffset`       samples   48 kHz                 session zero    yes
===========================  ========  =====================  ==============  ======

Reconciling them, inferring midnight rollover, and rasterizing onto the working sample
grid are M2's. This module's whole job is to preserve what the evidence actually says.

**Nothing here reads a filename or a modification time.** INV-12 forbids inventing a
time, and both are exactly the plausible-looking sources that would let it happen
quietly. When no strategy matches, the result is a fatal error whose message names the
override that would fix it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from dnd_audio.config import SourceTimeOverride
from dnd_audio.errors import RecoveryError, TimecodeError
from dnd_audio.timecode import FrameRate, frame_index, parse_timecode

__all__ = [
    "CANONICAL_SAMPLE_RATE",
    "BwfSampleReference",
    "DeclinedStrategy",
    "SessionOffset",
    "SourceContext",
    "StartEvidence",
    "StartTime",
    "TimecodeReference",
    "extract_start_time",
    "strategy_names",
]

#: The rate a recovery offset is expressed at, per the spec and `session.yaml`.
CANONICAL_SAMPLE_RATE: Final = 48000

_TIME_REFERENCE_TAG: Final = "time_reference"
_TIMECODE_TAG: Final = "timecode"
_DATE_TAG: Final = "date"


@dataclass(frozen=True, slots=True)
class BwfSampleReference:
    """A BWF ``time_reference``: samples since midnight at the file's own rate.

    Kept as an integer and never rounded through a frame count (INV-04). The rate is
    carried alongside because it is the *file's*, which need not be the session's.
    """

    samples: int
    sample_rate: int
    origination_date: dt.date | None = None


@dataclass(frozen=True, slots=True)
class TimecodeReference:
    """A timecode tag, as a frame index rather than a sample position.

    A frame index is exact at every rate; a sample position is not, and the rounding
    rule for 8008/5 samples per frame belongs to M2 (ADR-0006).
    """

    text: str
    frames: int
    frame_rate_label: str
    frame_rate: Fraction
    drop_frame: bool
    recording_date: dt.date | None = None


@dataclass(frozen=True, slots=True)
class SessionOffset:
    """An operator-supplied offset: signed, 48 kHz, relative to session zero.

    Not a time of day. Placing it requires knowing where session zero is, which is M2's
    to determine — which is exactly why this is a distinct variant rather than a number
    converted into one of the others.
    """

    samples: int
    sample_rate: int = CANONICAL_SAMPLE_RATE
    recording_date: dt.date | None = None


StartEvidence = BwfSampleReference | TimecodeReference | SessionOffset


@dataclass(frozen=True, slots=True)
class DeclinedStrategy:
    """A strategy that ran and found nothing, and what it was looking for."""

    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class StartTime:
    """What was found, by which strategy, resting on which assumptions."""

    strategy: str
    evidence: StartEvidence
    #: Stated in full, each tagged with the open question it depends on, so `rg OQ-004`
    #: finds every manifest and every code path that would change if it is answered.
    assumptions: tuple[str, ...] = ()
    declined: tuple[DeclinedStrategy, ...] = ()
    #: Present only when a recovery override supplied the evidence. Required by the
    #: spec to be recorded prominently: the manifest has to be able to say why a time
    #: was not read from the file.
    override_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SourceContext:
    """Everything a strategy is allowed to look at.

    Deliberately narrow. There is no filename here and no ``stat`` result, so a strategy
    physically cannot reach for either (INV-12).
    """

    relative_path: str
    sha256: str
    sample_rate: int
    tags: Mapping[str, str]
    frame_rate: FrameRate
    override: SourceTimeOverride | None = None


@dataclass(frozen=True, slots=True)
class _Attempt:
    """One strategy's outcome: evidence, or a reason it declined."""

    evidence: StartEvidence | None = None
    assumptions: tuple[str, ...] = ()
    declined: str | None = None


@dataclass(frozen=True, slots=True)
class _Strategy:
    name: str
    extract: Callable[[SourceContext], _Attempt]


def _recovery_override_offset(context: SourceContext) -> _Attempt:
    override = context.override
    if override is None:
        return _Attempt(declined="no recovery override names this source")
    if override.start_offset_samples is None:
        return _Attempt(declined="the override supplies no start_offset_samples")
    return _Attempt(
        evidence=SessionOffset(
            samples=override.start_offset_samples,
            recording_date=override.recording_date,
        ),
        assumptions=(
            "the configured start_offset_samples is at 48 kHz and relative to session "
            "zero, per the session input contract",
        ),
    )


def _recovery_override_timecode(context: SourceContext) -> _Attempt:
    override = context.override
    if override is None:
        return _Attempt(declined="no recovery override names this source")
    if override.start_timecode is None:
        return _Attempt(declined="the override supplies no start_timecode")

    # Already validated at configuration load, so a failure here would be a bug rather
    # than bad input; it is still checked, because "validated elsewhere" decays.
    parsed = parse_timecode(override.start_timecode, context.frame_rate)
    return _Attempt(
        evidence=TimecodeReference(
            text=override.start_timecode,
            frames=frame_index(parsed),
            frame_rate_label=context.frame_rate.label,
            frame_rate=context.frame_rate.rate,
            drop_frame=context.frame_rate.drop_frame,
            recording_date=override.recording_date,
        ),
        assumptions=(
            "the operator-supplied timecode is in the configured frame rate "
            f"{context.frame_rate.label}",
        ),
    )


def _bwf_time_reference(context: SourceContext) -> _Attempt:
    raw = context.tags.get(_TIME_REFERENCE_TAG)
    if raw is None:
        return _Attempt(declined=f"no {_TIME_REFERENCE_TAG} tag (no bext chunk, or no reference)")
    try:
        samples = int(raw)
    except ValueError:
        return _Attempt(declined=f"the {_TIME_REFERENCE_TAG} tag {raw!r} is not an integer")
    if samples < 0:
        return _Attempt(declined=f"the {_TIME_REFERENCE_TAG} tag {raw!r} is negative")

    return _Attempt(
        evidence=BwfSampleReference(
            samples=samples,
            sample_rate=context.sample_rate,
            origination_date=_tag_date(context.tags.get(_DATE_TAG)),
        ),
        assumptions=(
            "OQ-001: FFprobe's time_reference tag carries the BWF bext sample reference",
            "OQ-004: it counts samples since midnight at the file's own sample rate, "
            "and is used as an integer without being rounded through a frame count",
        ),
    )


def _timecode_tag(context: SourceContext) -> _Attempt:
    raw = context.tags.get(_TIMECODE_TAG)
    if raw is None:
        return _Attempt(declined=f"no {_TIMECODE_TAG} tag")
    try:
        parsed = parse_timecode(raw, context.frame_rate)
    except TimecodeError as exc:
        return _Attempt(declined=f"the {_TIMECODE_TAG} tag {raw!r} is unusable: {exc}")

    return _Attempt(
        evidence=TimecodeReference(
            text=raw,
            frames=frame_index(parsed),
            frame_rate_label=context.frame_rate.label,
            frame_rate=context.frame_rate.rate,
            drop_frame=context.frame_rate.drop_frame,
            recording_date=_tag_date(context.tags.get(_DATE_TAG)),
        ),
        assumptions=(
            "OQ-001: the timecode tag is the recorder's start timecode",
            f"it is in the configured frame rate {context.frame_rate.label}; the tag "
            "itself carries no rate",
        ),
    )


#: Order is the contract. An operator's explicit evidence outranks the file's, because
#: an override exists precisely for the case where the file's metadata is wrong; among
#: the file's own, a sample reference outranks a frame-quantized timecode because it is
#: finer and needs no configured rate.
_STRATEGIES: Final[tuple[_Strategy, ...]] = (
    _Strategy("recovery_override_offset", _recovery_override_offset),
    _Strategy("recovery_override_timecode", _recovery_override_timecode),
    _Strategy("bwf_time_reference", _bwf_time_reference),
    _Strategy("timecode_tag", _timecode_tag),
)


def strategy_names() -> tuple[str, ...]:
    """Every strategy, in the order it is tried."""
    return tuple(strategy.name for strategy in _STRATEGIES)


def extract_start_time(context: SourceContext) -> StartTime:
    """Run the chain and return the first evidence found.

    Raises:
        RecoveryError: if an override names this source but its configured SHA-256 does
            not match the file. Applying it anyway would attach a field-log time to the
            wrong recording.
        TimecodeError: if no strategy found anything. INV-12: a source with no reliable
            timing has no timing, and the message says which override would supply it.
    """
    _verify_override_hash(context)

    declined: list[DeclinedStrategy] = []
    for strategy in _STRATEGIES:
        attempt = strategy.extract(context)
        if attempt.evidence is None:
            declined.append(
                DeclinedStrategy(name=strategy.name, reason=attempt.declined or "found nothing")
            )
            continue
        override = context.override
        return StartTime(
            strategy=strategy.name,
            evidence=attempt.evidence,
            assumptions=attempt.assumptions,
            declined=tuple(declined),
            override_reason=(
                override.reason
                if override is not None and strategy.name.startswith("recovery_")
                else None
            ),
        )

    raise _no_timing_error(context, tuple(declined))


def _verify_override_hash(context: SourceContext) -> None:
    override = context.override
    if override is None or override.sha256 is None:
        return
    if override.sha256 != context.sha256:
        message = (
            f"recovery.source_time_overrides[{context.relative_path!r}] expects sha256 "
            f"{override.sha256}, but that file hashes to {context.sha256}. The override "
            f"was written for different bytes; check the path, or update the hash if the "
            f"file was legitimately replaced."
        )
        raise RecoveryError(message)


def _no_timing_error(
    context: SourceContext, declined: tuple[DeclinedStrategy, ...]
) -> TimecodeError:
    """The INV-12 diagnostic. Actionable, because the alternative is a guess."""
    tried = "\n".join(f"  - {item.name}: {item.reason}" for item in declined)
    message = (
        f"no reliable start time for {context.relative_path}. Every strategy was tried:\n"
        f"{tried}\n"
        f"Timing is never inferred from a filename or a modification time (INV-12). If "
        f"you know when this file starts, record the evidence in session.yaml:\n"
        f"  recovery:\n"
        f"    source_time_overrides:\n"
        f'      "{context.relative_path}":\n'
        f"        sha256: {context.sha256}\n"
        f'        start_timecode: "HH:MM:SS:FF"   # or start_offset_samples, not both\n'
        f'        reason: "where this time came from"'
    )
    return TimecodeError(message)


def _tag_date(raw: str | None) -> dt.date | None:
    """An ISO date from a metadata tag, or nothing.

    A date that will not parse is discarded rather than guessed at. It is a recording
    *day*, not a time, and M2 is where it is reconciled with the rest.
    """
    if not raw:
        return None
    try:
        return dt.date.fromisoformat(raw.strip()[:10])
    except ValueError:
        return None

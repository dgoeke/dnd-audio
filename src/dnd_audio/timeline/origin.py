"""Where session zero is, and which day each chunk belongs to (ADR-0009).

Everything else in this milestone hangs off this module: a chunk's position is its
evidence minus session zero, so an error here moves every sample of every track by the
same amount and nothing downstream can detect it.

Three things happen, in this order, and the order matters.

**Rollover is resolved first**, because a chunk stamped ``00:05:00:00`` is either five
minutes past midnight or five minutes into the next day, and until that is settled it has
no position. Cycles are added in the evidence's own units — samples for a BWF reference,
frames for a timecode — never in seconds (see :mod:`~dnd_audio.timeline.rasterize`).

**Recorded dates beat inference.** When the files state their origination date, the day is
evidence rather than a guess, and INV-12 prefers evidence. Inference is the fallback for
the case M1 actually produces: partial dates, because an ``INFO``/``ISMP`` timecode tag
carries no date at all.

**Session zero comes last**, from the configured origin or from the earliest unwrapped
start. When it is derived, the whole timeline is shifted so that earliest start lands at
zero — which is what keeps a *signed* ``start_offset_samples`` meaningful. The spec permits
a negative offset, and a rule that made "before zero" fatal in every case would delete half
of that field's range; here the offsets form a relative coordinate system whose distances
are exact and only whose origin was unknown.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from dnd_audio.artifacts.manifest import (
    Manifest,
    ManifestSource,
    SessionOffsetRecord,
    StartEvidenceRecord,
    TimecodeRecord,
)
from dnd_audio.artifacts.timeline import SessionZero, TimelineDecision, TimelineNote
from dnd_audio.config import SessionConfig
from dnd_audio.determinism import to_samples
from dnd_audio.errors import TimecodeError
from dnd_audio.timecode import FrameRate, frame_index, parse_frame_rate, parse_timecode
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE
from dnd_audio.timeline.rasterize import (
    absolute_seconds,
    has_mixed_absolute_domains,
    is_absolute,
    relative_seconds,
    session_position,
    timecode_day_discrepancy_seconds,
)
from dnd_audio.timeline.warp import IdentityWarp, TimeWarp

__all__ = [
    "PLAUSIBLE_SPAN_SECONDS",
    "SessionOrigin",
    "SourceStart",
    "determine_origin",
]

#: Beyond this, an inferred session span warns. A session longer than half a day is
#: arithmetically unambiguous — rollover is unique inside one cycle — but implausible
#: enough that an operator should look. It fires no behaviour, only a warning, and what a
#: real session's length actually is remains **OQ-014**.
PLAUSIBLE_SPAN_SECONDS: Final = 12 * 3600


@dataclass(frozen=True, slots=True)
class SourceStart:
    """One selected source, placed."""

    track_id: str
    relative_path: str
    sha256: str
    evidence: StartEvidenceRecord
    #: Whole 24-hour cycles added to unwrap this evidence. Zero for almost everything.
    cycles: int
    #: Position on the session timeline, in canonical samples. Never negative: see the
    #: module docstring's third rule.
    session_start_sample: int


@dataclass(frozen=True, slots=True)
class SessionOrigin:
    """Where zero is, where every source starts, and why."""

    zero: SessionZero
    starts: tuple[SourceStart, ...]
    decisions: tuple[TimelineDecision, ...]
    warnings: tuple[TimelineNote, ...]

    def by_track(self, track_id: str) -> tuple[SourceStart, ...]:
        return tuple(start for start in self.starts if start.track_id == track_id)


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A selected source before its day has been decided."""

    track_id: str
    source: ManifestSource
    evidence: StartEvidenceRecord


def selected_sources(manifest: Manifest) -> tuple[tuple[str, ManifestSource], ...]:
    """Every source the timeline is built from, as ``(track_id, source)``.

    ``role == "selected"`` and nothing else. The manifest deliberately records ignored
    edits, duplicates, and files in unconfigured directories; putting any of them on a
    timeline would attribute audio nobody chose to a speaker.
    """
    return tuple(
        (track.track_id, source)
        for track in manifest.tracks
        for source in track.sources
        if source.role == "selected"
    )


def determine_origin(
    manifest: Manifest, config: SessionConfig, *, warp: TimeWarp | None = None
) -> SessionOrigin:
    """Place every selected source on one timeline.

    Args:
        warp: The future affine drift correction's seam, applied to every placement's
            exact elapsed time before it is quantized. Defaults to the identity, which is
            the only implementation the MVP ships (OQ-006).

    Raises:
        TimecodeError: when the day a chunk belongs to cannot be settled without guessing
            — an ambiguous rollover, a rollover forbidden by ``rollover_policy: reject``,
            or a mixture of evidence domains whose 24-hour cycles differ in a session that
            appears to cross one. Each message names the configuration that would settle
            it, because INV-12 forbids inventing the answer.
    """
    frame_rate = parse_frame_rate(config.timecode.frame_rate)
    time_warp = warp if warp is not None else IdentityWarp()
    candidates = [
        _Candidate(track_id=track_id, source=source, evidence=source.start_time.evidence)
        for track_id, source in selected_sources(manifest)
        if source.start_time is not None
    ]

    decisions: list[TimelineDecision] = []
    warnings: list[TimelineNote] = []

    absolute = [item for item in candidates if is_absolute(item.evidence)]
    relative = [item for item in candidates if not is_absolute(item.evidence)]

    configured_zero = _configured_zero(config, frame_rate)
    cycles = _resolve_cycles(absolute, config, frame_rate, configured_zero, decisions, warnings)

    unwrapped = {
        item.source.relative_path: absolute_seconds(
            item.evidence, frame_rate, cycles=cycles[item.source.relative_path]
        )
        for item in absolute
    }

    zero_seconds, zero_source = _choose_zero(configured_zero, unwrapped)
    zero_domain = _zero_domain(configured_zero, unwrapped, absolute)
    track_of = {item.source.relative_path: item.track_id for item in candidates}
    positions: dict[str, int] = {
        path: session_position(
            seconds,
            zero_seconds,
            CANONICAL_SAMPLE_RATE,
            adjust=lambda elapsed, track=track_of[path]: time_warp.warp(track, elapsed),  # type: ignore[misc]
        )
        for path, seconds in unwrapped.items()
    }
    for item in relative:
        # Already session-relative, so there is nothing to subtract: an offset *is* a
        # position. Quantizing it separately keeps the one-rounding rule intact.
        if not isinstance(item.evidence, SessionOffsetRecord):  # pragma: no cover - shape guard
            message = f"{item.source.relative_path} is not session-relative evidence"
            raise TimecodeError(message)
        positions[item.source.relative_path] = to_samples(
            time_warp.warp(item.track_id, relative_seconds(item.evidence)),
            CANONICAL_SAMPLE_RATE,
        )

    shift = _shift_for(positions, zero_source, decisions)
    starts = tuple(
        SourceStart(
            track_id=item.track_id,
            relative_path=item.source.relative_path,
            sha256=item.source.sha256,
            evidence=item.evidence,
            cycles=cycles.get(item.source.relative_path, 0),
            session_start_sample=positions[item.source.relative_path] + shift,
        )
        for item in sorted(candidates, key=lambda c: c.source.relative_path)
    )

    _warn_about_span(starts, manifest, warnings)
    _warn_about_mixed_domains([item.evidence for item in candidates], frame_rate, warnings)

    return SessionOrigin(
        zero=_zero_record(config, zero_source, zero_domain, zero_seconds, frame_rate, shift=shift),
        starts=starts,
        decisions=tuple(decisions),
        warnings=tuple(warnings),
    )


def _configured_zero(config: SessionConfig, frame_rate: FrameRate) -> Fraction | None:
    """Session zero from ``origin_timecode``, in exact seconds from the day origin.

    ``origin_date`` is required alongside it by the configuration model and is never
    inferred from a date-shaped ``session_id``: the spec forbids that, and the two can
    legitimately differ.
    """
    if config.timecode.origin_timecode is None:
        return None
    parsed = parse_timecode(config.timecode.origin_timecode, frame_rate)
    return Fraction(frame_index(parsed)) / frame_rate.rate


def _resolve_cycles(
    absolute: list[_Candidate],
    config: SessionConfig,
    frame_rate: FrameRate,
    configured_zero: Fraction | None,
    decisions: list[TimelineDecision],
    warnings: list[TimelineNote],
) -> dict[str, int]:
    """How many 24-hour cycles to add to each piece of absolute evidence."""
    if not absolute:
        return {}

    dated = _cycles_from_dates(absolute, config, decisions)
    if dated is not None:
        return dated

    if configured_zero is not None:
        return _cycles_against_a_known_zero(
            absolute, config, frame_rate, configured_zero, decisions
        )

    return _cycles_by_largest_gap(absolute, config, frame_rate, decisions, warnings)


def _cycles_from_dates(
    absolute: list[_Candidate], config: SessionConfig, decisions: list[TimelineDecision]
) -> dict[str, int] | None:
    """Use **operator-asserted** recording dates, when every source has one.

    Evidence beats inference (INV-12) — but only evidence that is worth more than the
    inference. A date read from a *file* is not. `bext.origination_date`/`origination_time`
    carry the receiver's real-time clock, and on 2026-08-03 two receivers' clocks were
    measured **48.7 s apart** while their timecode agreed to under one frame. This function
    applies day differences as whole 24-hour cycles, which is the coarsest unit available:
    two receivers whose clocks straddle midnight would be placed a *day* apart on evidence
    known to be a minute wrong (ADR-0031).

    So only an operator may assign a cycle, via a `recovery.source_time_overrides` entry's
    `recording_date`. The strategy that produced the evidence is what distinguishes them —
    the file's own timecode strategy also records a `recording_date`, descriptively, from
    the same untrustworthy tag.

    Returns ``None`` when any source lacks an asserted date, which is the ordinary case:
    the session then falls through to inference, which reads the counters themselves and
    involves no wall clock at all.

    Day differences are applied as whole cycles. At a fractional non-drop rate a timecode
    cycle is not a calendar day (OQ-015), so this is exact for a BWF reference and rests on
    the jam assumption for a timecode — the same assumption the domains already share.
    """
    dates: dict[str, dt.date] = {}
    for item in absolute:
        date = _asserted_date(item.source)
        if date is None:
            return None
        dates[item.source.relative_path] = date

    reference = config.timecode.origin_date or min(dates.values())
    cycles = {path: (date - reference).days for path, date in dates.items()}
    if any(count < 0 for count in cycles.values()):
        earliest = min(dates.values())
        message = (
            f"timecode.origin_date is {reference.isoformat()}, but a source states an "
            f"earlier origination date of {earliest.isoformat()}. A session cannot begin "
            f"after its own earliest recording; correct origin_date, or remove it to let "
            f"session zero come from the earliest source."
        )
        raise TimecodeError(message, code="origin_after_earliest_source")

    if any(cycles.values()):
        spanned = sorted({date.isoformat() for date in dates.values()})
        decisions.append(
            TimelineDecision(
                code="rollover_from_recorded_dates",
                subject=", ".join(spanned),
                detail=(
                    f"every source states its origination date, so the day each chunk "
                    f"belongs to is evidence rather than an inference; days are counted "
                    f"from {reference.isoformat()}"
                ),
            )
        )
    return cycles


def _cycles_against_a_known_zero(
    absolute: list[_Candidate],
    config: SessionConfig,
    frame_rate: FrameRate,
    zero: Fraction,
    decisions: list[TimelineDecision],
) -> dict[str, int]:
    """Infer forward from a configured origin.

    The rule ADR-0009 states: a single cycle is added exactly when the same-day reading
    would place the chunk before session zero. Under ``reject`` that situation is fatal
    instead, which is the escape for an operator who knows their session did not cross
    midnight and would rather see an error than a guess.
    """
    cycles: dict[str, int] = {}
    for item in absolute:
        path = item.source.relative_path
        same_day = absolute_seconds(item.evidence, frame_rate)
        if same_day >= zero:
            cycles[path] = 0
            continue
        if config.timecode.rollover_policy == "reject":
            message = (
                f"{path} starts before the configured origin "
                f"{config.timecode.origin_timecode} and timecode.rollover_policy is "
                f"'reject', so no midnight rollover is inferred. Either the source belongs "
                f"to the next day — set rollover_policy to 'infer_forward' — or the "
                f"configured origin is later than the session actually started."
            )
            raise TimecodeError(message, code="rollover_rejected")
        cycles[path] = 1
        decisions.append(
            TimelineDecision(
                code="rollover_inferred",
                subject=path,
                detail=(
                    f"its start reads earlier than the configured origin "
                    f"{config.timecode.origin_timecode}, so a single forward 24-hour "
                    f"cycle was added; the alternative reading is that the source predates "
                    f"the configured origin"
                ),
            )
        )
    return cycles


def _cycles_by_largest_gap(
    absolute: list[_Candidate],
    config: SessionConfig,
    frame_rate: FrameRate,
    decisions: list[TimelineDecision],
    warnings: list[TimelineNote],
) -> dict[str, int]:
    """Infer the day boundary from where the sources are *not*.

    With no configured origin, the evidence is a set of points on a 24-hour circle and the
    session is an arc through them. The arc that does not contain the largest gap is the
    session; midnight, if it was crossed, falls inside that gap.

    **This picks the shortest arc, which is a heuristic and not a proof** (OQ-016). Starts
    at 23:00 and 01:00 admit two readings — two hours across midnight, or twenty-two hours
    within one day — and nothing in the evidence excludes the second. Sessions are short,
    so the shortest arc is the right default, and every session that relies on it is warned
    that its days were inferred rather than read. An operator who records `origin_date` and
    `origin_timecode` never reaches this function.

    A *tie* is refused rather than resolved: two gaps of equal size mean two equally good
    readings with nothing to choose between them, which is the case where INV-12's "fail
    with an actionable diagnostic" applies rather than a coin flip.
    """
    cycle = _common_cycle(absolute, frame_rate)
    values = sorted(
        (absolute_seconds(item.evidence, frame_rate), item.source.relative_path)
        for item in absolute
    )
    if len(values) < 2:
        return {path: 0 for _, path in values}

    gaps = [(values[i + 1][0] - values[i][0], i) for i in range(len(values) - 1)]
    wrap_gap = values[0][0] + cycle - values[-1][0]
    widest, cut = max(gaps)

    if wrap_gap > widest:
        # The quiet stretch is the one that already contains midnight: nothing wrapped.
        return {path: 0 for _, path in values}
    if wrap_gap == widest or sum(1 for gap, _ in gaps if gap == widest) > 1:
        message = (
            f"this session's sources are spread around the 24-hour clock with no single "
            f"widest quiet stretch, so which of them fall after midnight is ambiguous "
            f"({len(values)} sources). Set timecode.origin_date and "
            f"timecode.origin_timecode to state where the session begins, or supply a "
            f"recovery override for the sources whose day is in doubt."
        )
        raise TimecodeError(message, code="rollover_ambiguous")

    if config.timecode.rollover_policy == "reject":
        message = (
            f"this session appears to cross midnight — its sources leave a "
            f"{float(wrap_gap) / 3600:.1f}-hour quiet stretch that is not the widest one — "
            f"and timecode.rollover_policy is 'reject', so no rollover is inferred. Set a "
            f"dated origin, or allow inference."
        )
        raise TimecodeError(message, code="rollover_rejected")

    wrapped = {path for _, path in values[: cut + 1]}
    decisions.append(
        TimelineDecision(
            code="rollover_inferred",
            subject=", ".join(sorted(wrapped)),
            detail=(
                "no origin was configured; the widest quiet stretch in the sources' "
                "start times is elsewhere, so these fall after midnight and gain one "
                "24-hour cycle"
            ),
        )
    )
    warnings.append(
        TimelineNote(
            code="midnight_rollover_inferred",
            message=(
                "the day each chunk belongs to was inferred from the spread of start "
                "times: the session was taken to be the shortest arc containing every "
                "start (OQ-016). That is a heuristic about how long a session runs, not a "
                "reading forced by the evidence. Recording an origin_date and "
                "origin_timecode in session.yaml turns this inference into evidence."
            ),
        )
    )
    return {path: (1 if path in wrapped else 0) for _, path in values}


def _common_cycle(absolute: list[_Candidate], frame_rate: FrameRate) -> Fraction:
    """The one 24-hour cycle this session's absolute evidence shares, in seconds.

    Raises:
        TimecodeError: when the evidence spans domains whose cycles differ — a BWF
            reference and a timecode at 23.98F or 29.97F, where a timecode day is 86.4
            seconds longer than a calendar one. Unwrapping them against each other would
            need a rule for which day is which, and there is no evidence for one.
    """
    lengths = {
        absolute_seconds(item.evidence, frame_rate, cycles=1)
        - absolute_seconds(item.evidence, frame_rate)
        for item in absolute
    }
    if len(lengths) > 1:
        message = (
            f"this session mixes timing-evidence domains whose 24-hour cycles differ "
            f"({', '.join(f'{float(length):.4f}s' for length in sorted(lengths))}) and no "
            f"origin is configured, so which sources fall after midnight cannot be "
            f"established. At {frame_rate.label} a timecode day is not a calendar day "
            f"(OQ-015). Set timecode.origin_date and timecode.origin_timecode."
        )
        raise TimecodeError(message, code="rollover_ambiguous")
    return lengths.pop()


def _choose_zero(
    configured: Fraction | None, unwrapped: dict[str, Fraction]
) -> tuple[Fraction, str]:
    """Session zero, and which rule produced it."""
    if configured is not None:
        return configured, "configured_origin"
    if unwrapped:
        return min(unwrapped.values()), "earliest_source"
    # Nothing absolute at all: the offsets are their own coordinate system, and zero is
    # wherever the shift puts it.
    return Fraction(0), "earliest_source"


def _shift_for(
    positions: dict[str, int], zero_source: str, decisions: list[TimelineDecision]
) -> int:
    """How far to move the whole timeline so nothing starts before zero.

    Zero with a derived origin, because zero is *defined* as the earliest start and the
    minimum is already there — except when a signed recovery offset reaches below it, which
    is exactly the case the spec's signed field exists for.

    Raises:
        TimecodeError: when the origin was configured and a source still lands before it.
            The operator asserted where zero is; moving their origin silently, or dropping
            the audio, are both worse than saying so.
    """
    if not positions:
        return 0
    earliest = min(positions.values())
    if earliest >= 0:
        return 0

    if zero_source == "configured_origin":
        offenders = sorted(path for path, value in positions.items() if value < 0)
        message = (
            f"{', '.join(offenders)} would start {abs(earliest)} samples before the "
            f"configured session origin. Audio is never truncated to fit an origin: either "
            f"the recovery offset's sign is wrong, or timecode.origin_timecode is later "
            f"than the session actually began."
        )
        raise TimecodeError(message, code="audio_before_session_zero")

    decisions.append(
        TimelineDecision(
            code="timeline_shifted_to_earliest_source",
            subject=str(abs(earliest)),
            detail=(
                "no origin was configured, so session zero is the earliest valid source "
                "start; a signed recovery offset reached below it and the whole timeline "
                "was shifted by this many samples. Every distance between sources is "
                "unchanged — only the origin moved, which is the one thing no evidence "
                "fixed."
            ),
        )
    )
    return -earliest


def _zero_domain(
    configured: Fraction | None,
    unwrapped: dict[str, Fraction],
    absolute: list[_Candidate],
) -> str:
    """Which coordinate system session zero's origin belongs to.

    Read from the evidence that actually produced zero rather than assumed. A derived zero
    inherits the domain of the earliest source, because that source's origin is the one
    every position is measured against — and the two absolute domains' 24-hour cycles are
    not the same length at a fractional non-drop rate (OQ-015).

    Neither domain's origin is real midnight. A BWF reference counts from the recorder's
    own timecode origin (OQ-004, ADR-0031), and a timecode's ``00:00:00:00`` is 86.4 seconds
    from a calendar day at 23.98F and 29.97F.
    """
    if configured is not None:
        return "timecode"
    if not unwrapped:
        return "relative"
    earliest = min(unwrapped, key=lambda path: unwrapped[path])
    for item in absolute:
        if item.source.relative_path == earliest:
            return "timecode" if isinstance(item.evidence, TimecodeRecord) else "recorder_epoch"
    return "recorder_epoch"


def _zero_record(
    config: SessionConfig,
    zero_source: str,
    zero_domain: str,
    zero_seconds: Fraction,
    frame_rate: FrameRate,
    *,
    shift: int,
) -> SessionZero:
    """Where zero is, as the artifact records it.

    ``shift`` is subtracted from the recorded origin. When a signed offset reached below
    the earliest absolute source the whole timeline moved later by that much, which means
    session sample 0 is now *earlier* in its domain than the source that used to define it
    — and recording the unshifted origin would declare a position that sample 0 does not
    have. Every mapping from a session sample back into the domain would then be wrong by
    the shift, in a way nothing downstream could detect.
    """
    if zero_source == "configured_origin":
        # A configured origin is never shifted: `_shift_for` refuses rather than moving an
        # origin the operator stated, so reaching here with a shift would mean those two
        # functions had drifted apart. Asserted rather than silently subtracted, because
        # `- shift` on a value that is always zero is arithmetic nothing can test.
        if shift:  # pragma: no cover - `_shift_for` raises before this is reachable
            message = (
                f"a configured origin was shifted by {shift} samples, which cannot happen: "
                f"audio before a stated origin is fatal, not something to move the origin for"
            )
            raise TimecodeError(message, code="audio_before_session_zero")
        return SessionZero(
            source="configured_origin",
            domain="timecode",
            origin_date=config.timecode.origin_date,
            origin_timecode=config.timecode.origin_timecode,
            since_domain_origin_samples=to_samples(zero_seconds, CANONICAL_SAMPLE_RATE),
            detail=(
                f"session zero is the configured origin {config.timecode.origin_timecode} "
                f"at {frame_rate.label} on {_day_text(config.timecode.origin_date)}"
            ),
        )
    if zero_domain == "relative":
        return SessionZero(
            source="earliest_source",
            domain="relative",
            origin_date=config.timecode.origin_date,
            detail=(
                "no source carried an absolute time, so session zero is the origin the "
                "recovery offsets are measured from"
            ),
        )
    return SessionZero(
        source="earliest_source",
        domain="timecode" if zero_domain == "timecode" else "recorder_epoch",
        origin_date=config.timecode.origin_date,
        since_domain_origin_samples=to_samples(zero_seconds, CANONICAL_SAMPLE_RATE) - shift,
        detail=("no origin was configured, so session zero is the earliest valid source start"),
    )


def _warn_about_span(
    starts: tuple[SourceStart, ...], manifest: Manifest, warnings: list[TimelineNote]
) -> None:
    """Flag a session long enough to suggest the rollover reading is wrong (OQ-014)."""
    if not starts:
        return
    lengths = {
        source.relative_path: (source.container.sample_count or 0)
        for _, source in selected_sources(manifest)
        if source.container is not None
    }
    span = max(start.session_start_sample + lengths.get(start.relative_path, 0) for start in starts)
    if span <= PLAUSIBLE_SPAN_SECONDS * CANONICAL_SAMPLE_RATE:
        return
    warnings.append(
        TimelineNote(
            code="implausible_session_span",
            message=(
                f"this session spans {span / CANONICAL_SAMPLE_RATE / 3600:.1f} hours. That "
                f"is arithmetically unambiguous but unusual; if a midnight rollover was "
                f"inferred, the other reading is that a source predates session zero "
                f"(OQ-014)."
            ),
        )
    )


def _warn_about_mixed_domains(
    evidence: list[StartEvidenceRecord], frame_rate: FrameRate, warnings: list[TimelineNote]
) -> None:
    """Flag the assumption that a recorder's ``00:00:00:00`` is real midnight (OQ-015)."""
    discrepancy = timecode_day_discrepancy_seconds(frame_rate)
    if discrepancy == 0 or not has_mixed_absolute_domains(evidence):
        return
    warnings.append(
        TimelineNote(
            code="mixed_time_domains",
            message=(
                f"this session mixes BWF sample references, which count from real "
                f"midnight, with timecode tags, which count from the recorder's "
                f"00:00:00:00. At {frame_rate.label} a timecode day is "
                f"{float(discrepancy):+.4f} seconds from a calendar day, so relating the "
                f"two origins assumes the receivers were jammed to real midnight "
                f"(OQ-015)."
            ),
        )
    )


def _day_text(date: dt.date | None) -> str:
    return date.isoformat() if date is not None else "an unstated date"


#: A start-time strategy the *operator* supplied rather than the file. `starttime.py` names
#: its recovery strategies with this prefix, and `StartTimeRecord.strategy` carries the name
#: into the manifest — so no schema change was needed to tell the two halves apart.
_OPERATOR_STRATEGY_PREFIX: Final = "recovery_"


def _asserted_date(source: ManifestSource) -> dt.date | None:
    """The recording date an **operator** stated for this source, if any (ADR-0031).

    Deliberately not "any date this source carries". Both `origination_date` and the
    timecode strategy's `recording_date` come from the receiver's real-time clock, which is
    descriptive and demonstrably wrong across receivers; reading either here is what would
    let a 48.7-second clock disagreement become a 24-hour placement error.
    """
    start = source.start_time
    if start is None or not start.strategy.startswith(_OPERATOR_STRATEGY_PREFIX):
        return None
    date = getattr(start.evidence, "recording_date", None)
    return date if isinstance(date, dt.date) else None

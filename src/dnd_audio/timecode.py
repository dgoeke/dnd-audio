"""DJI frame-rate labels, exact rational rates, and timecode syntax validation.

The spec is explicit that a fractional rate must never become a binary float during
timestamp arithmetic (INV-04), so ``29.97F`` is :class:`~fractions.Fraction`
``30000/1001`` here and stays that way. ``29.97`` appears nowhere in this module except
as a label.

Scope: labels, rates, whether a timecode string is well formed for its rate, and the
exact frame index it denotes.

The boundary that used to be drawn here was wrong. It said converting a timecode to a
position "is M2's", but M1 has to extract a start time from a timecode tag, and counting
frames is arithmetic *on a timecode* rather than placement *on a timeline*. ADR-0006
redraws it: :func:`frame_index` lives here, and M2 owns normalizing recording dates,
inferring midnight rollover, choosing session zero, and rasterizing onto the 48 kHz
grid. A frame index is exact and unit-free; a sample position is neither until a
quantization rule exists, and that rule is M2's to define.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from dnd_audio.errors import TimecodeError

__all__ = [
    "FRAME_RATES",
    "FRAME_RATE_LABELS",
    "FrameRate",
    "Timecode",
    "frame_index",
    "frames_per_day",
    "parse_frame_rate",
    "parse_timecode",
]


@dataclass(frozen=True, slots=True)
class FrameRate:
    """A DJI rate label and the exact rate it denotes."""

    label: str
    rate: Fraction
    drop_frame: bool

    @property
    def frames_per_timecode_second(self) -> int:
        """The frame-index modulus — 30 for 29.97, not 29.

        Timecode counts frames in whole-numbered seconds regardless of the real rate;
        that mismatch is the entire reason drop-frame exists.
        """
        return math.ceil(self.rate)


_RATES: Final = (
    FrameRate("23.98F", Fraction(24000, 1001), drop_frame=False),
    FrameRate("24F", Fraction(24), drop_frame=False),
    FrameRate("25F", Fraction(25), drop_frame=False),
    FrameRate("29.97F", Fraction(30000, 1001), drop_frame=False),
    FrameRate("29.97DF", Fraction(30000, 1001), drop_frame=True),
    FrameRate("30F", Fraction(30), drop_frame=False),
    FrameRate("50F", Fraction(50), drop_frame=False),
    FrameRate("60F", Fraction(60), drop_frame=False),
)

#: Every rate label the configuration accepts, keyed by label.
FRAME_RATES: Final[dict[str, FrameRate]] = {rate.label: rate for rate in _RATES}

#: Declaration order, for the configuration model's literal type and its test.
FRAME_RATE_LABELS: Final[tuple[str, ...]] = tuple(rate.label for rate in _RATES)

# HH:MM:SS:FF, or HH:MM:SS;FF for drop-frame. The ';' is the conventional marker; '.'
# is accepted as the same marker because some tools emit it.
_TIMECODE_PATTERN: Final = re.compile(
    r"^(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})(?P<separator>[:;.])(?P<frames>\d{2})$"
)
_DROP_FRAME_SEPARATORS: Final = frozenset({";", "."})


@dataclass(frozen=True, slots=True)
class Timecode:
    """A validated timecode, still in its own units. M2 converts it to samples."""

    hours: int
    minutes: int
    seconds: int
    frames: int
    frame_rate: FrameRate

    def __str__(self) -> str:
        separator = ";" if self.frame_rate.drop_frame else ":"
        return f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d}{separator}{self.frames:02d}"


def parse_frame_rate(label: str) -> FrameRate:
    """Resolve a DJI rate label to its exact rate.

    Raises:
        TimecodeError: if the label is not one the hardware produces. Guessing at an
            unknown label would put an invented rate at the base of every timestamp.
    """
    try:
        return FRAME_RATES[label]
    except KeyError:
        known = ", ".join(FRAME_RATE_LABELS)
        message = f"unknown frame rate {label!r}; expected one of: {known}"
        raise TimecodeError(message) from None


def parse_timecode(text: str, frame_rate: FrameRate) -> Timecode:
    """Parse and validate a timecode string against the rate it is claimed to be in.

    Rejects, in order: malformed syntax; drop-frame notation at a non-drop rate; a
    field out of range; and a frame number that drop-frame skips. A drop-frame rate
    written with ``:`` is accepted — that spelling is common and unambiguous, whereas
    ``;`` at 30F is a genuine contradiction about what the numbers mean.

    Raises:
        TimecodeError: on any of the above.
    """
    match = _TIMECODE_PATTERN.match(text)
    if match is None:
        message = (
            f"malformed timecode {text!r}; expected HH:MM:SS:FF, or HH:MM:SS;FF for drop-frame"
        )
        raise TimecodeError(message)

    separator = match.group("separator")
    if separator in _DROP_FRAME_SEPARATORS and not frame_rate.drop_frame:
        message = (
            f"timecode {text!r} uses drop-frame notation {separator!r}, "
            f"but the configured rate {frame_rate.label} is non-drop"
        )
        raise TimecodeError(message)

    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    frames = int(match.group("frames"))

    _reject_out_of_range(text, hours, minutes, seconds, frames, frame_rate)

    if frame_rate.drop_frame and _is_dropped(minutes, seconds, frames):
        message = (
            f"timecode {text!r} names a frame that {frame_rate.label} skips: "
            f"frames 00 and 01 do not exist at the start of a minute unless the "
            f"minute is a multiple of ten"
        )
        raise TimecodeError(message)

    return Timecode(
        hours=hours, minutes=minutes, seconds=seconds, frames=frames, frame_rate=frame_rate
    )


def frame_index(timecode: Timecode) -> int:
    """How many frames since midnight this timecode denotes.

    Exact, and an integer at every rate — a frame index counts frames, so no rate is
    fractional in these units. That is why the manifest stores an index and a rational
    rate rather than a sample position: converting to samples at 30000/1001 fps yields
    8008/5 samples per frame, and the rounding rule for that belongs with the rest of
    the timeline arithmetic in M2 (ADR-0006, INV-04).

    Drop-frame is handled by subtracting the labels it never assigns. It drops *labels*,
    not frames: no audio or video is discarded, so the index of a real frame is smaller
    than naive counting suggests, and it is the naive version that would drift by
    3.6 seconds an hour.
    """
    rate = timecode.frame_rate
    nominal = rate.frames_per_timecode_second
    seconds = timecode.hours * 3600 + timecode.minutes * 60 + timecode.seconds
    index = seconds * nominal + timecode.frames

    if not rate.drop_frame:
        return index

    total_minutes = timecode.hours * 60 + timecode.minutes
    dropped_per_minute = nominal // 15  # 2 at 30 fps, 4 at 60 fps
    return index - dropped_per_minute * (total_minutes - total_minutes // 10)


def frames_per_day(frame_rate: FrameRate) -> int:
    """How many frames one full 24-hour timecode cycle contains.

    The number a wrapped timecode must be unwrapped by, and **not** something that can be
    expressed in seconds without knowing the rate (ADR-0009). At 30000/1001 non-drop a day
    is 2 592 000 frames, which is 86 486.4 real seconds — 86.4 seconds longer than a
    calendar day, because non-drop fractional timecode does not track wall time. That is
    the entire reason drop-frame exists, and at 29.97DF the same cycle is 2 589 408 frames
    and 86 399.9136 seconds.

    Adding "a day" in seconds instead would misplace a wrapped chunk by those 86.4 seconds
    and manufacture a large false overlap out of correct evidence.

    This is :func:`frame_index` evaluated at ``24:00:00:00`` — the first label a day does
    not have — expressed directly, because that timecode cannot be parsed.
    """
    nominal = frame_rate.frames_per_timecode_second
    total = nominal * 86400
    if not frame_rate.drop_frame:
        return total
    total_minutes = 24 * 60
    dropped_per_minute = nominal // 15
    return total - dropped_per_minute * (total_minutes - total_minutes // 10)


def _reject_out_of_range(
    text: str, hours: int, minutes: int, seconds: int, frames: int, frame_rate: FrameRate
) -> None:
    limit = frame_rate.frames_per_timecode_second
    for name, value, bound in (
        ("hours", hours, 24),
        ("minutes", minutes, 60),
        ("seconds", seconds, 60),
        ("frames", frames, limit),
    ):
        if value >= bound:
            message = f"timecode {text!r} has {name}={value}, which is not below {bound}"
            raise TimecodeError(message)


def _is_dropped(minutes: int, seconds: int, frames: int) -> bool:
    """Whether this frame number is one drop-frame timecode never assigns.

    Drop-frame skips the labels 00 and 01 at the start of every minute except every
    tenth. It drops labels, not frames — no audio or video is discarded.
    """
    return seconds == 0 and frames < 2 and minutes % 10 != 0
